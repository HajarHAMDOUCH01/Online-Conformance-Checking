"""
adds disabled transitions geometry constraint

Usage:
    python train_v3.py --n 16
    python train_v3.py --n 32
    python train_v3.py --n 64
    python train_v3.py --n 8

The disabled geometry loss enforces R_t @ emb(m) ≈ emb(m) for every (m, t)
pair where t is not enabled at m.
"""

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import PetriNetAlignmentPredictor

from reachability_graph import (
    reachability_tensor,
    transition_to_enabled_markings,
    t_name_to_idx,
    num_m, num_t,
)
from dataset import (
    X_src_train, X_tgt_train, y_alphas_train,
    X_src_test,  X_tgt_test,
)

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--n',                type=int,   default=16,
                    help='SO(n) embedding dimension (try 8, 16, 32, 64)')
parser.add_argument('--lambda_disabled',  type=float, default=1.0,
                    help='Weight on disabled geometry term')
parser.add_argument('--phase1_epochs',    type=int,   default=200)
parser.add_argument('--phase2_epochs',    type=int,   default=150)
parser.add_argument('--phase3_epochs',    type=int,   default=100)
args = parser.parse_args()

N_DIM            = args.n
LAMBDA_DISABLED  = args.lambda_disabled
BATCH_SIZE       = 64
device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")
print(f"n={N_DIM}  lambda_disabled={LAMBDA_DISABLED}")

model = PetriNetAlignmentPredictor(reachability_tensor, n=N_DIM).to(device)
print(f"Net: {num_m} markings, {num_t} transitions")
print(f"Params/transition: model1={num_m**2}  model_v3={N_DIM*(N_DIM-1)//2}")

X_src_train    = X_src_train.to(device)
X_tgt_train    = X_tgt_train.to(device)
X_src_test     = X_src_test.to(device)
X_tgt_test     = X_tgt_test.to(device)
y_alphas_train = [y.to(device) for y in y_alphas_train]


# ── Dataset + collate ─────────────────────────────────────────────────────────

class AlignmentDataset(Dataset):
    def __init__(self, X_src, X_tgt, y_alphas):
        self.X_src    = X_src
        self.X_tgt    = X_tgt
        self.y_alphas = y_alphas

    def __len__(self):
        return len(self.X_src)

    def __getitem__(self, idx):
        return self.X_src[idx], self.X_tgt[idx], self.y_alphas[idx]


def collate_fn(batch):
    v_srcs, v_tgts, seqs = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], device=seqs[0].device)
    L_max   = lengths.max().item()

    targets = torch.full((len(batch), L_max), fill_value=-1,
                         dtype=torch.long, device=seqs[0].device)
    mask    = torch.zeros(len(batch), L_max, dtype=torch.bool,
                          device=seqs[0].device)

    for i, seq in enumerate(seqs):
        L = seq.size(0)
        targets[i, :L] = seq.argmax(dim=-1)
        mask[i, :L]    = True

    return (
        torch.stack(v_srcs),
        torch.stack(v_tgts),
        targets,
        mask,
        lengths,
    )


train_dataset = AlignmentDataset(X_src_train, X_tgt_train, y_alphas_train)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                           shuffle=True, collate_fn=collate_fn)


# ── Edge list (enabled) + disabled pairs ─────────────────────────────────────

def build_edge_list(transition_to_enabled_markings, t_name_to_idx):
    edges = []
    for t_name, marking_pairs in transition_to_enabled_markings.items():
        t_idx = t_name_to_idx[t_name]
        for pair in marking_pairs:
            edges.append((pair['from_idx'], t_idx, pair['to_idx']))
    return edges

edge_list = build_edge_list(transition_to_enabled_markings, t_name_to_idx)

# Complement: all (m, t) pairs NOT in the enabled set
enabled_set    = {(src, t) for src, t, dst in edge_list}
disabled_pairs = [
    (m, t)
    for m in range(num_m)
    for t in range(num_t)
    if (m, t) not in enabled_set
]

print(f"Enabled edges:    {len(edge_list)}")
print(f"Disabled pairs:   {len(disabled_pairs)}")
print(f"Total (m,t) pairs: {num_m * num_t}")


# ── Batched forward pass ──────────────────────────────────────────────────────

def forward_batch(model, v_src, v_tgt, lengths, tau, training):
    L_max   = lengths.max().item()
    v_cur   = model.encode(v_src)
    v_tgt_e = model.encode(v_tgt)
    logits_list = []

    for step in range(L_max):
        x      = torch.cat([v_cur, v_tgt_e], dim=-1)
        logits = model.network_step(x)
        logits_list.append(logits)

        active = (step < lengths).to(v_cur.device)
        v_new  = model._rotate(v_cur, logits, tau, training)
        v_cur  = torch.where(active.unsqueeze(-1), v_new, v_cur)

    return logits_list, v_cur


def batch_ce_loss(logits_list, targets, mask):
    logits_seq   = torch.stack(logits_list, dim=0).permute(1, 0, 2)
    flat_logits  = logits_seq.reshape(-1, logits_seq.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask    = mask.reshape(-1)
    return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])


# ── Phase 1: Geometry bootstrap ───────────────────────────────────────────────
print(f"\n── Phase 1: Geometry bootstrap ({args.phase1_epochs} epochs) ──")

for p in model.network_step.parameters():
    p.requires_grad = False

geo_optimizer = optim.Adam([model.emb_weight, model.transition_coeffs], lr=1e-2)

for epoch in range(args.phase1_epochs):
    geo_optimizer.zero_grad()
    loss = model.geometry_loss(edge_list, disabled_pairs, LAMBDA_DISABLED)
    loss.backward()
    geo_optimizer.step()
    if epoch % 40 == 0:
        with torch.no_grad():
            l_en  = model.geometry_loss(edge_list, [], 0.0)
            l_dis = model.geometry_loss([], disabled_pairs, 1.0) if disabled_pairs else torch.tensor(0.)
        print(f"  Epoch {epoch:3d} | Total: {loss.item():.6f} "
              f"| Enabled: {l_en.item():.6f} | Disabled: {l_dis.item():.6f}")

for p in model.network_step.parameters():
    p.requires_grad = True


# ── Phase 2: Policy learning (geometry frozen) ────────────────────────────────
print(f"\n── Phase 2: Policy learning ({args.phase2_epochs} epochs) ──")

model.emb_weight.requires_grad        = False
model.transition_coeffs.requires_grad = False

policy_optimizer = optim.Adam(model.network_step.parameters(), lr=1e-3)


def run_policy_epoch(epoch):
    model.train()
    total = 0.0
    tau   = max(0.3, 1.0 - epoch * 0.003)

    for v_src, v_tgt, targets, mask, lengths in train_loader:
        policy_optimizer.zero_grad()
        logits_list, _ = forward_batch(model, v_src, v_tgt, lengths, tau, training=True)
        loss = batch_ce_loss(logits_list, targets, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.network_step.parameters(), 1.0)
        policy_optimizer.step()
        total += loss.item()

    return total / len(train_loader)


for epoch in range(args.phase2_epochs):
    avg = run_policy_epoch(epoch)
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d} | CE loss: {avg:.6f}")

model.emb_weight.requires_grad        = True
model.transition_coeffs.requires_grad = True


# ── Phase 3: Joint fine-tuning ────────────────────────────────────────────────
print(f"\n── Phase 3: Joint fine-tuning ({args.phase3_epochs} epochs) ──")

joint_optimizer = optim.Adam(model.parameters(), lr=2e-4)


def run_joint_epoch(epoch):
    model.train()
    total = 0.0
    tau   = 0.3

    for v_src, v_tgt, targets, mask, lengths in train_loader:
        joint_optimizer.zero_grad()

        logits_list, v_cur_final = forward_batch(
            model, v_src, v_tgt, lengths, tau, training=True
        )
        loss_ce  = batch_ce_loss(logits_list, targets, mask)

        pred_scores  = model.decode_scores(v_cur_final)
        pred_marking = F.softmax(pred_scores, dim=-1)
        loss_pos     = F.mse_loss(pred_marking, v_tgt)

        loss_geo = model.geometry_loss(edge_list, disabled_pairs, LAMBDA_DISABLED)
        loss     = loss_ce + 0.1 * loss_pos + 0.05 * loss_geo

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        joint_optimizer.step()
        total += loss.item()

    return total / len(train_loader)


for epoch in range(args.phase3_epochs):
    avg = run_joint_epoch(epoch)
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d} | Joint loss: {avg:.6f}")


# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n── Evaluation ──")
model.eval()
successes = 0

with torch.no_grad():
    for i in range(len(X_src_test)):
        v_src_test = X_src_test[i:i+1]
        v_tgt_test = X_tgt_test[i:i+1]

        v_pred, pred_logits = model(v_src_test, v_tgt_test, training=False)

        src_idx  = v_src_test.argmax().item()
        tgt_idx  = v_tgt_test.argmax().item()
        pred_idx = v_pred.argmax().item()
        ok       = pred_idx == tgt_idx
        if ok:
            successes += 1

        steps = pred_logits.size(0)
        print(f"\n--- Test {i} | src={src_idx} tgt={tgt_idx} pred={pred_idx} "
              f"{'✓' if ok else '✗'} ({steps} steps) ---")
        for step in range(steps):
            probs = F.softmax(pred_logits[step], dim=-1)
            top_t = probs.argmax().item()
            if probs[top_t].item() > 0.1:
                print(f"  Step {step+1}: t{top_t+1} (conf={probs[top_t]:.2f})")

print(f"\n=== Accuracy: {successes}/{len(X_src_test)} "
      f"| n={N_DIM}  lambda_disabled={LAMBDA_DISABLED} "
      f"| {num_m/N_DIM:.1f}x compression ===")