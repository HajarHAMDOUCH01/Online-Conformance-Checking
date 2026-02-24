import torch
import torch.optim as optim
import torch.nn.functional as F
from model import PetriNetAlignmentPredictor

from reachability_graph import (
    reachability_tensor,
    transition_to_enabled_markings,   # {t_name: [{'from_idx': int, 'to_idx': int}]}
    t_name_to_idx,                    # {t_name: int}
    num_m, num_t,
)
from dataset import (
    X_src_train, X_tgt_train, y_alphas_train,
    X_src_test,  X_tgt_test,
)
N_DIM = 16

model = PetriNetAlignmentPredictor(reachability_tensor, n=N_DIM)
print(f"Net: {num_m} markings, {num_t} transitions")
print(f"Embedding: {num_m} → {N_DIM}  "
      f"(params/transition: model1={num_m**2}, model2={N_DIM*(N_DIM-1)//2})")

def build_edge_list(transition_to_enabled_markings, t_name_to_idx):
    edges = []
    for t_name, marking_pairs in transition_to_enabled_markings.items():
        t_idx = t_name_to_idx[t_name]
        for pair in marking_pairs:
            edges.append((pair['from_idx'], t_idx, pair['to_idx']))
    return edges

edge_list = build_edge_list(transition_to_enabled_markings, t_name_to_idx)
print(f"Graph edges for geometry pre-training: {len(edge_list)}")


for p in model.network_step.parameters():
    p.requires_grad = False

geo_optimizer = optim.Adam(
    [model.emb_weight, model.transition_coeffs], lr=1e-2
)

for epoch in range(200):
    geo_optimizer.zero_grad()
    loss = model.geometry_loss(edge_list)
    loss.backward()
    geo_optimizer.step()
    if epoch % 40 == 0:
        print(f"  Epoch {epoch:3d} | Geo loss: {loss.item():.6f}")

for p in model.network_step.parameters():
    p.requires_grad = True


model.emb_weight.requires_grad        = False
model.transition_coeffs.requires_grad = False

policy_optimizer = optim.Adam(model.network_step.parameters(), lr=1e-3)


def run_policy_epoch(epoch):
    model.train()
    total = 0.0
    tau = max(0.3, 1.0 - epoch * 0.003)

    for i in range(len(X_src_train)):
        policy_optimizer.zero_grad()

        v_s, v_t = X_src_train[i:i+1], X_tgt_train[i:i+1]
        y_true    = y_alphas_train[i]
        if y_true.dim() == 3:
            y_true = y_true.squeeze(1)

        target_indices = y_true.argmax(dim=-1).long()

        _, pred_seq = model(
            v_s, v_t,
            num_steps=target_indices.size(0),
            training=True, tau=tau
        )

        loss = F.cross_entropy(pred_seq, target_indices)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.network_step.parameters(), 1.0)
        policy_optimizer.step()
        total += loss.item()

    return total / len(X_src_train)


for epoch in range(150):
    avg = run_policy_epoch(epoch)
    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d} | CE loss: {avg:.6f}")

model.emb_weight.requires_grad        = True
model.transition_coeffs.requires_grad = True

joint_optimizer = optim.Adam(model.parameters(), lr=2e-4)


def run_joint_epoch(epoch):
    model.train()
    total = 0.0
    tau = 0.3

    for i in range(len(X_src_train)):
        joint_optimizer.zero_grad()

        v_s, v_t = X_src_train[i:i+1], X_tgt_train[i:i+1]
        y_true    = y_alphas_train[i]
        if y_true.dim() == 3:
            y_true = y_true.squeeze(1)

        target_indices = y_true.argmax(dim=-1).long()

        v_pred, pred_seq = model(
            v_s, v_t,
            num_steps=target_indices.size(0),
            training=True, tau=tau
        )

        loss_ce  = F.cross_entropy(pred_seq, target_indices)
        loss_pos = F.mse_loss(v_pred, v_t)
        # Geometry regularizer keeps rotations from drifting
        loss_geo = model.geometry_loss(edge_list)

        loss = loss_ce + 0.1 * loss_pos + 0.05 * loss_geo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        joint_optimizer.step()
        total += loss.item()

    return total / len(X_src_train)


for epoch in range(100):
    avg = run_joint_epoch(epoch)
    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d} | Joint loss: {avg:.6f}")

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
        ok = pred_idx == tgt_idx
        if ok:
            successes += 1

        steps = pred_logits.size(0)
        print(f"\n--- Test {i} | src={src_idx} tgt={tgt_idx} pred={pred_idx} "
              f"{'true' if ok else 'false'} ({steps} steps) ---")
        for step in range(steps):
            probs = F.softmax(pred_logits[step], dim=-1)
            top_t = probs.argmax().item()
            if probs[top_t].item() > 0.1:
                print(f"  Step {step+1}: t{top_t+1} (conf={probs[top_t]:.2f})")

print(f"\n=== Accuracy: {successes}/{len(X_src_test)} "
      f"| n={N_DIM} vs num_m={num_m} "
      f"({num_m/N_DIM:.1f}x compression) ===")