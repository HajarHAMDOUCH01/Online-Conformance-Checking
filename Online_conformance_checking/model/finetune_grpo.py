import copy
import random
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pm4py
from pm4py.objects.petri_net.semantics import ClassicSemantics

from dataset import PrefixConformanceDataset, collate_fn
from model  import PrefixConformanceModel
from train  import (
    Config, set_seed, build_dataloaders,
    qualitative_test, save_checkpoint,
)

# ---------------------------------------------------------------------------
# Phase 2 config (extends Phase 1 Config)
# ---------------------------------------------------------------------------

class GRPOConfig(Config):
    # paths
    PHASE1_CHECKPOINT = Path("/content/best_model.pt")
    PHASE2_CHECKPOINT = Path("/content/best_model_grpo.pt")
    MODEL_PATH = (
        r"/content"
        r"/spesis_reference_model.pnml"
    )
    # GRPO
    K_SAMPLES       = 6        # candidate sequences per noisy prefix
    SAMPLE_TEMP     = 0.8      # sampling temperature  (< 1 = less random)
    KL_BETA         = 0.05     # KL penalty weight vs reference policy
    GRPO_LAMBDA     = 1.0      # weight of GRPO loss in total loss

    # finetuning
    EPOCHS_P2       = 20
    LR_P2           = 5e-5     # lower LR for finetuning
    GRAD_CLIP       = 0.5
    BATCH_SIZE_P2   = 16       # smaller batch — K samples per item is expensive

    # generation
    MAX_GEN_LEN     = 30       # max tokens to generate per candidate


# ---------------------------------------------------------------------------
# Petri net helpers
# ---------------------------------------------------------------------------

class PetriNetOracle:
    """
    Wraps pm4py Petri net semantics for step-by-step token reward computation.
    Handles silent transitions automatically.
    """

    def __init__(self, net, mi, mf):
        self.net       = net
        self.mi        = mi
        self.mf        = mf
        self.semantics = ClassicSemantics()
        self._enabled_cache = {}

        # label → transition lookup (visible transitions only)
        self.label_to_trans = {
            t.label: t
            for t in net.transitions
            if t.label is not None
        }

    def initial_marking(self):
        """Return a fresh copy of the initial marking."""
        return copy.copy(self.mi)

    def _fire_silent_transitions(self, marking):
        """
        Greedily fire silent transitions until no more are enabled
        or a visible transition becomes reachable.
        Prevents silent transitions from blocking visible ones.
        """
        changed = True
        while changed:
            changed = False
            enabled = self.semantics.enabled_transitions(self.net, marking)
            visible = {t for t in enabled if t.label is not None}
            if visible:
                break
            silent = {t for t in enabled if t.label is None}
            if silent:
                t = next(iter(silent))
                marking = self.semantics.execute(t, self.net, marking)
                changed = True
        return marking

    # def enabled_labels(self, marking):
    #     """Return set of activity label strings enabled at this marking."""
    #     marking = self._fire_silent_transitions(marking)
    #     enabled = self.semantics.enabled_transitions(self.net, marking)
    #     return {t.label for t in enabled if t.label is not None}

    def enabled_labels(self, marking):
        key = frozenset((p.name, c) for p, c in marking.items())
        if key not in self._enabled_cache:
            marking = self._fire_silent_transitions(marking)
            enabled = self.semantics.enabled_transitions(self.net, marking)
            self._enabled_cache[key] = {t.label for t in enabled if t.label is not None}
        return self._enabled_cache[key]

    def step(self, marking, activity_label: str):
        """
        Try to fire the transition corresponding to activity_label.
        Returns (new_marking, success).
        If not enabled, returns (marking, False) — marking unchanged.
        """
        marking = self._fire_silent_transitions(marking)
        trans = self.label_to_trans.get(activity_label)
        if trans is None:
            return marking, False
        enabled = self.semantics.enabled_transitions(self.net, marking)
        if trans not in enabled:
            return marking, False
        new_marking = self.semantics.execute(trans, self.net, marking)
        return new_marking, True


# ---------------------------------------------------------------------------
# Token-level reward
# ---------------------------------------------------------------------------

def compute_sequence_reward(
    generated_labels: list[str],
    ground_truth_labels: list[str],
    optimal_cost: int,
    oracle: PetriNetOracle,
    cost_per_invalid: float = 1.0,
    cost_penalty_scale: float = 0.5
) -> float:
    marking   = oracle.initial_marking()
    generated_cost = 0
    validity_rewards   = []

    for t, token in enumerate(generated_labels):
        enabled_set = oracle.enabled_labels(marking)

        if token not in enabled_set:
            validity_rewards.append(-1.0)
            generated_cost += cost_per_invalid
        else:
            # token is enabled — check against ground truth
            if t < len(ground_truth_labels) and token == ground_truth_labels[t]:
                validity_rewards.append(1.0)
                # generated_cost += 0
            else:
                validity_rewards.append(0.0)
                generated_cost += cost_per_invalid
            # advance marking
            marking, _ = oracle.step(marking, token)
    avg_validity = sum(validity_rewards) / max(len(validity_rewards), 1)
    cost_difference = generated_cost - optimal_cost
    if cost_difference > 0:
        cost_penalty = -cost_penalty_scale * cost_difference
    else:
        cost_penalty = 0.0

    if generated_cost == optimal_cost:
        cost_penalty +=0.1
    
    return avg_validity + cost_penalty

# ---------------------------------------------------------------------------
# Stochastic decoding with log-probabilities
# ---------------------------------------------------------------------------

def sample_candidates(
    model: PrefixConformanceModel,
    z: torch.Tensor,               # [1, d_model]  single sample
    K: int,
    max_len: int,
    temperature: float,
    vocab_size: int,
    pad_idx: int,
    eos_idx: int,
    device: torch.device,
) -> tuple[list[list[int]], torch.Tensor]:
    """
    Sample K candidate sequences autoregressively from the decoder.

    Returns
    -------
    candidates    : list of K token-index lists  (variable length, no PAD)
    log_probs_sum : [K]  sum of log-probs for each candidate (for GRPO loss)
    """
    candidates    = []
    log_probs_sum = []

    for _ in range(K):
        generated = [model.decoder.bos_idx] 
        lp_sum      = 0.0
        lp_list     = []

        for _ in range(max_len):
            inp    = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model.decoder(z, inp)          # [1, cur_len, vocab_size]
            logits_last = logits[0, -1, :] / temperature   # [vocab_size]

            probs      = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            log_p      = torch.log(probs[next_token] + 1e-9).item()

            lp_list.append(log_p)
            generated.append(next_token)

            if next_token == model.decoder.eos_idx:
                break

        candidates.append(generated[1:])        # strip BOS
        log_probs_sum.append(sum(lp_list))

    return candidates, torch.tensor(log_probs_sum, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# KL penalty vs reference policy
# ---------------------------------------------------------------------------

def kl_penalty_vs_reference(
    model: PrefixConformanceModel,
    ref_model: PrefixConformanceModel,
    z_noisy: torch.Tensor,          # [1, d_model]
    conforming: torch.Tensor,       # [1, conf_len]
    pad_idx: int,
) -> torch.Tensor:
    """
    KL divergence between current decoder and frozen Phase 1 reference decoder.
    Computed on the ground truth conforming prefix (teacher-forced).
    Prevents the finetuned decoder from drifting too far from Phase 1.
    """
    dec_input = conforming[:, :-1]
    if dec_input.size(1) == 0:
        return torch.tensor(0.0, device=z_noisy.device)

    with torch.no_grad():
        ref_logits = ref_model.decoder(z_noisy, dec_input)     # [1, T-1, V]

    cur_logits = model.decoder(z_noisy, dec_input)             # [1, T-1, V]

    cur_log_probs = F.log_softmax(cur_logits, dim=-1)
    ref_probs     = F.softmax(ref_logits,     dim=-1)

    # KL(current || reference) = sum ref * (log ref - log cur)  — but we want
    # KL(current || reference) to keep current close to ref:
    # = sum current_probs * (log_current - log_ref)
    cur_probs = F.softmax(cur_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    kl = (cur_probs * (cur_log_probs - ref_log_probs)).sum(dim=-1).mean()
    return kl


# ---------------------------------------------------------------------------
# GRPO loss for one sample
# ---------------------------------------------------------------------------

def grpo_loss_one_sample(
    model: PrefixConformanceModel,
    ref_model: PrefixConformanceModel,
    noisy: torch.Tensor,            # [1, noisy_len]
    aligned: torch.Tensor,          # [1, aligned_len]
    optimal_cost: float,            
    oracle: PetriNetOracle,
    inv_vocab: dict[int, str],
    cfg: GRPOConfig,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    
    z_noisy = model.encode(noisy)
    
    # Ground truth labels (without BOS/EOS for comparison)
    gt_labels = [
        inv_vocab[i.item()]
        for i in aligned[0]
        if i.item() not in [model.pad_idx, model.decoder.bos_idx, model.decoder.eos_idx]
    ]
    
    # Sample K candidates
    with torch.no_grad():
        candidates, _ = sample_candidates(
            model, z_noisy,
            K=cfg.K_SAMPLES,
            max_len=cfg.MAX_GEN_LEN,
            temperature=cfg.SAMPLE_TEMP,
            vocab_size=model.decoder.vocab_size,
            pad_idx=model.decoder.pad_idx,
            eos_idx=model.decoder.eos_idx,
            device=device,
        )
    model.train()  

    # ── score each candidate ─────────────────────────────────────────────
    rewards = []
    for cand_indices in candidates:
        skip = {model.pad_idx, model.decoder.bos_idx, model.decoder.eos_idx}
        cand_labels = [
            inv_vocab.get(i, "<UNK>")
            for i in cand_indices
            if i not in skip
        ]
        r = compute_sequence_reward(
                    generated_labels=cand_labels,
                    ground_truth_labels=gt_labels,
                    optimal_cost=optimal_cost, 
                    oracle=oracle,
                    cost_penalty_scale=0.5,
                )        
        rewards.append(r)

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)  # [K]

    # ── group-relative advantage ─────────────────────────────────────────
    r_mean = rewards_t.mean()
    r_std  = rewards_t.std() + 1e-8
    advantages = (rewards_t - r_mean) / r_std                               # [K]

    # ── policy gradient loss ─────────────────────────────────────────────
    # recompute log-probs WITH gradients for each candidate
    pg_loss = torch.tensor(0.0, device=device)

    from torch.nn.utils.rnn import pad_sequence

    valid_candidates = []
    valid_advantages = []
    for cand_indices, adv in zip(candidates, advantages):
        if model.decoder.eos_idx in cand_indices:
            eos_pos = cand_indices.index(model.decoder.eos_idx)
            cand_indices = cand_indices[:eos_pos + 1]
        if len(cand_indices) > 1:
            valid_candidates.append(
                torch.tensor(cand_indices, dtype=torch.long, device=device)
            )
            valid_advantages.append(adv)

    if valid_candidates:
        # pad all K candidates to same length → [K, max_cand_len]
        cand_batch = pad_sequence(valid_candidates, batch_first=True, padding_value=model.pad_idx)
        dec_in_batch  = cand_batch[:, :-1]   # [K, T-1]
        dec_tgt_batch = cand_batch[:, 1:]    # [K, T-1]

        # expand z_noisy for all K candidates → [K, d_model]
        z_expanded = z_noisy.expand(len(valid_candidates), -1)

        # ONE forward pass for all K
        logits_batch = model.decoder(z_expanded, dec_in_batch)   # [K, T-1, V]
        log_p_batch  = F.log_softmax(logits_batch, dim=-1)        # [K, T-1, V]

        # gather chosen token log-probs
        chosen_log_p = log_p_batch.gather(
            2, dec_tgt_batch.unsqueeze(-1)
        ).squeeze(-1)                                             # [K, T-1]

        # mask padding
        pad_mask = (dec_tgt_batch != model.pad_idx).float()       # [K, T-1]
        seq_log_p = (chosen_log_p * pad_mask).sum(dim=-1) / pad_mask.sum(dim=-1).clamp(min=1)
                                                                # [K]

        adv_tensor = torch.stack(valid_advantages)                # [K]
        pg_loss = -(adv_tensor * seq_log_p).mean()
    


    # ── KL penalty ───────────────────────────────────────────────────────
    kl = kl_penalty_vs_reference(
        model, ref_model, z_noisy, aligned, model.pad_idx
    )

    # ── total loss ───────────────────────────────────────────────────────
    total = cfg.GRPO_LAMBDA * pg_loss + cfg.KL_BETA * kl

    return total, rewards_t.mean().item(), kl.item()


# ---------------------------------------------------------------------------
# Phase 2 training epoch
# ---------------------------------------------------------------------------

def grpo_train_epoch(
    model: PrefixConformanceModel,
    ref_model: PrefixConformanceModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    oracle: PetriNetOracle,
    inv_vocab: dict[int, str],
    cfg: GRPOConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    One epoch of GRPO finetuning.
    Processes one sample at a time (batch_size=1 inside the loop)
    because each sample requires K forward passes.
    """
    total_loss, total_reward, total_kl, count = 0.0, 0.0, 0.0, 0

    for batch in loader:
        noisy = batch['noisy_padded'].to(device)
        aligned = batch['aligned_padded'].to(device)
        costs = batch['costs'].to(device)

        # process each sample in the batch individually
        batch_loss = torch.tensor(0.0, device=device)
        for i in range(noisy.size(0)):

            loss, reward, kl = grpo_loss_one_sample(
                model, ref_model,
                noisy[i:i+1], aligned[i:i+1], costs[i].item(),  # ← cost added
                oracle, inv_vocab, cfg, device
            )
            batch_loss   = batch_loss + loss
            total_reward += reward
            total_kl     += kl
            count        += 1

        batch_loss = batch_loss / noisy.size(0)

        optimizer.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        total_loss += batch_loss.item()

    n = max(count, 1)
    return total_loss / max(len(loader), 1), total_reward / n, total_kl / n


# ---------------------------------------------------------------------------
# Phase 2 validation — reward on held-out set
# ---------------------------------------------------------------------------

@torch.no_grad()
def grpo_validate(
    model: PrefixConformanceModel,
    loader: DataLoader,
    oracle: PetriNetOracle,
    inv_vocab: dict[int, str],
    cfg: GRPOConfig,
    device: torch.device,
) -> float:
    model.eval()
    total_reward, count = 0.0, 0
    
    for batch in loader:
        noisy = batch['noisy_padded'].to(device)
        aligned = batch['aligned_padded'].to(device)
        costs = batch['costs'].to(device)
        
        for i in range(noisy.size(0)):
            # Greedy alignment
            pred = model.align(
                noisy[i:i+1],
                max_len=cfg.MAX_GEN_LEN,
                eos_idx=model.decoder.eos_idx
            )
            
            gt_labels = [
                inv_vocab[j.item()]
                for j in aligned[i]
                if j.item() not in [model.pad_idx, model.decoder.bos_idx, model.decoder.eos_idx]
            ]
            pred_labels = [
                inv_vocab.get(j.item(), "<UNK>")
                for j in pred[0]
                if j.item() not in [model.pad_idx, model.decoder.bos_idx, model.decoder.eos_idx]
            ]
            
            r = compute_sequence_reward(
                generated_labels=pred_labels,
                ground_truth_labels=gt_labels,
                optimal_cost=costs[i].item(), 
                oracle=oracle,
                cost_penalty_scale=0.5,
            )
            total_reward += r
            count += 1
    
    return total_reward / max(count, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg    = GRPOConfig()
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    print(f"Device : {device}")

    # ── load dataset ──────────────────────────────────────────────────────
    print("\nLoading dataset...")
    dataset = PrefixConformanceDataset.load(cfg.DATASET_PATH)
    print(f"  total pairs : {len(dataset):,}  |  vocab size : {dataset.vocab_size}")

    inv_vocab = {v: k for k, v in dataset.vocab.items()}

    # ── dataloaders ───────────────────────────────────────────────────────
    print("\nBuilding splits...")
    train_loader, valid_loader, test_loader = build_dataloaders(cfg, dataset)

    # use smaller batch for GRPO (K samples per item is expensive)
    grpo_collate     = partial(collate_fn, pad_idx=dataset.pad_idx)
    grpo_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=cfg.BATCH_SIZE_P2,
        shuffle=True,
        collate_fn=grpo_collate,
        drop_last=True,
    )
    grpo_valid_loader = DataLoader(
        valid_loader.dataset,
        batch_size=cfg.BATCH_SIZE_P2,
        shuffle=False,
        collate_fn=grpo_collate,
    )

    # ── load Petri net ────────────────────────────────────────────────────
    print("\nLoading Petri net model...")
    net, mi, mf = pm4py.read_pnml(cfg.MODEL_PATH)
    oracle      = PetriNetOracle(net, mi, mf)
    print(f"  transitions : {len(net.transitions)}  |  places : {len(net.places)}")

    # ── build model ───────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = PrefixConformanceModel(
        vocab_size=dataset.vocab_size,
        d_model=cfg.D_MODEL,
        nhead=cfg.NHEAD,
        num_encoder_layers=cfg.NUM_ENC_LAYERS,
        num_decoder_layers=cfg.NUM_DEC_LAYERS,
        dim_feedforward=cfg.DIM_FEEDFORWARD,
        dropout=cfg.DROPOUT,
        pad_idx=dataset.pad_idx,
        bos_idx=dataset.bos_idx,
        eos_idx=dataset.eos_idx,
    ).to(device)

    # ── load Phase 1 checkpoint ───────────────────────────────────────────
    print(f"\nLoading Phase 1 checkpoint from {cfg.PHASE1_CHECKPOINT} ...")
    ckpt = torch.load(cfg.PHASE1_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"  Phase 1 best valid_loss = {ckpt['valid_loss']:.4f}")

    # ── frozen reference policy (Phase 1 weights, never updated) ─────────
    ref_model = copy.deepcopy(model).to(device)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    # ── freeze encoder — only finetune decoder ────────────────────────────
    for p in model.encoder.parameters():
        p.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable parameters (decoder only) : {trainable:,}")

    # ── optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR_P2,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # ── Phase 2 training loop ─────────────────────────────────────────────
    print(f"\nPhase 2 GRPO finetuning for {cfg.EPOCHS_P2} epochs...\n")
    print(f"{'Epoch':>6}  {'Loss':>10}  {'Train Rew':>10}  {'Valid Rew':>10}  {'KL':>8}")
    print("-" * 55)

    best_valid_reward = -float("inf")

    for epoch in range(1, cfg.EPOCHS_P2 + 1):

        t_loss, t_reward, t_kl = grpo_train_epoch(
            model, ref_model,
            grpo_train_loader, optimizer,
            oracle, inv_vocab, cfg, device,
        )

        v_reward = grpo_validate(
            model, grpo_valid_loader,
            oracle, inv_vocab, cfg, device,
        )

        print(
            f"{epoch:>6}  {t_loss:>10.4f}  {t_reward:>10.4f}  "
            f"{v_reward:>10.4f}  {t_kl:>8.4f}"
        )

        # save best by validation reward (higher = better)
        if v_reward > best_valid_reward:
            best_valid_reward = v_reward
            save_checkpoint(model, optimizer, epoch, -v_reward, cfg.PHASE2_CHECKPOINT)
            print(f"         ↑ new best model saved  (valid_reward={best_valid_reward:.4f})")

    # ── qualitative test ──────────────────────────────────────────────────
    print(f"\nLoading best Phase 2 model for test evaluation...")
    ckpt = torch.load(cfg.PHASE2_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model"])

    qualitative_test(
        model, test_loader, dataset,
        num_samples=cfg.NUM_TEST_SAMPLES,
        device=device,
    )

if __name__ == "__main__":
    main()