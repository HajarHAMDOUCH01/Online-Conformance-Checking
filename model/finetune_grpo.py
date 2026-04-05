import copy
import random
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.reachability_graph import construct_reachability_graph

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
    PHASE1_CHECKPOINT = Path("checkpoints/best_model.pt")
    PHASE2_CHECKPOINT = Path("checkpoints/best_model_grpo.pt")
    MODEL_PATH = (
        r"C:\Users\LENONVO\OneDrive\Desktop\graphs\sujet-CRAN\datasets\spesis"
        r"\spesis_reference_model.pnml"
    )

    # GRPO
    K_SAMPLES       = 3        # candidate sequences per noisy prefix
    SAMPLE_TEMP     = 0.8      # sampling temperature  (< 1 = less random)
    KL_BETA         = 0.2     # KL penalty weight vs reference policy
    GRPO_LAMBDA     = 1.0      # weight of GRPO loss in total loss

    # finetuning
    EPOCHS_P2       = 20
    LR_P2           = 1e-5     # lower LR for finetuning
    GRAD_CLIP       = 0.1
    BATCH_SIZE_P2   = 64       

    # generation
    MAX_GEN_LEN     = 10       # max tokens to generate per candidate

# ---------------------------------------------------------------------------
# Petri net helpers
# ---------------------------------------------------------------------------

class PetriNetOracle:
    def __init__(self, net, mi, mf):
        self.net  = net
        self.mi   = mi
        self.mf   = mf
        self.label_to_trans = {
            t.label: t
            for t in net.transitions
            if t.label is not None
        }
        self._enabled_cache = {}
        self._step_cache    = {}

        self._stable_marking_cache = {}   

    def _get_stable_marking(self, marking):
        """Get marking after firing all silent transitions, with caching."""
        key = self._marking_key(marking)
        if key not in self._stable_marking_cache:
            self._stable_marking_cache[key] = self._fire_silent_transitions(marking)
        return self._stable_marking_cache[key]

    def _get_enabled(self, marking) -> set:
        # print("yoooooo : ", type(marking))
        """Return set of enabled transitions using manual arc checking."""
        enabled = set()
        for t in self.net.transitions:
            can_fire = True
            for arc in t.in_arcs:
                if marking.get(arc.source, 0) < arc.weight:
                    can_fire = False
                    break
            if can_fire:
                enabled.add(t)
        return enabled

    def _fire(self, marking, transition: PetriNet.Transition):
        """Fire a transition and return new marking."""
        new_marking = Marking(marking)
        for arc in transition.in_arcs:
            new_marking[arc.source] = new_marking.get(arc.source, 0) - arc.weight
            if new_marking[arc.source] == 0:
                del new_marking[arc.source]
        for arc in transition.out_arcs:
            new_marking[arc.target] = new_marking.get(arc.target, 0) + arc.weight
        return new_marking

    def _marking_key(self, marking):
        return frozenset((p.name, c) for p, c in marking.items())

    def initial_marking(self):
        from pm4py.objects.petri_net.obj import Marking
        return Marking(self.mi)

    def _fire_silent_transitions(self, marking: Marking):
        changed = True
        while changed:
            changed = False
            enabled = self._get_enabled(marking)
            visible = {t for t in enabled if t.label is not None}
            if visible:
                break
            silent = {t for t in enabled if t.label is None}
            if silent:
                t = next(iter(silent))
                marking = self._fire(marking, t)
                changed = True
        return marking
    
    def transition_enabled_from_markings(self, transition: PetriNet.Transition, all_unique_markings: list[Marking]):
        t_enabled_from_markings: list[Marking] = []
        for marking in all_unique_markings:
            enabled_transitions_lables = self.enabled_labels(marking)
            for enabled_transition_label in enabled_transitions_lables:
                # print("aaaaaaaaaaa : ", type(enabled_transition))
                if transition.label == enabled_transition_label:
                    t_enabled_from_markings.append(marking)
                    break
        return t_enabled_from_markings

    def enabled_labels(self, marking: Marking):
        key = self._marking_key(marking)
        if key not in self._enabled_cache:
            m = self._get_stable_marking(marking)
            enabled = self._get_enabled(m)
            self._enabled_cache[key] = frozenset(
                t.label for t in enabled if t.label is not None
            )
        return self._enabled_cache[key]

    def step(self, marking, activity_label: str):
        key = (self._marking_key(marking), activity_label)
        if key not in self._step_cache:
            m = self._get_stable_marking(marking)
            trans = self.label_to_trans.get(activity_label)
            if trans is None:
                self._step_cache[key] = (marking, False)
            else:
                enabled = self._get_enabled(m)
                if trans not in enabled:
                    self._step_cache[key] = (marking, False)
                else:
                    new_m = self._fire(m, trans)
                    self._step_cache[key] = (new_m, True)
        result_marking, success = self._step_cache[key]
        return Marking(result_marking), success
    
def prewarm_oracle(oracle: PetriNetOracle):
    from collections import deque

    visited = set()
    queue = deque([Marking(oracle.mi)])
    all_markings = []
    
    while queue:
        marking = queue.popleft()
        key = oracle._marking_key(marking)
        if key in visited:
            continue
        visited.add(key)
        all_markings.append(marking)
        m = oracle._fire_silent_transitions(Marking(marking))
        
        # cache stable marking
        oracle._stable_marking_cache[key] = m
        
        enabled = oracle._get_enabled(m)
        
        # cache enabled labels
        oracle._enabled_cache[key] = frozenset(
            t.label for t in enabled if t.label is not None
        )

        for trans in enabled:
            new_marking = oracle._fire(Marking(m), trans)
            
            # cache step result
            step_key = (key, trans.label)
            oracle._step_cache[step_key] = (new_marking, True)
            
            queue.append(new_marking)
    return all_markings
    

    print(f"  oracle cache pre-warmed: {len(visited)} unique markings")
    print(f"  enabled_cache entries : {len(oracle._enabled_cache)}")
    print(f"  step_cache entries    : {len(oracle._step_cache)}")
    print(f"  stable_marking_cache  : {len(oracle._stable_marking_cache)}")
# ---------------------------------------------------------------------------
# Token-level reward
# ---------------------------------------------------------------------------

# def compute_sequence_reward_based_on_cost(
#         generated_labels: list[str],
#         optimal_cost: int, # ground truth alignement cost
#         oracle: PetriNetOracle, # the reference model,
#         cost_per_invalid: int,
# ) -> float:
#     marking_pred = oracle.initial_marking() 
#     generated_cost = 0
#     count_invalid = 0
#     for t, token in enumerate(generated_labels): 
#         enabled_set_pred = oracle.enabled_labels(marking_pred) 
#         if token not in enabled_set_pred:
#             generated_cost += cost_per_invalid
#             count_invalid += 1

def compute_predicted_alignement_reward_manual_steps(
    all_unique_markings,
    generated_labels: list[str],
    ground_truth_labels: list[str],
    optimal_cost: int,
    oracle: PetriNetOracle,
    cost_per_invalid: float = 1.0,
    cost_penalty_scale: float = 0.1
) -> float:
    marking_gt   = oracle.initial_marking() 
    init = generated_labels[0]
    # print(init)
    if init == "ER Registration":
        marking_pred = oracle.initial_marking()
    else:
        return -2.0
    generated_cost = 0
    validity_rewards   = []
    kept_marking_gt = None

    for t, token in enumerate(generated_labels): 
        enabled_set_pred = oracle.enabled_labels(marking_pred) 
        # enabled_transitions = oracle._get_enabled(marking_pred)
        if kept_marking_gt is not None:
            if marking_pred == kept_marking_gt: 
                    marking_gt = kept_marking_gt 
                    kept_marking_gt = None
        if token not in enabled_set_pred:
            # markings where token t is enabled 
            markings_distribution = oracle.transition_enabled_from_markings(oracle.label_to_trans.get(token), all_unique_markings)
            estimated_marking = markings_distribution[0] # to do : fix 
            marking_pred = None
            marking_pred = estimated_marking
            validity_rewards.append(-1.0)
            generated_cost += cost_per_invalid

        else:
            gt_transition = ground_truth_labels[t] if t < len(ground_truth_labels) else 'end' 
            marking_pred, _ = oracle.step(marking_pred, token) 
            if gt_transition != 'end': 
                marking_gt, _ = oracle.step(marking_gt, gt_transition) 
            if gt_transition == 'end': 
                validity_rewards.append(0.0) 

            elif t < len(ground_truth_labels) and marking_gt == marking_pred:
                validity_rewards.append(1.0)

            else:
                validity_rewards.append(0.0)
                generated_cost += cost_per_invalid
                if marking_gt != marking_pred: 
                    kept_marking_gt = marking_gt

    avg_validity = sum(validity_rewards) / max(len(validity_rewards), 1)
    cost_difference = generated_cost - optimal_cost
    if cost_difference > 0:
        cost_penalty = -cost_penalty_scale * cost_difference
    else:
        cost_penalty = 0.0

    if generated_cost == optimal_cost:
        cost_penalty +=0.1
    
    result = avg_validity + cost_penalty
    return max(result, -2.0)

# ---------------------------------------------------------------------------
# Stochastic decoding with log-probabilities
# ---------------------------------------------------------------------------

def sample_candidates(
    model: PrefixConformanceModel,
    z: torch.Tensor,               # [1, d_model]  single sample
    K: int,
    max_len: int, ## to do : remove 
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
        log_probs_sum.append(sum(lp_list)) # likelihood of generating the seq

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
    all_unique_markings,
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
    
    # Ground truth labels (for reward computation only)
    gt_labels = [
        inv_vocab[i.item()]
        for i in aligned[0]
        if i.item() not in [model.pad_idx, model.decoder.bos_idx, model.decoder.eos_idx]
    ]
    
    with torch.no_grad():
        candidates, old_log_probs = sample_candidates(
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
        # old_log_probs: [K] - these are from the model before update
    
    rewards = []
    for cand_indices in candidates:
        skip = {model.pad_idx, model.decoder.bos_idx, model.decoder.eos_idx}
        cand_labels = [
            inv_vocab.get(i, "<UNK>")
            for i in cand_indices
            if i not in skip
        ]
        # print("in cand_idices in loss : ", len(cand_indices))
        r = compute_predicted_alignement_reward_manual_steps(
            all_unique_markings,
            generated_labels=cand_labels,
            ground_truth_labels=gt_labels,
            optimal_cost=optimal_cost,
            oracle=oracle,
            cost_penalty_scale=0.1,
        )
        if r == -1089:
            continue
        rewards.append(r)
    
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)  # [K]
    
    r_mean = rewards_t.mean()
    r_std = rewards_t.std() + 1e-8
    advantages = (rewards_t - r_mean) / r_std  # [K]
    valid_indices = []
    new_log_probs = []
    valid_candidates = []
    valid_advantages = []
    
    for i, (cand_indices, adv) in enumerate(zip(candidates, advantages)):
        if model.decoder.eos_idx in cand_indices:
            eos_pos = cand_indices.index(model.decoder.eos_idx)
            cand_indices = cand_indices[:eos_pos + 1]
        
        if len(cand_indices) < 2:  # Too short, skip
            continue
        valid_indices.append(i)          
        valid_candidates.append(cand_indices)
        valid_advantages.append(adv)
        # Convert to tensor
        cand_tensor = torch.tensor([cand_indices], dtype=torch.long, device=device)
        
        # Compute log probability using current model (with gradients)
        log_prob_sum = 0.0
        
        # Start with BOS
        decoder_input = torch.tensor([[model.decoder.bos_idx]], device=device)
        
        for t, token in enumerate(cand_indices):
            # Get logits for next token
            logits = model.decoder(z_noisy, decoder_input)  # [1, t+1, vocab_size]
            logits_last = logits[0, -1, :]  # [vocab_size]
            
            # Get probability of the actual token
            probs = F.softmax(logits_last, dim=-1)
            token_prob = probs[token]
            log_prob = torch.log(token_prob + 1e-8)
            
            log_prob_sum = log_prob_sum + log_prob
            
            # Append token for next step
            decoder_input = torch.cat([
                decoder_input,
                torch.tensor([[token]], device=device)
            ], dim=1)
            
            if token == model.decoder.eos_idx:
                break
        
        new_log_probs.append(log_prob_sum)
    
    if not valid_candidates:
        return torch.tensor(0.0, device=device), 0.0, 0.0
    
    # Convert to tensors
    new_log_probs = torch.stack(new_log_probs)  # [K_valid]
    old_log_probs_valid = old_log_probs[valid_indices] 
    advantages_valid = torch.stack(valid_advantages)  # [K_valid]
    # ratio = exp(new_log_p - old_log_p) = π_new / π_old
    ratio = torch.exp(new_log_probs - old_log_probs_valid)  # [K_valid]
    
    epsilon = 0.2  # Clipping range
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    
    # Policy loss (negative because we minimize)
    loss_per_candidate = -torch.min(
        ratio * advantages_valid,
        clipped_ratio * advantages_valid
    )
    policy_loss = loss_per_candidate.mean()
    # Compute log probs using reference model (no gradients)
    with torch.no_grad():
        ref_log_probs = []
        for cand_indices in valid_candidates:
            # Similar computation as above but with ref_model
            log_prob_sum = 0.0
            decoder_input = torch.tensor([[ref_model.decoder.bos_idx]], device=device)
            
            for token in cand_indices:
                logits = ref_model.decoder(z_noisy, decoder_input)
                logits_last = logits[0, -1, :]
                probs = F.softmax(logits_last, dim=-1)
                log_prob = torch.log(probs[token] + 1e-8)
                log_prob_sum = log_prob_sum + log_prob
                
                decoder_input = torch.cat([
                    decoder_input,
                    torch.tensor([[token]], device=device)
                ], dim=1)
                
                if token == ref_model.decoder.eos_idx:
                    break
            
            ref_log_probs.append(log_prob_sum)
        
        ref_log_probs = torch.stack(ref_log_probs)
    
    # KL divergence: KL(π_θ || π_ref) = π_θ * log(π_θ / π_ref)
    # Approximated by mean of (log π_θ - log π_ref)
    kl_penalty = (new_log_probs - ref_log_probs.detach()).mean()
    total_loss = cfg.GRPO_LAMBDA * policy_loss + cfg.KL_BETA * kl_penalty
    
    return total_loss, rewards_t.mean().item(), kl_penalty.item()

# ---------------------------------------------------------------------------
# Phase 2 training epoch
# ---------------------------------------------------------------------------

def grpo_train_epoch(
    all_unique_markings,
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
    """
    total_loss, total_reward, total_kl, count = 0.0, 0.0, 0.0, 0

    for batch_idx, batch in enumerate(loader):
        noisy = batch['noisy_padded'].to(device)
        aligned = batch['aligned_padded'].to(device)
        costs = batch['costs'].to(device)

        # process each sample in the batch individually
        batch_loss = torch.tensor(0.0, device=device)
        for i in range(noisy.size(0)):

            loss, reward, kl = grpo_loss_one_sample(
                all_unique_markings,
                model, ref_model,
                noisy[i:i+1], aligned[i:i+1], costs[i].item(),  
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
        # if batch_idx % 50 == 0:
        print(f"    batch {batch_idx}/{len(loader)}  reward={total_reward/max(count,1):.3f}", flush=True)
    n = max(count, 1)
    return total_loss / max(len(loader), 1), total_reward / n, total_kl / n

# ---------------------------------------------------------------------------
# Phase 2 validation — reward on held-out set
# ---------------------------------------------------------------------------

@torch.no_grad()
def grpo_validate(
    all_unique_markings,
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
            
            r = compute_predicted_alignement_reward_manual_steps(
                all_unique_markings,
                generated_labels=pred_labels,
                ground_truth_labels=gt_labels,
                optimal_cost=costs[i].item(), 
                oracle=oracle,
                cost_penalty_scale=0.1,
            )
            if r == -1089:
                return -1089
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
    print("Pre-warming oracle cache...")
    unique_markings = prewarm_oracle(oracle)

    # print()
    # print(unique_markings[0])
    # print()
    # print(type(unique_markings[0]))
    # print()
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

    # ── what to freeze or not to (encoder - decoder) ────────────────────────────
    for p in model.encoder.parameters():
        p.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable parameters : {trainable:,}")

    # ── optimizer ─────────────────────────────────────────────────────────
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
    # ckpt_continue = torch.load(cfg.PHASE2_CHECKPOINT, map_location=device)
    # model.load_state_dict(ckpt_continue["model"])
    # print(f"\n{ckpt_continue["valid_loss"]}\n")
    for epoch in range(3, cfg.EPOCHS_P2 + 1):

        t_loss, t_reward, t_kl = grpo_train_epoch(
            unique_markings,
            model, ref_model,
            grpo_train_loader, optimizer,
            oracle, inv_vocab, cfg, device,
        )

        v_reward = grpo_validate(
            model, grpo_valid_loader,
            oracle, inv_vocab, cfg, device,
        )
        if v_reward == -1089:
            continue

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