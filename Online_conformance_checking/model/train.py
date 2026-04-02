import os
import random
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import PrefixConformanceDataset, collate_fn, PADDING_TOKEN
from model import PrefixConformanceModel, NTXentLoss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    # paths
    DATASET_PATH = (
        r"C:\Users\LENONVO\OneDrive\Desktop\graphs\sujet-CRAN\datasets\spesis"
        r"\confo_non_conform_prefixes_playout_dataset.pkl"
    )
    CHECKPOINT_DIR = Path("checkpoints")

    # splits
    TRAIN_RATIO = 0.70
    VALID_RATIO = 0.15
    # TEST_RATIO  = 0.15  (remainder)

    # model
    D_MODEL         = 128
    NHEAD           = 4
    NUM_ENC_LAYERS  = 3
    NUM_DEC_LAYERS  = 3
    DIM_FEEDFORWARD = 256
    DROPOUT         = 0.1

    # training
    BATCH_SIZE      = 64
    EPOCHS          = 100
    LR              = 3e-4
    WEIGHT_DECAY    = 1e-4
    LAMBDA_CONTRAST = 0.5       # weight of contrastive loss
    TEMPERATURE     = 0.07      # NT-Xent temperature
    GRAD_CLIP       = 1.0       # max gradient norm

    # test print
    NUM_TEST_SAMPLES = 20       # how many examples to print at test time

    # misc
    SEED   = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def strip_special_tokens(seq: torch.Tensor, bos_idx: int, eos_idx: int, pad_idx: int) -> torch.Tensor:
    # remove BOS at position 0
    seq = seq[1:] if seq[0] == bos_idx else seq
    # find EOS position and truncate
    eos_positions = (seq == eos_idx).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        seq = seq[:eos_positions[0]]
    # remove any remaining PAD
    seq = seq[seq != pad_idx]
    return seq

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_collate(pad_idx: int):
    return partial(collate_fn, pad_idx=pad_idx)


def build_dataloaders(cfg: Config, dataset: PrefixConformanceDataset):
    n       = len(dataset)
    n_train = int(n * cfg.TRAIN_RATIO)
    n_valid = int(n * cfg.VALID_RATIO)
    n_test  = n - n_train - n_valid

    train_ds, valid_ds, test_ds = random_split(
        dataset,
        [n_train, n_valid, n_test],
        generator=torch.Generator().manual_seed(cfg.SEED),
    )

    collate = make_collate(dataset.pad_idx)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE,
        shuffle=True,  collate_fn=collate, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.BATCH_SIZE,
        shuffle=False, collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,  batch_size=1,            # one sample at a time for printing
        shuffle=True,  collate_fn=collate,
    )

    print(f"  train : {n_train:,}  |  valid : {n_valid:,}  |  test : {n_test:,}")
    return train_loader, valid_loader, test_loader


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    model: PrefixConformanceModel,
    batch: dict,
    contrastive_loss_fn: NTXentLoss,
    lambda_contrast: float,
    pad_idx: int,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    """
    Returns total_loss, recon_loss_item, contrast_loss_item.

    Reconstruction:
        Decoder input  = conforming[:, :-1]   (all but last token)
        Decoder target = conforming[:, 1:]    (all but first token)
        The model reconstructs the conforming prefix from z_noisy.

    Contrastive:
        NT-Xent on (z_noisy, z_conforming).
    """
    noisy      = batch["noisy_padded"].to(device)       # [B, noisy_len]
    conforming = batch["aligned_padded"].to(device)  # [B, conf_len]

    # ── reconstruction: shift conforming by 1 ────────────────────────────
    dec_input  = conforming[:, :-1]     # feed:    [BOS, a1, a2, ..., a_{T-1}]
    dec_target = conforming[:, 1:]      # predict: [a1,  a2, ..., a_T        ]

    # skip batches too short to shift (length-1 conforming sequences)
    if dec_input.size(1) == 0:
        return None, 0.0, 0.0

    z_noisy, z_conforming, logits = model(noisy, dec_input)
    # logits : [B, T-1, vocab_size]

    recon_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        dec_target.reshape(-1),
        ignore_index=pad_idx,
    )

    # ── contrastive ───────────────────────────────────────────────────────
    contrast_loss = contrastive_loss_fn(z_noisy, z_conforming)

    total_loss = recon_loss + lambda_contrast * contrast_loss

    return total_loss, recon_loss.item(), contrast_loss.item()


# ---------------------------------------------------------------------------
# Train / Validate one epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model, loader, optimizer, contrastive_loss_fn, cfg, device
) -> tuple[float, float, float]:
    model.train()
    total, recon_sum, contrast_sum, count = 0.0, 0.0, 0.0, 0

    for batch in loader:
        optimizer.zero_grad()
        loss, r, c = compute_loss(
            model, batch, contrastive_loss_fn,
            cfg.LAMBDA_CONTRAST, model.pad_idx, device,
        )
        if loss is None:
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        total       += loss.item()
        recon_sum   += r
        contrast_sum += c
        count       += 1

    n = max(count, 1)
    return total / n, recon_sum / n, contrast_sum / n


@torch.no_grad()
def validate_epoch(
    model, loader, contrastive_loss_fn, cfg, device
) -> tuple[float, float, float]:
    model.eval()
    total, recon_sum, contrast_sum, count = 0.0, 0.0, 0.0, 0

    for batch in loader:
        loss, r, c = compute_loss(
            model, batch, contrastive_loss_fn,
            cfg.LAMBDA_CONTRAST, model.pad_idx, device,
        )
        if loss is None:
            continue
        total       += loss.item()
        recon_sum   += r
        contrast_sum += c
        count       += 1

    n = max(count, 1)
    return total / n, recon_sum / n, contrast_sum / n


# ---------------------------------------------------------------------------
# Qualitative test evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def qualitative_test(
    model: PrefixConformanceModel,
    test_loader: DataLoader,
    dataset: PrefixConformanceDataset,
    num_samples: int,
    device: torch.device,
):
    """
    For each test sample:
        1. Encode the noisy prefix
        2. Autoregressively generate the aligned prefix
        3. Print: noisy input | predicted alignment | ground truth conforming prefix
    """
    model.eval()
    inv_vocab = {v: k for k, v in dataset.vocab.items()}

    def decode(indices: list[int]) -> list[str]:
        skip = {dataset.pad_idx, dataset.bos_idx, dataset.eos_idx}
        return [inv_vocab.get(i, "<UNK>") for i in indices if i not in skip]

    print("\n" + "=" * 80)
    print("QUALITATIVE TEST EVALUATION")
    print("=" * 80)

    printed = 0
    for batch in test_loader:
        if printed >= num_samples:
            break

        noisy      = batch["noisy_padded"].to(device)       # [1, noisy_len]
        conforming = batch["aligned_padded"].to(device)  # [1, conf_len]

        # greedy decode from noisy encoding
        predicted = model.align(noisy, max_len=conforming.size(1) + 5, eos_idx=dataset.eos_idx)
                                                            # [1, pred_len]

        noisy_acts    = decode(noisy[0].tolist())
        conf_acts     = decode(conforming[0].tolist())
        pred_acts     = decode(predicted[0].tolist())

        # conformance score
        conf_clean = strip_special_tokens(
        conforming[0], dataset.bos_idx, dataset.eos_idx, dataset.pad_idx
        ).unsqueeze(0)  

        score = model.conformance_score(noisy, conf_clean).item()

        print(f"\n── Sample {printed + 1} " + "─" * 60)
        print(f"  NOISY INPUT   : {noisy_acts}")
        print(f"  PREDICTED     : {pred_acts}")
        print(f"  GROUND TRUTH  : {conf_acts}")
        print(f"  CONFORM SCORE : {score:.4f}   (cosine sim in latent space)")

        # simple token accuracy
        min_len  = min(len(pred_acts), len(conf_acts))
        matches  = sum(p == g for p, g in zip(pred_acts[:min_len], conf_acts[:min_len]))
        accuracy = matches / max(len(conf_acts), 1)
        print(f"  TOKEN OVERLAP : {matches}/{len(conf_acts)} = {accuracy:.2%}")

        printed += 1

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, valid_loss, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "valid_loss": valid_loss,
    }, path)


def load_checkpoint(model, optimizer, path: Path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from epoch {ckpt['epoch']}  (valid_loss={ckpt['valid_loss']:.4f})")
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = Config()
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    print(f"Device : {device}")

    # ── load dataset ──────────────────────────────────────────────────────
    print("\nLoading dataset...")
    dataset = PrefixConformanceDataset.load(cfg.DATASET_PATH)
    vocab  =dataset.vocab
    print(f"\n vocab verification : , {dataset.pad_idx}\n {dataset.bos_idx}\n{dataset.eos_idx}\n")
    print()
    print(f"  total pairs : {len(dataset):,}  |  vocab size : {dataset.vocab_size}")

    # ── dataloaders ───────────────────────────────────────────────────────
    print("\nBuilding splits...")
    train_loader, valid_loader, test_loader = build_dataloaders(cfg, dataset)

    # ── model ─────────────────────────────────────────────────────────────
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
        eos_idx=dataset.eos_idx
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable parameters : {n_params:,}")

    # ── optimiser & loss ──────────────────────────────────────────────────
    optimizer           = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler           = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    contrastive_loss_fn = NTXentLoss(temperature=cfg.TEMPERATURE).to(device)

    # ── optional: resume from checkpoint ──────────────────────────────────
    best_ckpt  = cfg.CHECKPOINT_DIR / "best_model.pt"
    start_epoch = 0
    best_valid  = float("inf")

    # ── training loop ─────────────────────────────────────────────────────
    print(f"\nTraining for {cfg.EPOCHS} epochs...\n")
    print(f"{'Epoch':>6}  {'Train':>10}  {'T_rec':>8}  {'T_con':>8}  "
          f"{'Valid':>10}  {'V_rec':>8}  {'V_con':>8}  {'LR':>10}")
    print("-" * 80)

    # ckpt_init = torch.load(best_ckpt, map_location=device)
    # model.load_state_dict(ckpt_init["model"])

    for epoch in range(start_epoch + 1, cfg.EPOCHS + 1):

        t_loss, t_rec, t_con = train_epoch(
            model, train_loader, optimizer, contrastive_loss_fn, cfg, device
        )
        v_loss, v_rec, v_con = validate_epoch(
            model, valid_loader, contrastive_loss_fn, cfg, device
        )

        scheduler.step(v_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:>6}  {t_loss:>10.4f}  {t_rec:>8.4f}  {t_con:>8.4f}  "
            f"{v_loss:>10.4f}  {v_rec:>8.4f}  {v_con:>8.4f}  {current_lr:>10.2e}"
        )

        # save best checkpoint
        if v_loss < best_valid:
            best_valid = v_loss
            save_checkpoint(model, optimizer, epoch, v_loss, best_ckpt)
            print(f"         ↑ new best model saved  (valid_loss={best_valid:.4f})")

    # ── test: qualitative evaluation ──────────────────────────────────────
    print(f"\nLoading best model for test evaluation...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    qualitative_test(
        model, test_loader, dataset,
        num_samples=cfg.NUM_TEST_SAMPLES,
        device=device,
    )


if __name__ == "__main__":
    main()