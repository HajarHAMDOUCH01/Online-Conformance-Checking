import os
import random
import pickle
import copy
from pathlib import Path

import pm4py
from pm4py.objects.log.obj import Trace, Event
from pm4py.algo.simulation.playout.petri_net import algorithm as petri_net_playout

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Noise injection helpers
# ---------------------------------------------------------------------------

def _random_deletion(activities: list[str]) -> list[str]:
    """Remove one random activity."""
    if len(activities) <= 1:
        return activities[:]
    idx = random.randrange(len(activities))
    result = activities[:idx] + activities[idx + 1:]
    return result


def _random_insertion(activities: list[str], vocabulary: list[str]) -> list[str]:
    """Insert a random activity at a random position."""
    idx = random.randrange(len(activities) + 1)
    act = random.choice(vocabulary)
    return activities[:idx] + [act] + activities[idx:]


def _random_replacement(activities: list[str], vocabulary: list[str]) -> list[str]:
    """Replace one random activity with a randomly chosen one."""
    if not activities:
        return activities[:]
    result = activities[:]
    idx = random.randrange(len(result))
    result[idx] = random.choice(vocabulary)
    return result


def _random_swap(activities: list[str]) -> list[str]:
    """Swap two randomly chosen activities."""
    if len(activities) < 2:
        return activities[:]
    result = activities[:]
    i, j = random.sample(range(len(result)), 2)
    result[i], result[j] = result[j], result[i]
    return result


NOISE_FUNCTIONS = {
    "deletion":    _random_deletion,
    "insertion":   _random_insertion,
    "replacement": _random_replacement,
    "swap":        _random_swap,
}


def inject_noise(
    activities: list[str],
    vocabulary: list[str],
    noise_type: str | None = None,
    num_operations: int = 1,
) -> list[str]:
    """
    Apply `num_operations` noise operations of `noise_type` to a copy of
    `activities`.  If `noise_type` is None a type is chosen uniformly at random.
    Returns the noisy activity list (original is not modified).
    """
    result = activities[:]
    for _ in range(num_operations):
        ntype = noise_type or random.choice(list(NOISE_FUNCTIONS.keys()))
        fn = NOISE_FUNCTIONS[ntype]
        if ntype in ("insertion", "replacement"):
            result = fn(result, vocabulary)
        else:
            result = fn(result)
        # guard against empty prefix after deletion
        if not result:
            result = activities[:]
    return result


# ---------------------------------------------------------------------------
# Playout & prefix extraction
# ---------------------------------------------------------------------------

def import_petrinet_model(pnml_file_path: str):
    petrinet, mi, mf = pm4py.read_pnml(pnml_file_path)
    return petrinet, mi, mf


def import_event_log(event_log_file_path: str):
    return pm4py.read_xes(event_log_file_path)


def playout_model(
    petrinet,
    mi,
    mf,
    num_traces: int = 2000,
    max_trace_length: int = 100,
) -> list[list[str]]:
    """
    Play out the Petri net and return a list of activity-label sequences.
    """
    parameters = {
        petri_net_playout.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: num_traces,
        petri_net_playout.Variants.BASIC_PLAYOUT.value.Parameters.MAX_TRACE_LENGTH: max_trace_length,
    }
    sim_log = petri_net_playout.apply(
        petrinet, mi, mf,
        variant=petri_net_playout.Variants.BASIC_PLAYOUT,
        parameters=parameters,
    )
    traces = []
    for trace in sim_log:
        acts = [event["concept:name"] for event in trace]
        if acts:
            traces.append(acts)
    return traces


def extract_prefixes(traces: list[list[str]]) -> list[list[str]]:
    """Return all non-empty prefixes of every trace."""
    prefixes = []
    for trace in traces:
        for k in range(1, len(trace) + 1):
            prefixes.append(trace[:k])
    return prefixes


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

PADDING_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"


def build_vocabulary(traces: list[list[str]]) -> dict[str, int]:
    """Build activity → integer index mapping."""
    vocab = {PADDING_TOKEN: 0, UNKNOWN_TOKEN: 1}
    for trace in traces:
        for act in trace:
            if act not in vocab:
                vocab[act] = len(vocab)
    return vocab


def encode_sequence(
    activities: list[str],
    vocab: dict[str, int],
) -> list[int]:
    unk = vocab[UNKNOWN_TOKEN]
    return [vocab.get(a, unk) for a in activities]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_pairs(
    conforming_prefixes: list[list[str]],
    vocabulary_list: list[str],
    noise_variants_per_prefix: int = 3,
    max_noise_operations: int = 2,
    balance_by_length: bool = True,
    max_per_length: int = 500,
    seed: int = 42,
) -> list[tuple[list[str], list[str]]]:
    """
    For every conforming prefix generate `noise_variants_per_prefix` noisy
    copies.  Returns a list of (noisy_prefix, conforming_prefix) pairs as
    raw activity-label lists (encoding happens later).

    `balance_by_length` caps the number of conforming prefixes used per
    prefix length to avoid overwhelming short-prefix bias.
    """
    random.seed(seed)

    # optionally balance by length
    if balance_by_length:
        from collections import defaultdict
        by_len: dict[int, list] = defaultdict(list)
        for p in conforming_prefixes:
            by_len[len(p)].append(p)
        balanced = []
        for length, plist in by_len.items():
            sample = plist if len(plist) <= max_per_length else random.sample(plist, max_per_length)
            balanced.extend(sample)
        conforming_prefixes = balanced

    pairs = []
    for conf_prefix in conforming_prefixes:
        for _ in range(noise_variants_per_prefix):
            n_ops = random.randint(1, max(1, max_noise_operations))
            noisy = inject_noise(
                conf_prefix,
                vocabulary_list,
                noise_type=None,       # random noise type each time
                num_operations=n_ops,
            )
            pairs.append((noisy, conf_prefix))

    return pairs


def encode_pairs(
    pairs: list[tuple[list[str], list[str]]],
    vocab: dict[str, int],
) -> list[tuple[list[int], list[int]]]:
    return [
        (encode_sequence(noisy, vocab), encode_sequence(conf, vocab))
        for noisy, conf in pairs
    ]


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PrefixConformanceDataset(Dataset):
    """
    PyTorch Dataset of (noisy_prefix, conforming_prefix) integer-encoded pairs.

    Each item is a dict with keys:
        - 'noisy'      : LongTensor  [noisy_len]
        - 'conforming' : LongTensor  [conf_len]
        - 'noisy_len'  : int
        - 'conf_len'   : int

    Collate with `collate_fn` (provided below) for padded batch tensors.
    """

    def __init__(
        self,
        encoded_pairs: list[tuple[list[int], list[int]]],
        vocab: dict[str, int],
    ):
        self.pairs = encoded_pairs
        self.vocab = vocab
        self.pad_idx = vocab[PADDING_TOKEN]

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"pairs": self.pairs, "vocab": self.vocab}, f)
        print(f"Dataset saved → {path}  ({len(self.pairs):,} pairs)")

    @classmethod
    def load(cls, path: str) -> "PrefixConformanceDataset":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["pairs"], data["vocab"])

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        noisy, conforming = self.pairs[idx]
        return {
            "noisy":      torch.tensor(noisy,      dtype=torch.long),
            "conforming": torch.tensor(conforming, dtype=torch.long),
            "noisy_len":  len(noisy),
            "conf_len":   len(conforming),
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def decode(self, indices: list[int]) -> list[str]:
        inv = {v: k for k, v in self.vocab.items()}
        return [inv.get(i, UNKNOWN_TOKEN) for i in indices]


def collate_fn(batch: list[dict], pad_idx: int = 0) -> dict:
    """
    Pads noisy and conforming sequences to the max length in the batch.
    Returns:
        noisy_padded      : LongTensor [B, max_noisy_len]
        noisy_lengths     : LongTensor [B]
        conforming_padded : LongTensor [B, max_conf_len]
        conf_lengths      : LongTensor [B]
    """
    noisy_seqs  = [item["noisy"]      for item in batch]
    conf_seqs   = [item["conforming"] for item in batch]
    noisy_lens  = torch.tensor([item["noisy_len"]  for item in batch], dtype=torch.long)
    conf_lens   = torch.tensor([item["conf_len"]   for item in batch], dtype=torch.long)

    noisy_padded = torch.nn.utils.rnn.pad_sequence(
        noisy_seqs, batch_first=True, padding_value=pad_idx
    )
    conf_padded = torch.nn.utils.rnn.pad_sequence(
        conf_seqs, batch_first=True, padding_value=pad_idx
    )

    return {
        "noisy_padded":      noisy_padded,
        "noisy_lengths":     noisy_lens,
        "conforming_padded": conf_padded,
        "conf_lengths":      conf_lens,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── paths ──────────────────────────────────────────────────────────────
    LOG_FILE   = r"C:\Users\LENONVO\OneDrive\Desktop\graphs\sujet-CRAN\datasets\spesis\spesis_full_log.xes"
    MODEL_FILE = r"C:\Users\LENONVO\OneDrive\Desktop\graphs\sujet-CRAN\datasets\spesis\spesis_reference_model.pnml"

    OUTPUT_DIR  = Path(MODEL_FILE).parent
    OUTPUT_FILE = OUTPUT_DIR / "confo_non_conform_prefixes_playout_dataset.pkl"

    # ── hyper-parameters ───────────────────────────────────────────────────
    NUM_PLAYOUT_TRACES      = 2000   # traces played out from the model
    MAX_TRACE_LENGTH        = 100    # maximum trace length during playout
    NOISE_VARIANTS          = 3      # noisy copies per conforming prefix
    MAX_NOISE_OPS           = 2      # max noise operations per copy
    BALANCE_MAX_PER_LENGTH  = 500    # cap per prefix length (balancing)
    SEED                    = 42

    # ── 1. load model ──────────────────────────────────────────────────────
    print("Loading Petri net model...")
    petrinet, mi, mf = import_petrinet_model(MODEL_FILE)

    # ── 2. playout ─────────────────────────────────────────────────────────
    print(f"Playing out {NUM_PLAYOUT_TRACES} traces...")
    conforming_traces = playout_model(
        petrinet, mi, mf,
        num_traces=NUM_PLAYOUT_TRACES,
        max_trace_length=MAX_TRACE_LENGTH,
    )
    print(f"  → {len(conforming_traces)} conforming traces generated")

    # ── 3. extract prefixes ────────────────────────────────────────────────
    print("Extracting prefixes...")
    conforming_prefixes = extract_prefixes(conforming_traces)
    print(f"  → {len(conforming_prefixes):,} conforming prefixes")

    # ── 4. build vocabulary ────────────────────────────────────────────────
    print("Building vocabulary...")
    vocab = build_vocabulary(conforming_traces)
    vocab_list = [a for a in vocab if a not in (PADDING_TOKEN, UNKNOWN_TOKEN)]
    print(f"  → vocabulary size: {len(vocab)} (including PAD and UNK)")

    # ── 5. build (noisy, conforming) pairs ─────────────────────────────────
    print("Building noisy pairs...")
    raw_pairs = build_pairs(
        conforming_prefixes,
        vocabulary_list=vocab_list,
        noise_variants_per_prefix=NOISE_VARIANTS,
        max_noise_operations=MAX_NOISE_OPS,
        balance_by_length=True,
        max_per_length=BALANCE_MAX_PER_LENGTH,
        seed=SEED,
    )
    print(f"  → {len(raw_pairs):,} (noisy, conforming) pairs")

    # ── 6. encode ──────────────────────────────────────────────────────────
    print("Encoding pairs...")
    encoded_pairs = encode_pairs(raw_pairs, vocab)

    # ── 7. create dataset and save ─────────────────────────────────────────
    dataset = PrefixConformanceDataset(encoded_pairs, vocab)
    dataset.save(str(OUTPUT_FILE))

    # ── 8. quick sanity check ──────────────────────────────────────────────
    print("\n── Sanity check ──")
    sample = dataset[9000]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['conforming'].tolist())}")
    sample = dataset[7000]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['conforming'].tolist())}")
    sample = dataset[6000]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['conforming'].tolist())}")
    sample = dataset[5000]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['conforming'].tolist())}")
    sample = dataset[20000]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['conforming'].tolist())}")
    sample = dataset[10000]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['conforming'].tolist())}")
    sample = dataset[56]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['conforming'].tolist())}")
    sample = dataset[3600]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['conforming'].tolist())}")
    sample = dataset[8060]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['conforming'].tolist())}")
    sample = dataset[9200]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['conforming'].tolist())}")
    sample = dataset[82]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['conforming'].tolist())}")
    print(f"  vocab size       : {dataset.vocab_size}")
    print(f"  total pairs      : {len(dataset):,}")
    print("\nDone.")