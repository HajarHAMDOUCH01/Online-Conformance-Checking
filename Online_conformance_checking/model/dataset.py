import os
import random
import pickle
import copy
from pathlib import Path

import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import Trace, Event
from pm4py.algo.simulation.playout.petri_net import algorithm as petri_net_playout

import torch
from torch.utils.data import Dataset

import heapq
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import deque


def get_input_places(transition) -> List:
    """Return places connected to transition via incoming arcs."""
    return [arc.source for arc in transition.in_arcs if isinstance(arc.source, PetriNet.Place)]

def get_output_places(transition) -> List:
    """Return places connected to transition via outgoing arcs."""
    return [arc.target for arc in transition.out_arcs if isinstance(arc.target, PetriNet.Place)]



@dataclass
class AlignmentState:
    """State in A* search for prefix alignment."""
    marking: Marking              # current Petri net marking
    event_idx: int                # how many events processed so far
    cost: int                     # total alignment cost so far
    alignment: List[Tuple]        # sequence of moves (event, transition_label)
    parent: Optional['AlignmentState'] = None
    f_score: float = field(default=0.0, compare=False)
    
    def __lt__(self, other):
        return self.f_score < other.f_score


def state_to_key(marking: Marking, event_idx: int) -> Tuple[frozenset, int]:
    place_key = frozenset((place.name, count) for place, count in marking.items())
    return (place_key, event_idx)

def get_enabled_transitions(marking: Marking, petri_net: PetriNet) -> List[PetriNet.Transition]:
    enabled = []
    for transition in petri_net.transitions:
        enabled_flag = True
        for arc in transition.in_arcs:
            source_place = arc.source
            if isinstance(source_place, PetriNet.Place):
                if marking.get(source_place, 0) < arc.weight:
                    enabled_flag = False
                    break
        if enabled_flag:
            enabled.append(transition)
    return enabled

def fire_transition(marking: Marking, transition: PetriNet.Transition) -> Marking:
    """
    Fire a transition and return new marking.
    Removes tokens from input places, adds tokens to output places.
    """
    new_marking = Marking(marking)  # copy
    
    # Remove tokens from input places
    for arc in transition.in_arcs:
        source_place = arc.source
        if isinstance(source_place, PetriNet.Place):
            new_marking[source_place] -= arc.weight
            if new_marking[source_place] == 0:
                del new_marking[source_place]
    
    # Add tokens to output places
    for arc in transition.out_arcs:
        target_place = arc.target
        if isinstance(target_place, PetriNet.Place):
            new_marking[target_place] += arc.weight
    
    return new_marking

def get_activity_to_transitions(petri_net: PetriNet) -> Dict[str, List[PetriNet.Transition]]:
    activity_map = {}
    for transition in petri_net.transitions:
        if transition.label is not None:  
            label = transition.label
            if label not in activity_map:
                activity_map[label] = []
            activity_map[label].append(transition)
    return activity_map

def get_labeled_transitions(petri_net: PetriNet) -> List[PetriNet.Transition]:
    return [t for t in petri_net.transitions if t.label is not None]


def get_silent_transitions(petri_net: PetriNet) -> List[PetriNet.Transition]:
    """Return all silent transitions (label is None)."""
    return [t for t in petri_net.transitions if t.label is None]

def compute_prefix_alignment(
    prefix: List[str],
    petri_net: PetriNet,
    initial_marking: Marking,
    sync_cost: int = 0,
    log_cost: int = 1,
    model_cost: int = 1,
    max_states: int = 50000,
) -> Dict:
    """
    Compute optimal alignment for a prefix using A* search.
    
    Args:
        prefix: List of activity names (events observed so far)
        petri_net: PM4Py Petri net object
        initial_marking: Initial marking (Marking object)
        sync_cost: Cost for sync move (event matches model)
        log_cost: Cost for log move (event with no model transition)
        model_cost: Cost for model move (model transition without event)
        max_states: Maximum states to explore (prevents infinite search)
    
    Returns:
        Dict with keys:
            'aligned_prefix': List[str] - sequence after optimal alignment
            'cost': int - total alignment cost
            'moves': List[Tuple] - alignment moves
            'states_explored': int - number of states explored
            'success': bool - whether optimal alignment was found
    """
    
    # Pre-compute mappings
    activity_to_transitions = get_activity_to_transitions(petri_net)
    silent_transitions = get_silent_transitions(petri_net)
    all_labeled_transitions = get_labeled_transitions(petri_net)
    
    # Start state
    start_state = AlignmentState(
        marking=Marking(initial_marking),
        event_idx=0,
        cost=0,
        alignment=[],
        parent=None,
        f_score=0
    )
    
    # Priority queue: (f_score, state)
    open_set = [(0, start_state)]
    
    # Visited states: key → best cost
    visited: Dict[Tuple[frozenset, int], int] = {}
    
    best_final_state = None
    best_cost = float('inf')
    states_explored = 0
    
    while open_set and states_explored < max_states:
        f_score, current = heapq.heappop(open_set)
        states_explored += 1
        
        # Skip if we found a better path to this state
        state_key = state_to_key(current.marking, current.event_idx)
        if state_key in visited and visited[state_key] < current.cost:
            continue
        visited[state_key] = current.cost
        
        # Check if we've processed all events
        if current.event_idx == len(prefix):
            if current.cost < best_cost:
                best_cost = current.cost
                best_final_state = current
            continue
        
        # Get next event to process
        next_event = prefix[current.event_idx]
        
        # --------------------------------------------------------------
        # Move 1: Sync move (event matches a model transition)
        # --------------------------------------------------------------
        if next_event in activity_to_transitions:
            for transition in activity_to_transitions[next_event]:
                if transition in get_enabled_transitions(current.marking, petri_net):
                    new_marking = fire_transition(current.marking, transition)
                    new_state = AlignmentState(
                        marking=new_marking,
                        event_idx=current.event_idx + 1,
                        cost=current.cost + sync_cost,
                        alignment=current.alignment + [(next_event, transition.label)],
                        parent=current,
                        f_score=current.cost + sync_cost
                    )
                    heapq.heappush(open_set, (new_state.f_score, new_state))
        
        # --------------------------------------------------------------
        # Move 2: Log move (skip the event, no model transition)
        # --------------------------------------------------------------
        new_state_log = AlignmentState(
            marking=current.marking,
            event_idx=current.event_idx + 1,
            cost=current.cost + log_cost,
            alignment=current.alignment + [(next_event, None)],
            parent=current,
            f_score=current.cost + log_cost
        )
        heapq.heappush(open_set, (new_state_log.f_score, new_state_log))
        
        # --------------------------------------------------------------
        # Move 3: Model moves (fire transitions without consuming event)
        # These include both silent and labeled transitions
        # --------------------------------------------------------------
        
        # 3a: Silent transitions
        for transition in silent_transitions:
            if transition in get_enabled_transitions(current.marking, petri_net):
                new_marking = fire_transition(current.marking, transition)
                new_state = AlignmentState(
                    marking=new_marking,
                    event_idx=current.event_idx,
                    cost=current.cost + model_cost,
                    alignment=current.alignment + [(None, None)],  # silent move
                    parent=current,
                    f_score=current.cost + model_cost
                )
                heapq.heappush(open_set, (new_state.f_score, new_state))
        
        # 3b: Labeled transitions (insert an activity)
        for transition in all_labeled_transitions:
            if transition in get_enabled_transitions(current.marking, petri_net):
                new_marking = fire_transition(current.marking, transition)
                new_state = AlignmentState(
                    marking=new_marking,
                    event_idx=current.event_idx,
                    cost=current.cost + model_cost,
                    alignment=current.alignment + [(None, transition.label)],
                    parent=current,
                    f_score=current.cost + model_cost
                )
                heapq.heappush(open_set, (new_state.f_score, new_state))
    
    # Build result
    if best_final_state is None:
        # Fallback: log all events
        return {
            'aligned_prefix': prefix,
            'cost': len(prefix) * log_cost,
            'moves': [(event, None) for event in prefix],
            'states_explored': states_explored,
            'success': False
        }
    
    # Reconstruct aligned prefix from moves
    aligned_prefix = []
    for event, transition_label in best_final_state.alignment:
        if transition_label is not None:
            # Sync move or labeled model move
            aligned_prefix.append(transition_label)
        elif event is not None:
            # Log move
            aligned_prefix.append(event)
        # Silent moves (None, None) are not added to aligned prefix
    
    return {
        'aligned_prefix': aligned_prefix,
        'cost': best_final_state.cost,
        'moves': best_final_state.alignment,
        'states_explored': states_explored,
        'success': True
    }
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
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"


def build_vocabulary(traces: list[list[str]]) -> dict[str, int]:
    """Build activity → integer index mapping."""
    vocab = {
        PADDING_TOKEN: 0,
        UNKNOWN_TOKEN: 1,
        BOS_TOKEN: 2,    
        EOS_TOKEN: 3,  
    }
    idx = len(vocab)
    for trace in traces:
        for act in trace:
            if act not in vocab:
                vocab[act] = idx
                idx += 1
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
    petri_net,
    initial_marking,
    source_prefixes: list[list[str]],
    vocabulary_list: list[str],
    noise_variants_per_prefix: int = 3,
    max_noise_operations: int = 4,
    balance_by_length: bool = True,
    max_per_length: int = 50,
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
        for p in source_prefixes:
            by_len[len(p)].append(p)
        balanced = []
        for length, plist in by_len.items():
            sample = plist if len(plist) <= max_per_length else random.sample(plist, max_per_length)
            balanced.extend(sample)
        source_prefixes = balanced

    pairs = []
    
    for source_prefix in source_prefixes:
        for _ in range(noise_variants_per_prefix):
            # Randomly choose noise type
            noise_type = random.choice(["deletion", "replacement", "swap"])
            n_ops = random.randint(1, max_noise_operations)
            
            # Apply noise
            noisy = inject_noise(
                source_prefix, vocabulary_list, noise_type, n_ops
            )
            
            # Determine if A* is needed
            if noise_type == "insertion":
                # Insertion: run A* to find optimal alignment
                result = compute_prefix_alignment(noisy, petri_net, initial_marking)
                aligned_prefix = result['aligned_prefix']
                optimal_cost = result['cost']
            else:
                aligned_prefix = source_prefix
                optimal_cost = n_ops
            
            aligned_with_special = [BOS_TOKEN] + aligned_prefix + [EOS_TOKEN]
            pairs.append((noisy, aligned_with_special, optimal_cost))
    
    return pairs


def encode_pairs(
    pairs: list[tuple[list[str], list[str], int]],
    vocab: dict,
) -> list[tuple[list[int], list[int], int]]:
    """Returns list of (noisy_encoded, aligned_encoded, cost)"""
    return [
        (encode_sequence(noisy, vocab), encode_sequence(aligned, vocab), cost)
        for noisy, aligned, cost in pairs
    ]


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PrefixConformanceDataset(Dataset):
    def __init__(self, encoded_pairs: list[tuple[list[int], list[int], int]], vocab: dict):
        """
        encoded_pairs: list of (noisy, aligned, cost)
        """
        self.pairs = encoded_pairs  # now includes cost
        self.vocab = vocab
        self.pad_idx = vocab[PADDING_TOKEN]
        self.bos_idx = vocab[BOS_TOKEN]
        self.eos_idx = vocab[EOS_TOKEN]

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

    def __getitem__(self, idx):
        noisy, aligned, cost = self.pairs[idx]  
        return {
            'noisy': torch.tensor(noisy, dtype=torch.long),
            'aligned': torch.tensor(aligned, dtype=torch.long),
            'cost': cost, 
            'noisy_len': len(noisy),
            'aligned_len': len(aligned),
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def bos_token_id(self) -> int:
        return self.bos_idx
    
    @property
    def eos_token_id(self) -> int:
        return self.eos_idx

    def decode(self, indices: list[int]) -> list[str]:
        inv = {v: k for k, v in self.vocab.items()}
        return [inv.get(i, UNKNOWN_TOKEN) for i in indices]


def collate_fn(batch: list[dict], pad_idx: int = 0) -> dict:
    noisy_seqs = [item['noisy'] for item in batch]
    aligned_seqs = [item['aligned'] for item in batch]
    costs = torch.tensor([item['cost'] for item in batch], dtype=torch.float32)  # ← new
    
    noisy_lengths = torch.tensor([len(seq) for seq in noisy_seqs], dtype=torch.long)
    aligned_lengths = torch.tensor([len(seq) for seq in aligned_seqs], dtype=torch.long)
    
    noisy_padded = torch.nn.utils.rnn.pad_sequence(
        noisy_seqs, batch_first=True, padding_value=pad_idx
    )
    aligned_padded = torch.nn.utils.rnn.pad_sequence(
        aligned_seqs, batch_first=True, padding_value=pad_idx
    )
    
    return {
        'noisy_padded': noisy_padded,
        'aligned_padded': aligned_padded,
        'costs': costs,  
        'noisy_lengths': noisy_lengths,
        'aligned_lengths': aligned_lengths,
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
    MAX_NOISE_OPS           = 4      # max noise operations per copy
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
    vocab_list = [a for a in vocab if a not in (PADDING_TOKEN, UNKNOWN_TOKEN, BOS_TOKEN, EOS_TOKEN)]
    print(f"  → vocabulary size: {len(vocab)} (including PAD and UNK and BOS ad EOS)")

    # ── 5. build (noisy, conforming) pairs ─────────────────────────────────
    print("Building noisy pairs...")
    raw_pairs = build_pairs(
        petrinet,
        mi,
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
    print(f"  conforming prefix: {dataset.decode(sample['aligned'].tolist())}")
    sample = dataset[7000]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['aligned'].tolist())}")
    sample = dataset[6000]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['aligned'].tolist())}")
    sample = dataset[5000]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['aligned'].tolist())}")
    sample = dataset[20000]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['aligned'].tolist())}")
    sample = dataset[10000]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['aligned'].tolist())}")
    sample = dataset[56]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['aligned'].tolist())}")
    sample = dataset[3600]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['aligned'].tolist())}")
    sample = dataset[8060]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['aligned'].tolist())}")
    sample = dataset[9200]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['aligned'].tolist())}")
    sample = dataset[82]
    print(f"  noisy prefix     : {dataset.decode(sample['noisy'].tolist())}")
    print(f"  conforming prefix: {dataset.decode(sample['aligned'].tolist())}")
    print(f"  vocab size       : {dataset.vocab_size}")
    print(f"  total pairs      : {len(dataset):,}")
    print("\nDone.")