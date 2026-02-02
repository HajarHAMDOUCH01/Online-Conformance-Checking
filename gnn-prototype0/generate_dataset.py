"""
Dataset Generation for GNN-based Online Conformance Checking

This module generates training data by simulating event streams over the N1 Petri net,
creating prefix windows with markings, and computing labels for:
1. Next transition prediction (multi-label classification)
2. Conformance classification (binary classification)
"""
import sys 
sys.path.append("/kaggle/working/GNN-classifer-for-an-event-stream")
import torch
import numpy as np
from typing import List, Tuple, Dict, Set
import random
from collections import defaultdict
import pickle


def build_n1_petri_net_structure():
    """
    Build the N1 Petri net structure as a heterogeneous graph
    This represents the STATIC structure of the process model
    
    Places: pi, p1, p2, p3, p4, p5, po (indices 0-6)
    Transitions: t1(a), t2(b), t3(c), t4(d), t5(d), t6(τ), t7(e), t8(f) (indices 0-7)
    
    NOTE: We have 8 transitions total:
    - 6 visible activities: a, b, c, d, e, f
    - Activity 'd' appears in 2 transitions: t4 and t5
    - t6 is silent (τ)
    
    Returns edge_index for pre and post arcs
    """
    # Pre-arcs: (place, pre, transition)
    pre_arcs = [
        [0, 0],  # pi -> t1(a)
        [1, 1],  # p1 -> t2(b)
        [2, 2],  # p2 -> t3(c)
        [1, 3], [4, 3],  # p1, p4 -> t4(d)
        [3, 4], [4, 4],  # p3, p4 -> t5(d)
        [5, 5],  # p5 -> t6(τ)
        [5, 6],  # p5 -> t7(e)
        [5, 7]   # p5 -> t8(f)
    ]
    
    # Post-arcs: (transition, post, place)
    post_arcs = [
        [0, 1], [0, 2],  # t1 -> p1, p2
        [1, 3],  # t2 -> p3
        [2, 4],  # t3 -> p4
        [3, 5],  # t4 -> p5
        [4, 5],  # t5 -> p5
        [5, 1], [5, 2],  # t6 -> p1, p2 (loop)
        [6, 6],  # t7 -> po
        [7, 6]   # t8 -> po
    ]
    
    pre_edge_index = torch.tensor(pre_arcs, dtype=torch.long).t().contiguous()
    post_edge_index = torch.tensor(post_arcs, dtype=torch.long).t().contiguous()
    
    return pre_edge_index, post_edge_index


class PetriNetSimulator:
    """Simulates execution of the N1 Petri net"""
    
    def __init__(self):
        self.num_places = 7
        self.num_transitions = 8
        
        # Transition labels (visible activities + silent)
        self.transition_labels = {
            0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'd', 5: 'τ', 6: 'e', 7: 'f'
        }
        
        # Activity to transition mapping (for visible activities)
        self.activity_to_transitions = {
            'a': [0],
            'b': [1],
            'c': [2],
            'd': [3, 4],  # Two transitions for 'd'
            'e': [6],
            'f': [7]
        }
        
        # Define pre and post conditions
        self.pre_conditions = {
            0: [0],      # t1: pi
            1: [1],      # t2: p1
            2: [2],      # t3: p2
            3: [1, 4],   # t4: p1, p4
            4: [3, 4],   # t5: p3, p4
            5: [5],      # t6: p5
            6: [5],      # t7: p5
            7: [5]       # t8: p5
        }
        
        self.post_conditions = {
            0: [1, 2],   # t1 -> p1, p2
            1: [3],      # t2 -> p3
            2: [4],      # t3 -> p4
            3: [5],      # t4 -> p5
            4: [5],      # t5 -> p5
            5: [1, 2],   # t6 -> p1, p2 (loop)
            6: [6],      # t7 -> po
            7: [6]       # t8 -> po
        }
        
        # Initial and final markings
        self.initial_marking = [1, 0, 0, 0, 0, 0, 0]  # [pi]
        self.final_marking = [0, 0, 0, 0, 0, 0, 1]    # [po]
    
    def get_initial_marking(self):
        """Return initial marking as a list"""
        return self.initial_marking.copy()
    
    def is_enabled(self, transition: int, marking: List[int]) -> bool:
        """Check if a transition is enabled in the current marking"""
        for place in self.pre_conditions[transition]:
            if marking[place] < 1:
                return False
        return True
    
    def get_enabled_transitions(self, marking: List[int]) -> List[int]:
        """Get all enabled transitions in the current marking"""
        enabled = []
        for t in range(self.num_transitions):
            if self.is_enabled(t, marking):
                enabled.append(t)
        return enabled
    
    def fire_transition(self, transition: int, marking: List[int]) -> List[int]:
        """Fire a transition and return the new marking"""
        if not self.is_enabled(transition, marking):
            raise ValueError(f"Transition {transition} is not enabled")
        
        new_marking = marking.copy()
        
        # Consume tokens
        for place in self.pre_conditions[transition]:
            new_marking[place] -= 1
        
        # Produce tokens
        for place in self.post_conditions[transition]:
            new_marking[place] += 1
        
        return new_marking
    
    def is_final_marking(self, marking: List[int]) -> bool:
        """Check if marking is the final marking"""
        return marking == self.final_marking
    
    def generate_conformant_trace(self, max_length: int = 20, 
                                  allow_loops: bool = True) -> Tuple[List[str], List[int], List[List[int]]]:
        """
        Generate a conformant trace (sequence of activities)
        
        Returns:
            - activities: List of activity labels
            - transitions: List of transition indices fired
            - markings: List of markings after each transition
        """
        marking = self.get_initial_marking()
        activities = []
        transitions_fired = []
        markings = [marking.copy()]
        
        max_iterations = max_length * 2  # Prevent infinite loops
        iterations = 0
        
        while not self.is_final_marking(marking) and iterations < max_iterations:
            enabled = self.get_enabled_transitions(marking)
            
            if not enabled:
                break
            
            # Filter out silent transitions and loops if needed
            visible_enabled = [t for t in enabled if self.transition_labels[t] != 'τ']
            
            if not allow_loops and 5 in enabled:  # t6 is the loop transition
                visible_enabled = [t for t in visible_enabled if t != 5]
            
            # Prefer visible transitions
            if visible_enabled:
                transition = random.choice(visible_enabled)
            else:
                transition = random.choice(enabled)
            
            # Fire transition
            marking = self.fire_transition(transition, marking)
            transitions_fired.append(transition)
            markings.append(marking.copy())
            
            # Add activity (skip silent transitions)
            if self.transition_labels[transition] != 'τ':
                activities.append(self.transition_labels[transition])
            
            iterations += 1
        
        return activities, transitions_fired, markings
    
    def introduce_deviation(self, activities: List[str], 
                           deviation_type: str = 'skip') -> List[str]:
        """
        Introduce deviations into a conformant trace
        
        Types:
        - 'skip': Remove an activity
        - 'insert': Add a random activity
        - 'swap': Swap two adjacent activities
        - 'wrong': Replace an activity with a wrong one
        """
        if len(activities) < 2:
            return activities
        
        deviated = activities.copy()
        
        if deviation_type == 'skip':
            # Remove a random activity (not first or last)
            if len(deviated) > 2:
                idx = random.randint(1, len(deviated) - 2)
                deviated.pop(idx)
        
        elif deviation_type == 'insert':
            # Insert a random valid activity
            idx = random.randint(0, len(deviated))
            random_activity = random.choice(['a', 'b', 'c', 'd', 'e', 'f'])
            deviated.insert(idx, random_activity)
        
        elif deviation_type == 'swap':
            # Swap two adjacent activities
            if len(deviated) > 1:
                idx = random.randint(0, len(deviated) - 2)
                deviated[idx], deviated[idx + 1] = deviated[idx + 1], deviated[idx]
        
        elif deviation_type == 'wrong':
            # Replace a random activity with a wrong one
            idx = random.randint(0, len(deviated) - 1)
            all_activities = ['a', 'b', 'c', 'd', 'e', 'f']
            all_activities.remove(deviated[idx])
            deviated[idx] = random.choice(all_activities)
        
        return deviated


class ConformanceDatasetGenerator:
    """Generate dataset for GNN-based conformance checking"""
    
    def __init__(self, window_size: int = 3):
        self.simulator = PetriNetSimulator()
        self.window_size = window_size
        self.pre_edge_index, self.post_edge_index = build_n1_petri_net_structure()
    
    def replay_prefix(self, activities: List[str]) -> Tuple[List[int], bool]:
        """
        Replay a prefix of activities and return final marking and conformance status
        
        Returns:
            - marking: Final marking after replay
            - is_conformant: Whether the prefix is conformant
        """
        marking = self.simulator.get_initial_marking()
        is_conformant = True
        
        for activity in activities:
            # Get possible transitions for this activity
            if activity not in self.simulator.activity_to_transitions:
                # Unknown activity - non-conformant
                is_conformant = False
                continue
            
            possible_transitions = self.simulator.activity_to_transitions[activity]
            
            # Try to fire one of the possible transitions
            fired = False
            for transition in possible_transitions:
                if self.simulator.is_enabled(transition, marking):
                    marking = self.simulator.fire_transition(transition, marking)
                    fired = True
                    break
            
            if not fired:
                # Could not fire any transition for this activity - non-conformant
                is_conformant = False
        
        return marking, is_conformant
    
    def create_sample(self, activities: List[str], window_end: int) -> Dict:
        """
        Create a training sample from an activity sequence
        
        Args:
            activities: Full sequence of activities
            window_end: Index where the window ends (exclusive)
        
        Returns:
            Dictionary with features and labels
        """
        # Get window
        window_start = max(0, window_end - self.window_size)
        window = activities[window_start:window_end]
        
        # Replay to get marking
        marking, is_conformant = self.replay_prefix(window)
        
        # Get next activity (if exists)
        next_activity = activities[window_end] if window_end < len(activities) else None
        
        # Determine next enabled transitions
        enabled_transitions = self.simulator.get_enabled_transitions(marking)
        
        # Create features
        # Place features: marking (token count)
        place_features = torch.tensor(marking, dtype=torch.float32).unsqueeze(1)
        
        # Transition features: one-hot encoding of activity labels
        # Features: [is_enabled, is_a, is_b, is_c, is_d, is_e, is_f, is_silent]
        transition_features = torch.zeros(self.simulator.num_transitions, 8)
        
        for t in range(self.simulator.num_transitions):
            # Is enabled
            transition_features[t, 0] = 1.0 if t in enabled_transitions else 0.0
            
            # Activity label one-hot
            label = self.simulator.transition_labels[t]
            if label == 'a':
                transition_features[t, 1] = 1.0
            elif label == 'b':
                transition_features[t, 2] = 1.0
            elif label == 'c':
                transition_features[t, 3] = 1.0
            elif label == 'd':
                transition_features[t, 4] = 1.0
            elif label == 'e':
                transition_features[t, 5] = 1.0
            elif label == 'f':
                transition_features[t, 6] = 1.0
            elif label == 'τ':
                transition_features[t, 7] = 1.0
        
        # Prefix encoding (recent activities)
        activity_vocab = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5}
        prefix_encoding = torch.zeros(self.window_size, 6)
        for i, act in enumerate(window):
            if act in activity_vocab:
                prefix_encoding[i, activity_vocab[act]] = 1.0
        
        # Labels
        # Label 1: Next transitions (multi-label)
        next_transitions_label = torch.zeros(self.simulator.num_transitions)
        if next_activity and next_activity in self.simulator.activity_to_transitions:
            for t in self.simulator.activity_to_transitions[next_activity]:
                if t in enabled_transitions:
                    next_transitions_label[t] = 1.0
        
        # Label 2: Conformance (binary)
        conformance_label = torch.tensor([1.0 if is_conformant else 0.0])
        
        return {
            'place_features': place_features,
            'transition_features': transition_features,
            'prefix_encoding': prefix_encoding.flatten(),  # Flatten for easier processing
            'pre_edge_index': self.pre_edge_index,
            'post_edge_index': self.post_edge_index,
            'next_transitions': next_transitions_label,
            'is_conformant': conformance_label,
            'marking': torch.tensor(marking, dtype=torch.float32),
            'enabled_transitions': torch.tensor(enabled_transitions, dtype=torch.long)
        }
    
    def generate_dataset(self, num_traces: int = 1000, 
                        deviation_ratio: float = 0.3) -> List[Dict]:
        """
        Generate a complete dataset
        
        Args:
            num_traces: Number of traces to generate
            deviation_ratio: Ratio of traces with deviations
        
        Returns:
            List of samples
        """
        samples = []
        
        num_deviated = int(num_traces * deviation_ratio)
        num_conformant = num_traces - num_deviated
        
        print(f"Generating {num_conformant} conformant traces...")
        
        # Generate conformant traces
        for i in range(num_conformant):
            activities, _, _ = self.simulator.generate_conformant_trace()
            
            # Create samples for each prefix
            for end_idx in range(1, len(activities) + 1):
                sample = self.create_sample(activities, end_idx)
                samples.append(sample)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_conformant} conformant traces")
        
        print(f"\nGenerating {num_deviated} deviated traces...")
        
        # Generate deviated traces
        deviation_types = ['skip', 'insert', 'swap', 'wrong']
        for i in range(num_deviated):
            # First generate conformant trace
            activities, _, _ = self.simulator.generate_conformant_trace()
            
            # Then introduce deviation
            deviation_type = random.choice(deviation_types)
            deviated_activities = self.simulator.introduce_deviation(activities, deviation_type)
            
            # Create samples for each prefix
            for end_idx in range(1, len(deviated_activities) + 1):
                sample = self.create_sample(deviated_activities, end_idx)
                samples.append(sample)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_deviated} deviated traces")
        
        print(f"\nTotal samples generated: {len(samples)}")
        return samples
    
    def save_dataset(self, samples: List[Dict], filepath: str):
        """Save dataset to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(samples, f)
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> List[Dict]:
        """Load dataset from file"""
        with open(filepath, 'rb') as f:
            samples = pickle.load(f)
        print(f"Dataset loaded from {filepath}: {len(samples)} samples")
        return samples


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("Conformance Checking Dataset Generator")
    print("=" * 60)
    
    # Create generator
    generator = ConformanceDatasetGenerator(window_size=3)
    
    # Generate dataset
    dataset = generator.generate_dataset(
        num_traces=1000,
        deviation_ratio=0.3
    )
    
    # Save dataset
    generator.save_dataset(dataset, '/kaggle/working/conformance_dataset.pkl')
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    conformant_count = sum(1 for s in dataset if s['is_conformant'].item() == 1.0)
    non_conformant_count = len(dataset) - conformant_count
    
    print(f"Total samples: {len(dataset)}")
    print(f"Conformant prefixes: {conformant_count} ({conformant_count/len(dataset)*100:.1f}%)")
    print(f"Non-conformant prefixes: {non_conformant_count} ({non_conformant_count/len(dataset)*100:.1f}%)")
    
    # Sample statistics
    sample = dataset[0]
    print(f"\nSample structure:")
    print(f"  Place features shape: {sample['place_features'].shape}")
    print(f"  Transition features shape: {sample['transition_features'].shape}")
    print(f"  Prefix encoding shape: {sample['prefix_encoding'].shape}")
    print(f"  Pre-edges: {sample['pre_edge_index'].shape}")
    print(f"  Post-edges: {sample['post_edge_index'].shape}")