"""
GNN Model with Sequential Conformance Checking 

Key improvement: Stage 2 checks each transition transition-by-transition without seeing
the full enablement vector upfront. This forces the model to actually learn
Petri net semantics instead of just comparing vectors.
"""
import sys 
sys.path.append("/kaggle/working/GNN-classifer-for-an-event-stream")
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class HeteroGraphConv(nn.Module):
    """Heterogeneous Graph Convolution Layer"""
    
    def __init__(self, place_dim: int, transition_dim: int, hidden_dim: int):
        super(HeteroGraphConv, self).__init__()
        
        # making the features of places nodes and transitions nodes into embeddings
        self.place_to_trans_msg = nn.Linear(place_dim, hidden_dim)
        self.trans_to_place_msg = nn.Linear(transition_dim, hidden_dim)

        # a place recieves messages and concatenates them with its own features
        self.place_update = nn.Linear(place_dim + hidden_dim, place_dim)
        self.trans_update = nn.Linear(transition_dim + hidden_dim, transition_dim)

        # this layers take a message and output a logit (score)
        self.place_attention = nn.Linear(hidden_dim, 1)
        self.trans_attention = nn.Linear(hidden_dim, 1)
    
    # forward pass takes current features of all places and transitions of the graph
    def forward(self, place_features: torch.Tensor, 
                transition_features: torch.Tensor,
                pre_edge_index: torch.Tensor, # which places connects to which transitions
                post_edge_index: torch.Tensor # which transitions connects to which places
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_places = place_features.size(0)
        num_transitions = transition_features.size(0)
        
        # Places -> Transitions : to accumulate messages to each transition
        trans_messages = torch.zeros(num_transitions, self.place_to_trans_msg.out_features, 
                                     device=place_features.device)
        
        if pre_edge_index.size(1) > 0:
            # the connected places to transitions
            place_idx = pre_edge_index[0]
            trans_idx = pre_edge_index[1]
            messages = self.place_to_trans_msg(place_features[place_idx])
            attention_scores = torch.softmax(self.trans_attention(messages), dim=0)
            weighted_messages = messages * attention_scores
            # for each transition , all the weighted messages it received are stored in the transition idx
            trans_messages.index_add_(0, trans_idx, weighted_messages)
        
        # Transitions -> Places
        place_messages = torch.zeros(num_places, self.trans_to_place_msg.out_features,
                                     device=transition_features.device)
        
        if post_edge_index.size(1) > 0:
            trans_idx = post_edge_index[0]
            place_idx = post_edge_index[1]
            messages = self.trans_to_place_msg(transition_features[trans_idx])
            attention_scores = torch.softmax(self.place_attention(messages), dim=0)
            weighted_messages = messages * attention_scores
            place_messages.index_add_(0, place_idx, weighted_messages)
        
        # Update features
        place_features_new = self.place_update(
            torch.cat([place_features, place_messages], dim=1)
        )
        transition_features_new = self.trans_update(
            torch.cat([transition_features, trans_messages], dim=1)
        )
        
        # relu witha residual connection 
        place_features = F.relu(place_features + place_features_new)
        transition_features = F.relu(transition_features + transition_features_new)
        
        return place_features, transition_features


class SequentialConformanceChecker(nn.Module):
    """
    Sequential conformance checker - examines each transition step-by-step
    -> NO CHEATING: Doesn't see the full enablement vector upfront 
    """
    
    def __init__(self, hidden_dim: int, num_transitions: int, dropout: float = 0.3):
        super(SequentialConformanceChecker, self).__init__()
        
        self.num_transitions = num_transitions
        
        # Per-step conformance classifier
        # Input: graph context + transition embedding + predicted probability + marking state
        self.step_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3 + num_transitions + 1, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Conformance score for this step
        )
        
        # Optional: Learn to aggregate step scores
        self.aggregator = nn.Linear(1, 1)  # Can be extended to attention-based
    
    def check_transition_enabled(self, trans_idx: int, marking: torch.Tensor,
                                 pre_edge_index: torch.Tensor) -> bool:
        """Check if a specific transition is enabled"""
        # Find input places for this transition
        input_places = pre_edge_index[0][pre_edge_index[1] == trans_idx]
        
        if len(input_places) == 0:
            return True  # No prerequisites
        
        # All input places must have tokens
        return torch.all(marking[input_places] > 0).item()
    
    def fire_transition(self, trans_idx: int, marking: torch.Tensor,
                       pre_edge_index: torch.Tensor, 
                       post_edge_index: torch.Tensor) -> torch.Tensor:
        """Simulate firing a transition (Petri net semantics)"""
        new_marking = marking.clone()
        
        # Remove tokens from input places
        input_places = pre_edge_index[0][pre_edge_index[1] == trans_idx]
        if len(input_places) > 0:
            new_marking[input_places] -= 1
        
        # Add tokens to output places
        output_places = post_edge_index[1][post_edge_index[0] == trans_idx]
        if len(output_places) > 0:
            new_marking[output_places] += 1
        
        return new_marking
    
    def forward(self, combined_features: torch.Tensor,
                predicted_transitions: torch.Tensor,
                initial_marking: torch.Tensor,
                pre_edge_index: torch.Tensor,
                post_edge_index: torch.Tensor,
                threshold: float = 0.5) -> Tuple[torch.Tensor, List[dict]]:
        """
        Sequential conformance checking
        
        Returns:
            - conformance: [1] overall conformance score
            - step_info: List of dicts with info about each step (for debugging)
        """
        # Get predicted transitions above threshold
        predicted_indices = (predicted_transitions > threshold).nonzero(as_tuple=True)[0]
        
        if len(predicted_indices) == 0:
            # No transitions predicted - check if this is correct
            # If marking has enabled transitions but nothing predicted, that's suspicious
            any_enabled = False
            for t in range(self.num_transitions):
                if self.check_transition_enabled(t, initial_marking, pre_edge_index):
                    any_enabled = True
                    break
            
            # If transitions are enabled but none predicted, give medium score
            score = torch.tensor([0.7 if not any_enabled else 0.5], 
                               device=combined_features.device)
            return score, [{"status": "no_predictions", "any_enabled": any_enabled}]
        
        # Check each predicted transition sequentially
        step_scores = []
        step_info = []
        current_marking = initial_marking.clone()
        
        for step_idx, trans_idx in enumerate(predicted_indices):
            # Check if this transition is enabled at current marking
            # each transition is checked incrementaly from current marking
            is_enabled = self.check_transition_enabled(trans_idx, current_marking, pre_edge_index)
            
            # Create input for classifier
            trans_one_hot = torch.zeros(self.num_transitions, device=combined_features.device)
            trans_one_hot[trans_idx] = 1.0
            
            step_input = torch.cat([
                combined_features,                                    # Graph context
                trans_one_hot,                                        # Which transition
                predicted_transitions[trans_idx].unsqueeze(0)         # Prediction confidence
            ], dim=0)
            
            # Classify this step
            step_score = torch.sigmoid(
                self.step_classifier(step_input.unsqueeze(0))
            ).squeeze()
            
            # If not enabled, penalize the score
            if not is_enabled:
                step_score = step_score * 0.1  # Heavy penalty for violation 
                # but violations can be too bad or moderate , or in another case for example , 
                # if many succesive violations happen penalty has to get higher 
                # Or depending on data , violations paths that happen so model will predict them as likely to happen 
                # should get higher penalty ; how is does this help ? => from a dataset model will learn fast what violations paths happen more 
                # it will converge on classifying them as non conformant with a good confidence 
                # modeling probabilities of paths from data (HMM)
            
            """
            here i don't update the current marking for this sequence once a deviation happens 
            and this doesn't let the model learn that if a deviation happens cascading effects happen
            or 
            it is better to just catch only the first deviation in the predicteed sequence and report it 
            to the system => it can tell where the deviation will start

            the other case : 
            shows cascading effects: "If this violation happens, what breaks next?

            => we can do another variable for the what if marking
            """


            step_scores.append(step_score)
            step_info.append({
                "transition": trans_idx.item(),
                "enabled": is_enabled,
                "predicted_prob": predicted_transitions[trans_idx].item(),
                "conformance_score": step_score.item()
            })
            
            # Simulate firing if enabled (for next step)
            if is_enabled:
                current_marking = self.fire_transition(
                    trans_idx, current_marking, pre_edge_index, post_edge_index
                )
        
        # Aggregate scores (minimum = weakest link)
        if len(step_scores) > 0:
            conformance = torch.stack(step_scores).min().unsqueeze(0)
        else:
            conformance = torch.tensor([1.0], device=combined_features.device)
        
        return conformance, step_info


# we want a model that predicts which transitions will fire next in the process graph 
# and checks if they are valid 
class ConformanceGNN(nn.Module):
    """
    Two-Stage GNN with Sequential Conformance Checking
    """
    
    def __init__(self, 
                place_feature_dim: int = 1, # token count of a place node
                transition_feature_dim: int = 8, # one-hot encoding of the transition
                prefix_encoding_dim: int = 18, # encoding the past sequence as 18 dim embedding
                hidden_dim: int = 64, 
                num_gnn_layers: int = 3, # num of messages passing rounds
                num_transitions: int = 8,
                dropout: float = 0.3):
        super(ConformanceGNN, self).__init__()
        
        self.num_transitions = num_transitions
        self.hidden_dim = hidden_dim
        
        # Input projections
        self.place_embedding = nn.Linear(place_feature_dim, hidden_dim)
        self.transition_embedding = nn.Linear(transition_feature_dim, hidden_dim)
        self.prefix_embedding = nn.Linear(prefix_encoding_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            HeteroGraphConv(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_gnn_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Readout/pooling
        self.place_pooling = nn.Linear(hidden_dim, hidden_dim)
        self.transition_pooling = nn.Linear(hidden_dim, hidden_dim)
        
        # STAGE 1: Transition predictor
        self.transition_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), # place pooling + transition pooling + prefix encoding
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_transitions)
        )
        
        # STAGE 2: Sequential conformance checker
        self.conformance_checker = SequentialConformanceChecker(
            hidden_dim, num_transitions, dropout
        )
    
    def compute_enabled_transitions(self, place_features: torch.Tensor,
                                   pre_edge_index: torch.Tensor) -> torch.Tensor:
        """Compute enablement vector for ALL transitions"""
        num_transitions = self.num_transitions
        enabled = torch.ones(num_transitions, device=place_features.device)
        
        # only compute if there are edges connecting places to transitions
        if pre_edge_index.size(1) > 0:
            place_idx = pre_edge_index[0]
            trans_idx = pre_edge_index[1]
            marking = place_features.squeeze(-1) # count of each place tokens
            
            for t in range(num_transitions):
                input_places = place_idx[trans_idx == t] # for each transition , finds all places that are input for it 

                if len(input_places) > 0:
                    # a transition can only fire if all required places have > 0 token
                    enabled[t] = torch.all(marking[input_places] > 0).float()
        
        return enabled
    
    def forward(self, 
                place_features: torch.Tensor,
                transition_features: torch.Tensor,
                prefix_encoding: torch.Tensor,
                pre_edge_index: torch.Tensor,
                post_edge_index: torch.Tensor,
                return_step_info: bool = False) -> Tuple:
        """
        Forward pass with sequential conformance checking
        """
        # Embed inputs in hidden dim
        place_features_input = place_features.view(-1, 1)
        place_h = self.place_embedding(place_features_input)
        transition_h = self.transition_embedding(transition_features)
        prefix_h = self.prefix_embedding(prefix_encoding.unsqueeze(0))
        
        # GNN message passing
        for gnn_layer in self.gnn_layers:
            # the built gnn layer for message passing returns places and transitions features in hidden dim
            place_h, transition_h = gnn_layer(
                place_h, transition_h, pre_edge_index, post_edge_index
            ) # 3 layers => we get information from 3 hops away
            place_h = self.dropout(place_h)
            transition_h = self.dropout(transition_h)
        
        # Graph-level pooling
        # mean across all places features => [1, 64]
        place_global = torch.mean(self.place_pooling(place_h), dim=0, keepdim=True)
        transition_global = torch.mean(self.transition_pooling(transition_h), dim=0, keepdim=True)
        combined = torch.cat([place_global, transition_global, prefix_h], dim=1)
        
        # STAGE 1: Predict transitions
        next_transitions_logits = self.transition_predictor(combined).squeeze(0)
        next_transitions = torch.sigmoid(next_transitions_logits) # probability each transition will fire
        
        # Get current marking for sequential checking
        initial_marking = place_features_input.squeeze(-1)
        
        # STAGE 2: Sequential conformance checking
        conformance, step_info = self.conformance_checker(
            combined.squeeze(0),
            next_transitions,
            initial_marking,
            pre_edge_index,
            post_edge_index
        )
        
        # Also compute enabled transitions for loss computation (ground truth)
        enabled_transitions = self.compute_enabled_transitions(
            place_features_input, pre_edge_index
        )
        
        if return_step_info:
            return next_transitions, conformance, enabled_transitions, step_info
            """Check for violations
            if conformance < 0.5: Low conformance = violation likely
                print("WARNING: Non-conformant sequence predicted!")
                
                for step in step_info:
                    if not step['enabled'] and step['predicted_prob'] > 0.5:
                        print(f"Transition {step['transition']} will likely fire (should not)")
                        Take preventive action here!"""
        else:
            return next_transitions, conformance, enabled_transitions


class ConformanceLoss(nn.Module):
    """Loss function for sequential conformance checking"""
    
    def __init__(self, 
                 transition_weight: float = 1.0,
                 conformance_weight: float = 1.0,
                 ):
        super(ConformanceLoss, self).__init__()
        
        self.transition_weight = transition_weight
        self.conformance_weight = conformance_weight
    
    def forward(self, pred_transitions, true_transitions, 
                pred_conformance, true_conformance,
                enabled_transitions=None):
        
        # for each transition in the predicted sequence we compare the pedicted probability to the label
        # then avg across all transitions
        """
        pedicted: [0.1, 0.8, 0.05, 0.9, 0.2, 0.6, 0.3, 0.1]
        True   : [0,   1,   0,    1,   0,   1,   0,   0  ]
        """
        transition_loss = F.binary_cross_entropy(pred_transitions, true_transitions)
        
        pred_conf = pred_conformance.view(-1)
        true_conf = true_conformance.view(-1)
        # training valid against non-valid sequence
        conformance_loss = F.binary_cross_entropy(pred_conf, true_conf)
        
        total_loss = (
            self.transition_weight * transition_loss + 
            self.conformance_weight * conformance_loss
        )
        
        return total_loss, transition_loss, conformance_loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("=" * 60)
    print("Sequential Conformance GNN - No Cheating Version")
    print("=" * 60)
    
    model = ConformanceGNN(
        place_feature_dim=1,
        transition_feature_dim=8,
        prefix_encoding_dim=18,
        hidden_dim=64,
        num_gnn_layers=3,
        num_transitions=8,
        dropout=0.3
    )
    
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    # Test with scenario: p1 has token, predict t1 (requires p1) - should be conformant
    print("\n" + "="*60)
    print("Test 1: Enabled transition predicted")
    print("="*60)
    place_features = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    transition_features = torch.randn(8, 8)
    prefix_encoding = torch.randn(18)
    
    pre_arcs = [[0, 0], [1, 1], [2, 2], [1, 3], [4, 3], [3, 4], [4, 4], [5, 5], [5, 6], [5, 7]]
    post_arcs = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 5], [5, 1], [5, 2], [6, 6], [7, 6]]
    
    pre_edge_index = torch.tensor(pre_arcs, dtype=torch.long).t().contiguous()
    post_edge_index = torch.tensor(post_arcs, dtype=torch.long).t().contiguous()
    
    with torch.no_grad():
        # Manually set high prediction for t1 (enabled)
        model.eval()
        next_trans, conformance, enabled, step_info = model(
            place_features, transition_features, prefix_encoding,
            pre_edge_index, post_edge_index, return_step_info=True
        )
        
        # Force t1 to be predicted
        next_trans = torch.zeros(8)
        next_trans[0] = 0.9  # Predict t1 with high confidence
        
        conformance, step_info = model.conformance_checker(
            torch.randn(64 * 3),
            next_trans,
            place_features,
            pre_edge_index,
            post_edge_index
        )
    
    print(f"Predicted transitions: {next_trans}")
    print(f"Enabled transitions: {enabled}")
    print(f"Conformance score: {conformance.item():.4f}")
    print(f"\nStep-by-step info:")
    for i, info in enumerate(step_info):
        print(f"  Step {i+1}: {info}")
    
    print("\n" + "="*60)
    print("Test 2: Disabled transition predicted (violation)")
    print("="*60)
    
    with torch.no_grad():
        next_trans = torch.zeros(8)
        next_trans[1] = 0.9  # Predict t2 but p2 has no token!
        
        conformance, step_info = model.conformance_checker(
            torch.randn(64 * 3),
            next_trans,
            place_features,
            pre_edge_index,
            post_edge_index
        )
    
    print(f"Predicted transitions: {next_trans}")
    print(f"Place features (marking): {place_features}")
    print(f"Conformance score: {conformance.item():.4f}")
    print(f"\nStep-by-step info:")
    for i, info in enumerate(step_info):
        print(f"  Step {i+1}: {info}")
    
    print("\n" + "="*60)
    print("Sequential checking test completed!")
    print("="*60)