"""
GNN Model Architecture for Online Conformance Checking - CORRECTED

This module implements a two-stage heterogeneous GNN:
1. Stage 1: Predict next enabled transitions (multi-label classification)
2. Stage 2: Check conformance by validating predicted transitions against Petri net structure

Key corrections:
- Conformance classifier now takes predicted transitions as input
- Added enablement checking logic based on current marking
- Proper two-stage forward pass
"""
import sys 
sys.path.append("/kaggle/working/GNN-classifer-for-an-event-stream")
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class HeteroGraphConv(nn.Module):
    """
    Heterogeneous Graph Convolution Layer
    Processes place->transition and transition->place message passing
    """
    
    def __init__(self, place_dim: int, transition_dim: int, hidden_dim: int):
        super(HeteroGraphConv, self).__init__()
        
        # Message functions for different edge types
        self.place_to_trans_msg = nn.Linear(place_dim, hidden_dim)
        self.trans_to_place_msg = nn.Linear(transition_dim, hidden_dim)
        
        # Update functions
        self.place_update = nn.Linear(place_dim + hidden_dim, place_dim)
        self.trans_update = nn.Linear(transition_dim + hidden_dim, transition_dim)
        
        # Attention weights
        self.place_attention = nn.Linear(hidden_dim, 1)
        self.trans_attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, place_features: torch.Tensor, 
                transition_features: torch.Tensor,
                pre_edge_index: torch.Tensor,
                post_edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            place_features: [num_places, place_dim]
            transition_features: [num_transitions, transition_dim]
            pre_edge_index: [2, num_pre_edges] - (place, transition)
            post_edge_index: [2, num_post_edges] - (transition, place)
        
        Returns:
            Updated place and transition features
        """
        num_places = place_features.size(0)
        num_transitions = transition_features.size(0)
        
        # Message passing: Places -> Transitions (via pre-arcs)
        trans_messages = torch.zeros(num_transitions, self.place_to_trans_msg.out_features, 
                                     device=place_features.device)
        
        if pre_edge_index.size(1) > 0:
            place_idx = pre_edge_index[0]
            trans_idx = pre_edge_index[1]
            
            # Compute messages
            messages = self.place_to_trans_msg(place_features[place_idx])
            
            # Attention-based aggregation
            attention_scores = torch.softmax(self.trans_attention(messages), dim=0)
            weighted_messages = messages * attention_scores
            
            # Aggregate messages for each transition
            trans_messages.index_add_(0, trans_idx, weighted_messages)
        
        # Message passing: Transitions -> Places (via post-arcs)
        place_messages = torch.zeros(num_places, self.trans_to_place_msg.out_features,
                                     device=transition_features.device)
        
        if post_edge_index.size(1) > 0:
            trans_idx = post_edge_index[0]
            place_idx = post_edge_index[1]
            
            # Compute messages
            messages = self.trans_to_place_msg(transition_features[trans_idx])
            
            # Attention-based aggregation
            attention_scores = torch.softmax(self.place_attention(messages), dim=0)
            weighted_messages = messages * attention_scores
            
            # Aggregate messages for each place
            place_messages.index_add_(0, place_idx, weighted_messages)
        
        # Update features
        place_features_new = self.place_update(
            torch.cat([place_features, place_messages], dim=1)
        )
        transition_features_new = self.trans_update(
            torch.cat([transition_features, trans_messages], dim=1)
        )
        
        # Residual connections + activation
        place_features = F.relu(place_features + place_features_new)
        transition_features = F.relu(transition_features + transition_features_new)
        
        return place_features, transition_features


class ConformanceGNN(nn.Module):
    """
    Two-Stage GNN model for conformance checking
    
    Stage 1: Predict next transitions (multi-label)
    Stage 2: Check conformance by validating predictions against Petri net structure
    """
    
    def __init__(self, 
                 place_feature_dim: int = 1,
                 transition_feature_dim: int = 8,
                 prefix_encoding_dim: int = 18,  # window_size * num_activities
                 hidden_dim: int = 64,
                 num_gnn_layers: int = 3,
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
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Readout/pooling layers
        self.place_pooling = nn.Linear(hidden_dim, hidden_dim)
        self.transition_pooling = nn.Linear(hidden_dim, hidden_dim)
        
        # STAGE 1: Next transition prediction (multi-label)
        self.transition_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # Combined features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_transitions)  # Multi-label output
        )
        
        # STAGE 2: Conformance classification
        # Takes: graph features + prefix + predicted transitions + enablement check
        self.conformance_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3 + num_transitions + num_transitions, hidden_dim * 2),
            #         ^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^
            #         graph representation   predicted trans  enablement vector
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Binary output
        )
    
    def compute_enabled_transitions(self,
                                    place_features: torch.Tensor,
                                    pre_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute which transitions are enabled based on current marking
        
        Args:
            place_features: [num_places, 1] - marking (tokens)
            pre_edge_index: [2, num_pre_edges] - (place, transition) connections
        
        Returns:
            enabled: [num_transitions] - 1 if enabled, 0 if not
        """
        num_transitions = self.num_transitions
        enabled = torch.ones(num_transitions, device=place_features.device)
        
        if pre_edge_index.size(1) > 0:
            place_idx = pre_edge_index[0]
            trans_idx = pre_edge_index[1]
            
            # Extract marking (tokens in places)
            marking = place_features.squeeze(-1)  # [num_places]
            
            # For each transition, check if all input places have tokens
            for t in range(num_transitions):
                # Find input places for this transition
                input_places = place_idx[trans_idx == t]
                
                if len(input_places) > 0:
                    # Transition is enabled only if ALL input places have tokens
                    enabled[t] = torch.all(marking[input_places] > 0).float()
        
        return enabled
    
    def forward(self, 
                place_features: torch.Tensor,
                transition_features: torch.Tensor,
                prefix_encoding: torch.Tensor,
                pre_edge_index: torch.Tensor,
                post_edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Two-stage forward pass
        
        Args:
            place_features: [num_places, place_feature_dim]
            transition_features: [num_transitions, transition_feature_dim]
            prefix_encoding: [prefix_encoding_dim]
            pre_edge_index: [2, num_pre_edges]
            post_edge_index: [2, num_post_edges]
        
        Returns:
            - next_transitions: [num_transitions] - probabilities for each transition
            - conformance: [1] - conformance probability
            - enabled_transitions: [num_transitions] - which transitions are actually enabled
        """
        # Embed inputs
        place_features_input = place_features.view(-1, 1)
        place_h = self.place_embedding(place_features_input)
        transition_h = self.transition_embedding(transition_features)
        prefix_h = self.prefix_embedding(prefix_encoding.unsqueeze(0))  # [1, hidden_dim]
        
        # GNN message passing
        for gnn_layer in self.gnn_layers:
            place_h, transition_h = gnn_layer(
                place_h, transition_h, pre_edge_index, post_edge_index
            )
            place_h = self.dropout(place_h)
            transition_h = self.dropout(transition_h)
        
        # Graph-level pooling (mean pooling)
        place_global = torch.mean(self.place_pooling(place_h), dim=0, keepdim=True)  # [1, hidden_dim]
        transition_global = torch.mean(self.transition_pooling(transition_h), dim=0, keepdim=True)  # [1, hidden_dim]
        
        # Combine all representations
        combined = torch.cat([place_global, transition_global, prefix_h], dim=1)  # [1, hidden_dim * 3]
        
        # STAGE 1: Predict next transitions
        next_transitions_logits = self.transition_predictor(combined).squeeze(0)  # [num_transitions]
        next_transitions = torch.sigmoid(next_transitions_logits)
        
        # Compute which transitions are actually enabled based on current marking
        enabled_transitions = self.compute_enabled_transitions(place_features_input, pre_edge_index)
        
        # STAGE 2: Conformance classification
        # Combine: graph representation + predicted transitions + enablement check
        conformance_input = torch.cat([
            combined.squeeze(0),           # [hidden_dim * 3] - graph context
            next_transitions,              # [num_transitions] - what model predicts
            enabled_transitions            # [num_transitions] - what's actually enabled
        ], dim=0)  # [hidden_dim * 3 + 2 * num_transitions]
        
        conformance_logit = self.conformance_classifier(conformance_input.unsqueeze(0))  # [1, 1]
        conformance = torch.sigmoid(conformance_logit).squeeze(0)  # [1] - keep as 1D tensor
        
        return next_transitions, conformance, enabled_transitions


class ConformanceLoss(nn.Module):
    """
    Combined loss function for two-stage multi-task learning
    
    Key improvement: Conformance loss considers enablement violations
    """
    
    def __init__(self, 
                 transition_weight: float = 1.0,
                 conformance_weight: float = 1.0,
                 enablement_penalty: float = 0.5):
        super(ConformanceLoss, self).__init__()
        
        self.transition_weight = transition_weight
        self.conformance_weight = conformance_weight
        self.enablement_penalty = enablement_penalty
    
    def forward(self,
                pred_transitions: torch.Tensor,
                true_transitions: torch.Tensor,
                pred_conformance: torch.Tensor,
                true_conformance: torch.Tensor,
                enabled_transitions: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            pred_transitions: Predicted transition probabilities [num_transitions]
            true_transitions: Ground truth transitions [num_transitions]
            pred_conformance: Predicted conformance probability [1] or scalar
            true_conformance: Ground truth conformance label [1] or scalar
            enabled_transitions: Which transitions are actually enabled (optional) [num_transitions]
        
        Returns:
            - total_loss
            - transition_loss
            - conformance_loss
            - enablement_violation_loss (if enabled_transitions provided)
        """
        # Transition prediction loss
        transition_loss = F.binary_cross_entropy(pred_transitions, true_transitions)
        
        # Conformance classification loss - ensure both are same shape
        pred_conf = pred_conformance.view(-1)  # Flatten to [1] or []
        true_conf = true_conformance.view(-1)  # Flatten to [1] or []
        conformance_loss = F.binary_cross_entropy(pred_conf, true_conf)
        
        # Enablement violation penalty
        # Penalize predicting transitions that are not enabled
        enablement_violation_loss = torch.tensor(0.0, device=pred_transitions.device)
        if enabled_transitions is not None:
            # Where enabled=0 but predicted high probability, add penalty
            violations = pred_transitions * (1 - enabled_transitions)
            enablement_violation_loss = violations.mean()
        
        # Combined loss
        total_loss = (
            self.transition_weight * transition_loss + 
            self.conformance_weight * conformance_loss +
            self.enablement_penalty * enablement_violation_loss
        )
        
        return total_loss, transition_loss, conformance_loss, enablement_violation_loss


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Example usage and testing
    print("=" * 60)
    print("Conformance GNN Model Architecture - CORRECTED")
    print("=" * 60)
    
    # Create model
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
    
    # Test forward pass with dummy data
    print("\nTesting two-stage forward pass...")
    
    # Create dummy inputs
    place_features = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])  # Marking
    transition_features = torch.randn(8, 8)  # 8 transitions
    prefix_encoding = torch.randn(18)  # window_size=3, num_activities=6
    
    # Edge indices (from build_n1_petri_net_structure)
    pre_arcs = [[0, 0], [1, 1], [2, 2], [1, 3], [4, 3], [3, 4], [4, 4], [5, 5], [5, 6], [5, 7]]
    post_arcs = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 5], [5, 1], [5, 2], [6, 6], [7, 6]]
    
    pre_edge_index = torch.tensor(pre_arcs, dtype=torch.long).t().contiguous()
    post_edge_index = torch.tensor(post_arcs, dtype=torch.long).t().contiguous()
    
    # Forward pass
    with torch.no_grad():
        next_trans, conformance, enabled = model(
            place_features, transition_features, prefix_encoding,
            pre_edge_index, post_edge_index
        )
    
    print(f"\nOutput shapes:")
    print(f"  Next transitions (predicted): {next_trans.shape} - expected: torch.Size([8])")
    print(f"  Conformance: {conformance.shape} - expected: torch.Size([1])")
    print(f"  Enabled transitions (actual): {enabled.shape} - expected: torch.Size([8])")
    
    print(f"\nPredicted transition probabilities: {next_trans}")
    print(f"Actually enabled transitions: {enabled}")
    print(f"Conformance probability: {conformance}")  # Now shows [1] tensor
    
    # Check for violations
    violations = (next_trans > 0.5) & (enabled == 0)
    print(f"\nPredicted disabled transitions (violations): {violations.nonzero().squeeze(-1).tolist()}")
    
    # Test loss
    print("\nTesting loss function...")
    true_transitions = torch.zeros(8)
    true_transitions[0] = 1.0  # t1 should fire next
    true_conformance = torch.tensor([1.0])  # Shape [1] to match model output
    
    loss_fn = ConformanceLoss(enablement_penalty=0.5)
    total_loss, trans_loss, conf_loss, enable_loss = loss_fn(
        next_trans, true_transitions,
        conformance, true_conformance,
        enabled
    )
    
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Transition loss: {trans_loss.item():.4f}")
    print(f"  Conformance loss: {conf_loss.item():.4f}")
    print(f"  Enablement violation loss: {enable_loss.item():.4f}")
    
    print("\nShape verification:")
    print(f"  conformance shape: {conformance.shape}")
    print(f"  true_conformance shape: {true_conformance.shape}")
    print(f"  Shapes match: {conformance.shape == true_conformance.shape}")
    
    print("\n" + "=" * 60)
    print("Model architecture test completed successfully!")
    print("=" * 60)