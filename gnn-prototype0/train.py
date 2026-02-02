"""
Training Script for GNN-based Online Conformance Checking

This module implements:
1. Dataset loading and batching
2. Training loop with validation
3. Metrics computation (accuracy, precision, recall, F1)
4. Model checkpointing
5. Visualization of training progress
"""

import sys 
sys.path.append("/kaggle/working/GNN-classifer-for-an-event-stream")
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import os
import json

from gnn_model import ConformanceGNN, ConformanceLoss, count_parameters
from generate_dataset import ConformanceDatasetGenerator


class ConformanceDataset(Dataset):
    """PyTorch Dataset for conformance checking samples"""
    
    def __init__(self, samples: List[Dict]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching
    Since each sample has the same graph structure, we can batch them
    """
    # Stack features
    place_features = torch.stack([sample['place_features'].squeeze() for sample in batch])
    transition_features = torch.stack([sample['transition_features'] for sample in batch])
    prefix_encoding = torch.stack([sample['prefix_encoding'] for sample in batch])
    
    # Labels
    next_transitions = torch.stack([sample['next_transitions'] for sample in batch])
    is_conformant = torch.stack([sample['is_conformant'] for sample in batch])
    
    # Edge indices are the same for all samples
    pre_edge_index = batch[0]['pre_edge_index']
    post_edge_index = batch[0]['post_edge_index']
    
    return {
        'place_features': place_features,
        'transition_features': transition_features,
        'prefix_encoding': prefix_encoding,
        'pre_edge_index': pre_edge_index,
        'post_edge_index': post_edge_index,
        'next_transitions': next_transitions,
        'is_conformant': is_conformant
    }


class MetricsTracker:
    """Track and compute metrics during training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.transition_preds = []
        self.transition_labels = []
        self.conformance_preds = []
        self.conformance_labels = []
        self.losses = []
    
    def update(self, pred_transitions, true_transitions, 
               pred_conformance, true_conformance, loss):
        """Update with batch predictions"""
        self.transition_preds.append(pred_transitions.detach().cpu())
        self.transition_labels.append(true_transitions.detach().cpu())
        self.conformance_preds.append(pred_conformance.detach().cpu())
        self.conformance_labels.append(true_conformance.detach().cpu())
        self.losses.append(loss.item())
    
    def compute(self, threshold: float = 0.5) -> Dict:
        """Compute metrics"""
        # Concatenate all batches
        trans_preds = torch.cat(self.transition_preds, dim=0).numpy()
        trans_labels = torch.cat(self.transition_labels, dim=0).numpy()
        conf_preds = torch.cat(self.conformance_preds, dim=0).numpy()
        conf_labels = torch.cat(self.conformance_labels, dim=0).numpy()
        
        # Threshold predictions
        trans_preds_binary = (trans_preds > threshold).astype(int)
        conf_preds_binary = (conf_preds > threshold).astype(int)
        
        # Transition prediction metrics (per-sample average)
        trans_accuracy = (trans_preds_binary == trans_labels).mean()
        
        # Conformance classification metrics
        conf_accuracy = accuracy_score(conf_labels, conf_preds_binary)
        conf_precision, conf_recall, conf_f1, _ = precision_recall_fscore_support(
            conf_labels, conf_preds_binary, average='binary', zero_division=0
        )
        
        # AUC if possible
        try:
            conf_auc = roc_auc_score(conf_labels, conf_preds)
        except:
            conf_auc = 0.0
        
        return {
            'loss': np.mean(self.losses),
            'transition_accuracy': trans_accuracy,
            'conformance_accuracy': conf_accuracy,
            'conformance_precision': conf_precision,
            'conformance_recall': conf_recall,
            'conformance_f1': conf_f1,
            'conformance_auc': conf_auc
        }


class Trainer:
    """Training manager for conformance GNN"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: ConformanceLoss,
                 device: str = 'cpu',
                 checkpoint_dir: str = './checkpoints'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        metrics_tracker = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move to device
            place_features = batch['place_features'].to(self.device)
            transition_features = batch['transition_features'].to(self.device)
            prefix_encoding = batch['prefix_encoding'].to(self.device)
            pre_edge_index = batch['pre_edge_index'].to(self.device)
            post_edge_index = batch['post_edge_index'].to(self.device)
            next_transitions = batch['next_transitions'].to(self.device)
            is_conformant = batch['is_conformant'].to(self.device)
            
            # Forward pass for each sample in batch
            batch_size = place_features.size(0)
            total_loss = 0
            
            for i in range(batch_size):
                pred_trans, pred_conf = self.model(
                    place_features[i],
                    transition_features[i],
                    prefix_encoding[i],
                    pre_edge_index,
                    post_edge_index
                )
                
                # Compute loss
                loss, _, _ = self.loss_fn(
                    pred_trans, next_transitions[i],
                    pred_conf, is_conformant[i]
                )
                
                total_loss += loss
                
                # Track metrics
                metrics_tracker.update(
                    pred_trans.unsqueeze(0), next_transitions[i].unsqueeze(0),
                    pred_conf.unsqueeze(0), is_conformant[i].unsqueeze(0),
                    loss
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            (total_loss / batch_size).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{total_loss.item()/batch_size:.4f}'})
        
        return metrics_tracker.compute()
    
    def validate(self) -> Dict:
        """Validate the model"""
        self.model.eval()
        metrics_tracker = MetricsTracker()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                place_features = batch['place_features'].to(self.device)
                transition_features = batch['transition_features'].to(self.device)
                prefix_encoding = batch['prefix_encoding'].to(self.device)
                pre_edge_index = batch['pre_edge_index'].to(self.device)
                post_edge_index = batch['post_edge_index'].to(self.device)
                next_transitions = batch['next_transitions'].to(self.device)
                is_conformant = batch['is_conformant'].to(self.device)
                
                batch_size = place_features.size(0)
                
                for i in range(batch_size):
                    pred_trans, pred_conf = self.model(
                        place_features[i],
                        transition_features[i],
                        prefix_encoding[i],
                        pre_edge_index,
                        post_edge_index
                    )
                    
                    # Compute loss
                    loss, _, _ = self.loss_fn(
                        pred_trans, next_transitions[i],
                        pred_conf, is_conformant[i]
                    )
                    
                    # Track metrics
                    metrics_tracker.update(
                        pred_trans.unsqueeze(0), next_transitions[i].unsqueeze(0),
                        pred_conf.unsqueeze(0), is_conformant[i].unsqueeze(0),
                        loss
                    )
        
        return metrics_tracker.compute()
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        # Save regular checkpoint
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  → Saved best model (val_loss: {metrics['loss']:.4f})")
    
    def train(self, num_epochs: int):
        """Full training loop"""
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Store history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # Print metrics
            print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
            print(f"  Transition Acc: {train_metrics['transition_accuracy']:.4f}")
            print(f"  Conformance Acc: {train_metrics['conformance_accuracy']:.4f}")
            print(f"  Conformance F1: {train_metrics['conformance_f1']:.4f}")
            
            print(f"\nVal Loss: {val_metrics['loss']:.4f}")
            print(f"  Transition Acc: {val_metrics['transition_accuracy']:.4f}")
            print(f"  Conformance Acc: {val_metrics['conformance_accuracy']:.4f}")
            print(f"  Conformance F1: {val_metrics['conformance_f1']:.4f}")
            print(f"  Conformance AUC: {val_metrics['conformance_auc']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(epoch, val_metrics, is_best)
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
    
    def plot_history(self, save_path: str = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Transition accuracy
        train_trans_acc = [m['transition_accuracy'] for m in self.history['train_metrics']]
        val_trans_acc = [m['transition_accuracy'] for m in self.history['val_metrics']]
        axes[0, 1].plot(train_trans_acc, label='Train')
        axes[0, 1].plot(val_trans_acc, label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Transition Prediction Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Conformance accuracy
        train_conf_acc = [m['conformance_accuracy'] for m in self.history['train_metrics']]
        val_conf_acc = [m['conformance_accuracy'] for m in self.history['val_metrics']]
        axes[1, 0].plot(train_conf_acc, label='Train')
        axes[1, 0].plot(val_conf_acc, label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Conformance Classification Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # F1 score
        train_f1 = [m['conformance_f1'] for m in self.history['train_metrics']]
        val_f1 = [m['conformance_f1'] for m in self.history['val_metrics']]
        axes[1, 1].plot(train_f1, label='Train')
        axes[1, 1].plot(val_f1, label='Validation')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Conformance F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


def main():
    """Main training function"""
    
    # Configuration
    config = {
        'window_size': 3,
        'num_traces': 2000,
        'deviation_ratio': 0.3,
        'batch_size': 32,
        'hidden_dim': 64,
        'num_gnn_layers': 3,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'train_split': 0.8,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Configuration:")
    print(json.dumps(config, indent=2))
    
    # Generate or load dataset
    dataset_path = '/kaggle/working/conformance_dataset.pkl'
    
    if not os.path.exists(dataset_path):
        print("\nGenerating dataset...")
        generator = ConformanceDatasetGenerator(window_size=config['window_size'])
        dataset = generator.generate_dataset(
            num_traces=config['num_traces'],
            deviation_ratio=config['deviation_ratio']
        )
        generator.save_dataset(dataset, dataset_path)
    else:
        print(f"\nLoading dataset from {dataset_path}...")
        generator = ConformanceDatasetGenerator(window_size=config['window_size'])
        dataset = generator.load_dataset(dataset_path)
    
    # Split dataset
    train_size = int(config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    
    # Shuffle dataset
    indices = torch.randperm(len(dataset)).tolist()
    train_dataset = [dataset[i] for i in indices[:train_size]]
    val_dataset = [dataset[i] for i in indices[train_size:]]
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        ConformanceDataset(train_dataset),
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        ConformanceDataset(val_dataset),
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    model = ConformanceGNN(
        place_feature_dim=1,
        transition_feature_dim=8,
        prefix_encoding_dim=config['window_size'] * 6,  # window_size * num_activities
        hidden_dim=config['hidden_dim'],
        num_gnn_layers=config['num_gnn_layers'],
        num_transitions=8,
        dropout=config['dropout']
    )
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = ConformanceLoss(transition_weight=1.0, conformance_weight=1.0)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=config['device'],
        checkpoint_dir='kaggle/wokring/checkpoints'
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'])
    
    # Plot results
    trainer.plot_history(save_path='/kaggle/working/training_history.png')
    
    # Save config
    with open('checkpoints/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nTraining complete! Best model saved to ./checkpoints/best_model.pt")


if __name__ == '__main__':
    main()