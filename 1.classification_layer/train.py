"""
Complete Training Script for Heterogeneous GNN-based Conformance Checking

This script trains the ConformanceGNN model on the generated dataset.
It handles HeteroData format with proper batching and evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader
import polars as pl
import json
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Import the model
from gnn_model import ConformanceGNN, create_hetero_data_from_sample


# ============================================================================
# Dataset Class for HeteroData
# ============================================================================
class ConformanceHeteroDataset(Dataset):
    """
    PyTorch Dataset that returns HeteroData objects
    Compatible with PyG DataLoader
    """
    
    def __init__(self, data_path):
        """
        Args:
            data_path: Path to parquet or json file
        """
        # Load from Parquet
        df = pl.read_parquet(data_path)
        self.samples = []
        for row in df.iter_rows(named=True):
            marking = [row['pi'], row['p1'], row['p2'], row['p3'],
                        row['p4'], row['p5'], row['po']]
            event = row['event']
            label = row['label']
            self.samples.append((marking, event, label))
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        marking, event, label = self.samples[idx]
        # Convert to HeteroData
        hetero_data = create_hetero_data_from_sample(marking, event, label)
        return hetero_data


# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        batch = batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(batch)
        
        # Handle batched data
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        
        loss = criterion(logits, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += batch.y.size(0)
        correct += (predicted == batch.y).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device, return_predictions=False):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            batch = batch.to(device)
            
            logits = model(batch)
            
            # Handle batched data
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)
            
            loss = criterion(logits, batch.y)
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'loss': total_loss / len(loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }
    
    if return_predictions:
        return metrics, all_preds, all_labels
    
    return metrics


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(history['val_f1'], label='Validation F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Precision & Recall
    axes[1, 1].plot(history['val_precision'], label='Precision')
    axes[1, 1].plot(history['val_recall'], label='Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision and Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    classes = ['Conformant', 'Non-Conformant']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


# ============================================================================
# Main Training Loop
# ============================================================================
def main():
    # ========================================================================
    # Configuration
    # ========================================================================
    config = {
        # Data
        'data_path': 'conformance_balanced.parquet',  
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        
        # Model
        'hidden_dim': 64,
        'num_layers': 2,
        
        # Training
        'batch_size': 1,  # HeteroData doesn't batch well, use 1
        'learning_rate': 0.001,
        'num_epochs': 100,
        'patience': 15,  # Early stopping patience
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Saving
        'save_dir': '/kaggle/working/checkpoints',
        'model_name': 'conformance_gnn_best.pth'
    }
    
    print("=" * 80)
    print("Training Heterogeneous GNN for Online Conformance Checking")
    print("=" * 80)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # ========================================================================
    # Load Dataset
    # ========================================================================
    print("Loading dataset...")
    dataset = ConformanceHeteroDataset(
        config['data_path']
    )
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(config['train_split'] * total_size)
    val_size = int(config['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    print()
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # ========================================================================
    # Initialize Model
    # ========================================================================
    print("Initializing model...")
    device = torch.device(config['device'])
    model = ConformanceGNN(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    print()
    
    # ========================================================================
    # Training Setup
    # ========================================================================
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    print("Starting training...")
    print("=" * 80)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(f"Confusion Matrix: TP={val_metrics['tp']}, FP={val_metrics['fp']}, TN={val_metrics['tn']}, FN={val_metrics['fn']}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            save_path = os.path.join(config['save_dir'], config['model_name'])
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'config': config
            }, save_path)
            print(f" Saved best model (F1: {best_val_f1:.4f}) to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config['patience']}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # ========================================================================
    # Final Evaluation on Test Set
    # ========================================================================
    print("\n" + "=" * 80)
    print("Final Evaluation on Test Set")
    print("=" * 80)
    
    # Load best model
    checkpoint = torch.load(os.path.join(config['save_dir'], config['model_name']), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
    
    # Evaluate
    test_metrics, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, return_predictions=True
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {test_metrics['tp']}")
    print(f"  False Positives: {test_metrics['fp']}")
    print(f"  True Negatives:  {test_metrics['tn']}")
    print(f"  False Negatives: {test_metrics['fn']}")
    
    # Detailed classification report
    print(f"\nClassification Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=['Conformant', 'Non-Conformant']
    ))
    
    # ========================================================================
    # Save Plots
    # ========================================================================
    print("\nSaving plots...")
    plot_training_history(history, os.path.join(config['save_dir'], 'training_history.png'))
    plot_confusion_matrix(test_metrics['confusion_matrix'], os.path.join(config['save_dir'], 'confusion_matrix.png'))
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Model saved to: {os.path.join(config['save_dir'], config['model_name'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()