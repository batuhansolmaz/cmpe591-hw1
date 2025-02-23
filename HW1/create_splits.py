import torch
from torch.utils.data import random_split
from models import PushDataset
import os

def create_data_splits(data_dir='push_data'):
    """
    Create and save train/val/test splits to be used by all models.
    """
    # Load dataset
    dataset = PushDataset(data_dir)
    
    # Create splits
    test_size = int(0.2 * len(dataset))  # 20% for test
    train_val_size = len(dataset) - test_size
    train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])
    
    # Split train into train and validation
    train_size = int(0.8 * train_val_size)  # 80% of remaining data
    val_size = train_val_size - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    
    # Save indices
    splits = {
        'train_indices': train_dataset.indices,
        'val_indices': val_dataset.indices,
        'test_indices': test_dataset.indices
    }
    
    torch.save(splits, 'data_splits.pt')
    print(f"Created splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

if __name__ == "__main__":
    create_data_splits() 