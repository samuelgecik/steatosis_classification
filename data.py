import os
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

class SteatosisDataset(Dataset):
    """Dataset for liver steatosis classification."""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 binary: bool = True,
                 transform = None) -> None:
        """
        Args:
            data_dir (str): Path to the dataset root directory
            split (str): 'train' or 'test' split
            binary (bool): If True, converts to binary classification 
                         (Normal vs. Steatosis)
            transform: Optional transform to be applied on images
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.binary = binary
        self.transform = transform

        # Get all image paths and labels
        self.image_paths, self.labels = self._load_dataset()
        
        # Convert labels to binary if specified
        if binary:
            self.labels = [0 if label == 'Normal' else 1 for label in self.labels]
        
        # Calculate class weights for weighted sampling
        self._calculate_class_weights()

    def _load_dataset(self) -> Tuple[List[Path], List[str]]:
        """Load dataset paths and labels."""
        image_paths = []
        labels = []
        
        # Define the dataset structure
        split_dir = self.data_dir / 'DataSet' / self.split
        
        # Load images from each class directory
        for class_name in ['Normal', 'Mild', 'Moderate', 'Severe']:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
                
            # Get all PNG files (excluding .Zone.Identifier files)
            class_files = [f for f in class_dir.glob('*.png')]
            
            image_paths.extend(class_files)
            labels.extend([class_name] * len(class_files))
        
        return image_paths, labels

    def _calculate_class_weights(self) -> None:
        """Calculate class weights for weighted sampling."""
        if self.binary:
            # Calculate weights for binary classification
            label_counts = np.bincount(self.labels)
            total_samples = len(self.labels)
            self.class_weights = torch.FloatTensor(
                total_samples / (len(label_counts) * label_counts)
            )
        else:
            # Calculate weights for multi-class classification
            labels_set = sorted(set(self.labels))
            label_counts = {label: self.labels.count(label) 
                          for label in labels_set}
            total_samples = len(self.labels)
            self.class_weights = torch.FloatTensor([
                total_samples / (len(label_counts) * label_counts[label]) 
                for label in labels_set
            ])

    def __len__(self) -> int:
        """Return the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label at index idx."""
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            
        # Get label
        label = self.labels[idx]
        
        return image, label

def get_transforms(split: str) -> transforms.Compose:
    """
    Get image transformations for training/testing.
    
    Args:
        split (str): 'train' or 'test' split
    
    Returns:
        transforms.Compose: Composition of transforms
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Standard input size
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    binary: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and testing DataLoaders.
    
    Args:
        data_dir (str): Path to dataset root directory
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        binary (bool): If True, creates binary classification dataloaders
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = SteatosisDataset(
        data_dir=data_dir,
        split='train',
        binary=binary,
        transform=get_transforms('train')
    )
    
    test_dataset = SteatosisDataset(
        data_dir=data_dir,
        split='test',
        binary=binary,
        transform=get_transforms('test')
    )
    
    # Create samplers for weighted sampling
    train_weights = [train_dataset.class_weights[label] 
                    for label in train_dataset.labels]
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def get_dataset_info(data_dir: str) -> None:
    """Print dataset information."""
    for split in ['train', 'test']:
        dataset = SteatosisDataset(data_dir, split=split, binary=True)
        print(f"\n{split.capitalize()} set:")
        if dataset.binary:
            normal_count = dataset.labels.count(0)
            steatosis_count = dataset.labels.count(1)
            print(f"Normal: {normal_count}")
            print(f"Steatosis: {steatosis_count}")
        else:
            for label in sorted(set(dataset.labels)):
                count = dataset.labels.count(label)
                print(f"{label}: {count}")

if __name__ == '__main__':
    # Example usage
    data_dir = "DataSet"
    
    # Print dataset information
    print("Dataset Information:")
    get_dataset_info(data_dir)
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(data_dir, binary=True)
    
    # Print sample batch
    images, labels = next(iter(train_loader))
    print(f"\nSample batch shape: {images.shape}")
    print(f"Sample labels: {labels.numpy()}")