"""
Data utilities for loading and preprocessing Brain Tumor MRI images.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split


class BrainTumorDataLoader:
    """Load and preprocess Brain Tumor MRI Dataset."""
    
    def __init__(self, data_dir='data', img_size=(224, 224)):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Path to the data directory
            img_size (tuple): Target image size (height, width)
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.classes = []
        self.class_to_idx = {}
        
    def discover_classes(self):
        """Discover available classes from the data directory."""
        # Look for common dataset structures
        possible_dirs = [
            self.data_dir,
            self.data_dir / 'Training',
            self.data_dir / 'train',
            self.data_dir / 'data'
        ]
        
        for dir_path in possible_dirs:
            if dir_path.exists() and dir_path.is_dir():
                # Get subdirectories as classes
                subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                if subdirs:
                    self.classes = sorted([d.name for d in subdirs])
                    self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
                    return dir_path
        
        return None
    
    def load_image(self, image_path):
        """
        Load and preprocess a single image.
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, self.img_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def load_dataset(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Load the complete dataset.
        
        Args:
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of training data to use for validation
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Discover classes
        data_path = self.discover_classes()
        if data_path is None:
            raise ValueError(f"Could not find data in {self.data_dir}")
        
        print(f"Found {len(self.classes)} classes: {self.classes}")
        
        # Load all images and labels
        images = []
        labels = []
        
        for class_name in self.classes:
            class_dir = data_path / class_name
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.jpeg')) + \
                         list(class_dir.glob('*.png'))
            
            print(f"Loading {len(image_files)} images from class '{class_name}'...")
            
            for img_path in image_files:
                img = self.load_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(class_idx)
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"\nTotal images loaded: {len(X)}")
        print(f"Image shape: {X[0].shape}")
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
        )
        
        print(f"\nDataset split:")
        print(f"  Training:   {len(X_train)} images")
        print(f"  Validation: {len(X_val)} images")
        print(f"  Testing:    {len(X_test)} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_class_distribution(self, labels):
        """
        Get the distribution of classes in the dataset.
        
        Args:
            labels (numpy.ndarray): Array of labels
            
        Returns:
            dict: Dictionary mapping class names to counts
        """
        unique, counts = np.unique(labels, return_counts=True)
        distribution = {}
        for idx, count in zip(unique, counts):
            class_name = self.classes[idx]
            distribution[class_name] = count
        return distribution


def get_data_info(data_dir='data'):
    """
    Print information about the dataset.
    
    Args:
        data_dir (str): Path to the data directory
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Data directory '{data_dir}' not found!")
        print("Please run 'python src/download_data.py' first.")
        return
    
    print("=" * 60)
    print("Brain Tumor MRI Dataset Information")
    print("=" * 60)
    
    # Find the actual data directory
    possible_dirs = [
        data_path,
        data_path / 'Training',
        data_path / 'train',
        data_path / 'data'
    ]
    
    for dir_path in possible_dirs:
        if dir_path.exists() and dir_path.is_dir():
            subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
            if subdirs:
                print(f"\nData location: {dir_path}")
                print(f"\nClasses found: {len(subdirs)}")
                
                for subdir in sorted(subdirs):
                    image_count = len(list(subdir.glob('*.jpg'))) + \
                                 len(list(subdir.glob('*.jpeg'))) + \
                                 len(list(subdir.glob('*.png')))
                    print(f"  - {subdir.name}: {image_count} images")
                
                print("=" * 60)
                return
    
    print("No organized class directories found in the data directory.")
    print("=" * 60)


if __name__ == '__main__':
    # Display dataset information
    get_data_info()
