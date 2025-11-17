"""
Training script for Brain Tumor MRI Classification.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from data_utils import BrainTumorDataLoader
from model import create_simple_cnn, create_transfer_learning_model, compile_model, get_model_summary


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Keras training history object
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def train_model(
    model_type='simple',
    data_dir='data',
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    img_size=(224, 224),
    test_size=0.2,
    val_size=0.1,
    save_model=True
):
    """
    Train a brain tumor classification model.
    
    Args:
        model_type (str): Type of model ('simple' or 'transfer')
        data_dir (str): Path to data directory
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        img_size (tuple): Image size (height, width)
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of training data for validation
        save_model (bool): Whether to save the trained model
    """
    print("=" * 80)
    print("BRAIN TUMOR MRI CLASSIFICATION - TRAINING")
    print("=" * 80)
    print(f"Model type: {model_type}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 80 + "\n")
    
    # Load data
    print("Loading dataset...")
    data_loader = BrainTumorDataLoader(data_dir=data_dir, img_size=img_size)
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_dataset(
        test_size=test_size,
        val_size=val_size
    )
    
    num_classes = len(data_loader.classes)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {data_loader.classes}")
    
    # Print class distribution
    print("\nTraining set class distribution:")
    train_dist = data_loader.get_class_distribution(y_train)
    for class_name, count in train_dist.items():
        print(f"  {class_name}: {count}")
    
    # Create model
    print("\nCreating model...")
    input_shape = (*img_size, 3)
    
    if model_type == 'simple':
        model = create_simple_cnn(input_shape=input_shape, num_classes=num_classes)
    elif model_type == 'transfer':
        model = create_transfer_learning_model(
            base_model_name='MobileNetV2',
            input_shape=input_shape,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = compile_model(model, learning_rate=learning_rate)
    get_model_summary(model)
    
    # Create callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("Starting training...")
    print("-" * 80)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("-" * 80)
    print("Training completed!\n")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Plot training history
    plot_training_history(history, save_path='training_history.png')
    
    # Save model
    if save_model:
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = models_dir / f'brain_tumor_model_{model_type}_{timestamp}.h5'
        model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save class names
        class_names_path = models_dir / f'class_names_{model_type}_{timestamp}.txt'
        with open(class_names_path, 'w') as f:
            f.write('\n'.join(data_loader.classes))
        print(f"Class names saved to {class_names_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("=" * 80)
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Brain Tumor MRI Classification Model')
    parser.add_argument('--model-type', type=str, default='simple',
                       choices=['simple', 'transfer'],
                       help='Type of model to train (default: simple)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to data directory (default: data)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Image size (default: 224)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save the trained model')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train model
    train_model(
        model_type=args.model_type,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        img_size=(args.img_size, args.img_size),
        save_model=not args.no_save
    )


if __name__ == '__main__':
    main()
