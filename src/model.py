"""
CNN models for Brain Tumor MRI Classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2


def create_simple_cnn(input_shape=(224, 224, 3), num_classes=4):
    """
    Create a simple CNN model for brain tumor classification.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of tumor classes
        
    Returns:
        keras.Model: Compiled CNN model
    """
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Conv Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_transfer_learning_model(base_model_name='VGG16', input_shape=(224, 224, 3), 
                                   num_classes=4, trainable_layers=0):
    """
    Create a transfer learning model using pre-trained weights.
    
    Args:
        base_model_name (str): Name of the base model ('VGG16', 'ResNet50', 'MobileNetV2')
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of tumor classes
        trainable_layers (int): Number of layers to unfreeze for fine-tuning (0 = freeze all)
        
    Returns:
        keras.Model: Compiled transfer learning model
    """
    # Select base model
    if base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Optionally unfreeze some layers for fine-tuning
    if trainable_layers > 0:
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
    
    # Create full model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model (keras.Model): Model to compile
        learning_rate (float): Learning rate for optimizer
        
    Returns:
        keras.Model: Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_model_summary(model):
    """
    Print model summary.
    
    Args:
        model (keras.Model): Model to summarize
    """
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    model.summary()
    print("=" * 80)
    
    # Count parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    # Test model creation
    print("Testing Simple CNN model...")
    simple_model = create_simple_cnn(num_classes=4)
    simple_model = compile_model(simple_model)
    get_model_summary(simple_model)
    
    print("\nTesting Transfer Learning model...")
    transfer_model = create_transfer_learning_model(base_model_name='MobileNetV2', num_classes=4)
    transfer_model = compile_model(transfer_model)
    get_model_summary(transfer_model)
