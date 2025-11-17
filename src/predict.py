"""
Prediction script for Brain Tumor MRI Classification.
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

import tensorflow as tf
from tensorflow import keras


def load_and_preprocess_image(image_path, img_size=(224, 224)):
    """
    Load and preprocess an image for prediction.
    
    Args:
        image_path (str): Path to the image file
        img_size (tuple): Target image size (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image ready for prediction
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, img_size)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def predict_tumor(model_path, image_path, class_names_path=None):
    """
    Predict brain tumor type from an MRI image.
    
    Args:
        model_path (str): Path to the trained model
        image_path (str): Path to the image to predict
        class_names_path (str): Path to the class names file (optional)
        
    Returns:
        tuple: (predicted_class, confidence, all_probabilities)
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Load class names
    if class_names_path and Path(class_names_path).exists():
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        # Default class names (common for brain tumor datasets)
        class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        print(f"Warning: Using default class names: {class_names}")
    
    # Load and preprocess image
    print(f"Loading image from {image_path}...")
    img = load_and_preprocess_image(image_path)
    
    # Make prediction
    print("Making prediction...")
    predictions = model.predict(img, verbose=0)
    
    # Get predicted class
    predicted_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_idx]
    confidence = predictions[0][predicted_idx]
    
    return predicted_class, confidence, predictions[0], class_names


def main():
    parser = argparse.ArgumentParser(description='Predict Brain Tumor Type from MRI Image')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to the trained model (.h5 file)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to the MRI image to predict')
    parser.add_argument('--class-names', type=str, default=None,
                       help='Path to class names file (optional)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Make prediction
    try:
        predicted_class, confidence, all_probs, class_names = predict_tumor(
            args.model,
            args.image,
            args.class_names
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Image: {args.image}")
        print(f"\nPredicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print("\nAll Probabilities:")
        for class_name, prob in zip(class_names, all_probs):
            print(f"  {class_name:20s}: {prob:.4f} ({prob*100:.2f}%)")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
