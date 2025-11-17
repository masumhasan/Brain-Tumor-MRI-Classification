#!/bin/bash

# Example usage script for Brain Tumor MRI Classification

echo "=========================================="
echo "Brain Tumor MRI Classification"
echo "Example Usage Script"
echo "=========================================="
echo ""

echo "Step 1: Download the dataset from Kaggle"
echo "Command: python src/download_data.py"
echo ""

echo "Step 2: View dataset information"
echo "Command: python src/data_utils.py"
echo ""

echo "Step 3: Train a simple CNN model"
echo "Command: python src/train.py --model-type simple --epochs 50 --batch-size 32"
echo ""

echo "Step 4: Train a transfer learning model"
echo "Command: python src/train.py --model-type transfer --epochs 30 --batch-size 16"
echo ""

echo "Step 5: Make predictions on a new image"
echo "Command: python src/predict.py --model models/brain_tumor_model_simple_YYYYMMDD_HHMMSS.h5 --image path/to/image.jpg"
echo ""

echo "Step 6: Explore data with Jupyter notebook"
echo "Command: jupyter notebook notebooks/exploration.ipynb"
echo ""

echo "=========================================="
echo "For more details, see README.md"
echo "=========================================="
