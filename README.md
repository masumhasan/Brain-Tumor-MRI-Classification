# Brain Tumor MRI Classification

A deep learning project for classifying brain tumors from MRI images using Convolutional Neural Networks (CNN).

## Dataset

This project uses the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masumhasan/brain-tumor-mri-dataset) from Kaggle, which contains MRI images of brain tumors categorized into different classes.

## Features

- **Automated Data Download**: Script to download and prepare the Kaggle dataset
- **Multiple Model Architectures**: 
  - Custom CNN model built from scratch
  - Transfer learning models (VGG16, ResNet50, MobileNetV2)
- **Training Pipeline**: Complete training pipeline with callbacks and model saving
- **Prediction Script**: Easy-to-use prediction script for classifying new images
- **Jupyter Notebook**: Interactive notebook for data exploration and visualization

## Installation

1. Clone this repository:
```bash
git clone https://github.com/masumhasan/Brain-Tumor-MRI-Classification.git
cd Brain-Tumor-MRI-Classification
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle API credentials:
   - Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Click "Create New API Token" to download `kaggle.json`
   - Place the file in `~/.kaggle/kaggle.json`
   - On Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`

## Usage

### 1. Download Dataset

First, download the dataset from Kaggle:

```bash
python src/download_data.py
```

**Note**: You must accept the dataset terms on the Kaggle website before downloading.

### 2. Explore Data

View dataset information:

```bash
python src/data_utils.py
```

Or use the Jupyter notebook for interactive exploration:

```bash
jupyter notebook notebooks/exploration.ipynb
```

### 3. Train Model

Train a simple CNN model:

```bash
python src/train.py --model-type simple --epochs 50 --batch-size 32
```

Train with transfer learning:

```bash
python src/train.py --model-type transfer --epochs 30 --batch-size 16
```

Available options:
- `--model-type`: Model architecture (`simple` or `transfer`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--img-size`: Image size (default: 224)
- `--data-dir`: Path to data directory (default: data)
- `--no-save`: Don't save the trained model

### 4. Make Predictions

Predict tumor type from a new MRI image:

```bash
python src/predict.py --model models/brain_tumor_model_simple_20250117_120000.h5 --image path/to/mri_image.jpg
```

With custom class names:

```bash
python src/predict.py --model models/model.h5 --image image.jpg --class-names models/class_names.txt
```

## Project Structure

```
Brain-Tumor-MRI-Classification/
│
├── data/                          # Dataset directory (created after download)
├── models/                        # Saved models directory
├── notebooks/                     # Jupyter notebooks
│   └── exploration.ipynb         # Data exploration notebook
├── src/                          # Source code
│   ├── download_data.py          # Dataset download script
│   ├── data_utils.py             # Data loading and preprocessing utilities
│   ├── model.py                  # Model architectures
│   ├── train.py                  # Training script
│   └── predict.py                # Prediction script
│
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Model Architectures

### Simple CNN
- 4 Convolutional blocks with batch normalization and dropout
- Global average pooling
- Dense layers with regularization
- ~5M parameters

### Transfer Learning
- Pre-trained base models (VGG16, ResNet50, MobileNetV2)
- Custom classification head
- Option to fine-tune layers

## Results

After training, the model will:
- Save the trained model to the `models/` directory
- Generate a training history plot (`training_history.png`)
- Display test set accuracy and loss
- Save class names for later use

## Requirements

- Python 3.7+
- TensorFlow 2.8+
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Kaggle API

See `requirements.txt` for complete list.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masumhasan/brain-tumor-mri-dataset)
- TensorFlow and Keras for deep learning framework

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.
