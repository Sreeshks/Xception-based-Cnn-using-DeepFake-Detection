# Deepfake Detection using Xception-based CNN

## Project Overview
This project implements a deepfake detection system using an Xception-based Convolutional Neural Network (CNN). The model is designed to classify videos as either real or fake by analyzing individual frames and identifying visual artifacts and inconsistencies typical of deepfake manipulations.

## Model Architecture
The detection system is built on the Xception architecture, which is well-suited for image classification tasks due to its efficient use of depthwise separable convolutions. The model consists of:

- Pre-trained Xception base (ImageNet weights)
- Global Average Pooling layer
- Dense layer (256 neurons with ReLU activation)
- Dropout layer (0.5 rate) for regularization
- Output layer with sigmoid activation for binary classification

The implementation uses a two-phase training approach:
1. Initial training with a frozen base model
2. Fine-tuning where deeper layers of the base model are unfrozen for better feature extraction

## Dataset
The model is trained on the "1000_videos" dataset, which contains:
- Training set: Located in `train/` directory
- Validation set: Located in `val/` directory
- Test set: Located in `test/` directory

Each directory contains categorized videos/frames under real and fake classes.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- scikit-learn
- numpy
- OpenCV (for video processing)

## Installation
```bash
pip install tensorflow scikit-learn numpy opencv-python
```

## Usage
### Training the Model
```python
python train_model.py --data_dir path/to/dataset --epochs 20
```

### Detecting Deepfakes in Videos
```python
python detect_video.py --input path/to/video --output results.txt
```

### Model Evaluation
The model achieves competitive performance on the test set with metrics including:
- Binary accuracy
- Precision and recall for both real and fake classes
- F1-score

## Model Weights
Pre-trained model weights are available in two formats:
- Keras format (.keras): `xception_model_best.keras`
- HDF5 format (.h5): `xception_model_best.h5`

## Implementation Details
The implementation includes:
- Data augmentation (horizontal flips, rotation, zoom) to improve generalization
- Early stopping to prevent overfitting
- Model checkpointing to save the best model during training
- Fine-tuning strategy that freezes early layers while training deeper layers
- Normalization of input images to the range [-1, 1]
