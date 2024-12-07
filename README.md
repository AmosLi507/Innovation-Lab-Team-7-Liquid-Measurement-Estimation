# Liquid Volume Classification using EfficientNet

## Overview
This project employs an EfficientNet-B0 model to classify liquid volumes in various containers. The model is trained on a labeled dataset of images, with each category representing a specific liquid volume (e.g., 0ml, 50ml, 100ml, 500ml, 1000ml). After training, the model is capable of accurately predicting the volume category of a new image.

## Key Features
- **Model Architecture:** EfficientNet-B0, fine-tuned for volume classification.
- **Categories:** Multiple discrete volume classes.
- **Accuracy:** Initial evaluations show that the model provides reliable predictions for well-prepared test images.
- **Outputs:** Classification results for test images are saved to a summary file, including detailed statistics per category.

## Requirements
- Python 3.7+
- PyTorch (or TensorFlow/Keras, depending on the chosen EfficientNet implementation)
- CUDA-compatible GPU (recommended but not mandatory)
- Additional packages as listed in `requirements.txt`
