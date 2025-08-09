# MNIST CNN Digit Classifier with GPU Training and Webcam Inference

This project trains a Convolutional Neural Network (CNN) on the MNIST handwritten digits dataset using PyTorch with GPU acceleration. After training, you can use your webcam to draw or show digits in real time and have the model classify them.

---

## Features

- Train a simple CNN on MNIST dataset with GPU support (CUDA 12.1 recommended).
- Load MNIST from local `.idx` files or download automatically.
- Real-time digit recognition using webcam input.
- Clean and modular PyTorch implementation.
- Easy setup via Conda environment or pip.

---

## Environment

- Python 3.10
- NVIDIA GPU with CUDA 12.1 drivers installed
- Conda (recommended) or pip

---

## Setup

### Using Conda (recommended)

1. Create and activate the environment:

```bash
conda env create mnist-cnn-gpu python=3.10
conda activate mnist-cnn-gpu

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install opencv-python idx2numpy numpy
```
2. Run the training script, this will not run again if network exist

```bash
python3 MNIST_train.py
```
3. Then run the webcam
```bash
python3 MNIST_webcam.py
