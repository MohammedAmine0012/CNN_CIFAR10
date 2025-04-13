# CNN Models for CIFAR-10 Classification

This repository contains implementations of Convolutional Neural Networks (CNNs) for image classification using the CIFAR-10 dataset. The project includes different approaches using popular deep learning frameworks and transfer learning techniques.

## Project Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The goal is to correctly classify these images into their respective categories.

## Repository Contents

- `CNN_cifar10_tf_.ipynb`: Implementation using TensorFlow
- `CNN_cifar10_with_torch.ipynb`: Implementation using PyTorch
- `cifar10-transfer-learning-resnet-50.ipynb`: Transfer learning approach using ResNet-50 architecture

## Features

- Multiple implementations using different deep learning frameworks
- Transfer learning approach with pre-trained ResNet-50
- Detailed notebooks with explanations and visualizations
- Model training and evaluation code

## Requirements

To run these notebooks, you'll need:

- Python 3.x
- TensorFlow
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook/Lab

## Usage

1. Clone this repository:
```bash
git clone https://github.com/yourusername/CNN_CIFAR10.git
cd CNN_CIFAR10
```

2. Install the required dependencies
3. Open and run the Jupyter notebooks:
```bash
jupyter notebook
```

## Models

### TensorFlow Implementation
- Custom CNN architecture
- Detailed training process
- Model evaluation and performance metrics

### PyTorch Implementation
- PyTorch-based CNN implementation
- Training and validation procedures
- Performance analysis

### Transfer Learning with ResNet-50
- Utilizes pre-trained ResNet-50 architecture
- Fine-tuning for CIFAR-10 classification
- Comparison with custom implementations

## Results

Each notebook contains detailed results including:
- Training and validation accuracy
- Loss curves
- Model performance metrics
- Confusion matrices
- Sample predictions

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to open issues or submit pull requests for any improvements.

## Acknowledgments

- The CIFAR-10 dataset provided by the Canadian Institute For Advanced Research
- TensorFlow and PyTorch communities for their excellent frameworks
- ResNet architecture developers for their contribution to deep learning 
