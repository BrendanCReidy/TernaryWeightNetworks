# Ternary Weight Networks

A PyTorch implementation of [Ternary Weight Networks](https://arxiv.org/abs/1605.04711)

## Training
This repository contains two example scripts to get started: main_mnist.py and main_cifar10.py.

### MNIST
For MNIST we use a ternary version of the LeNet5 architecture and train for 10 epochs
```
python3 main_mnist.py
```

### CIFAR10
For CIFAR10 we use a ternary version of the MobileNetV2 architecture and train for 200 epochs
```
python3 main_cifar10.py
```