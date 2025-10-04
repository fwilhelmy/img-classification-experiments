# CIFAR-10 Image Classification Experiments

## Project Overview
This repository collects the experiment code and reports I assembled while exploring neural network architectures on the CIFAR-10 image classification benchmark. The goal is to compare multilayer perceptrons, residual networks, and modern MLP-Mixer models to understand how architectural choices influence accuracy, training dynamics, and practical trade-offs.

## Usage Summary
The `assignment1_release` folder provides reusable PyTorch utilities, training scripts, and Jupyter notebooks for running experiments. Start from `main.py` or `main.ipynb`, configure a model in `model_configs/`, and track metrics under the `results/` directory. The same tooling supports both local development and GPU-backed environments such as Google Colab.

Maintainer: Felix Wilhelmy

## Dataset
All experiments rely on the CIFAR-10 dataset: 60,000 color images spread evenly across 10 categories, with 50,000 examples for training and 10,000 for evaluation. Each image measures 32×32 pixels, making CIFAR-10 a compact yet challenging benchmark for testing computer vision models.

## Credits
I gratefully acknowledge Prof. Aaron Courville for guiding the project. I also thank the lab assistant who provided the starter implementation that forms the base of this code, which I then heavily modified for my experiments.
