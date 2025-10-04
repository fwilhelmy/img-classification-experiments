from utils import cross_entropy_loss
import os
import json
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

# Function to compute convergence speed: epoch when validation accuracy first reaches a given threshold.
def convergence_speed(accs, threshold=0.65):
    for i, acc in enumerate(accs):
        if acc >= threshold:
            return i + 1  # epochs are 1-indexed
    return None

# Function to compute stability (variance) of a metric over the last n epochs.
def stability(metric, last_n=5):
    if len(metric) < last_n:
        return np.var(metric)
    else:
        return np.var(metric[-last_n:])

# Function to compute gradient norms for each parameter (using one batch from the dataloader).
def gradient_norms(model, dataloader, device):
    model.eval()
    grad_norms = {}
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(imgs)
        loss = cross_entropy_loss(outputs, labels)  # assuming cross_entropy_loss is defined
        loss.backward()
        # Compute L2 norm of gradients for each parameter
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.data.norm(2).item()
        break  # Process only one batch for analysis
    return grad_norms

# Function to evaluate the model on the test set: computes confusion matrix, weighted F1 score, and classification report.
def evaluate_model(model, test_dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds)
    return cm, f1, report
