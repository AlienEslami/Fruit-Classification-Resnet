#Data from "https://www.kaggle.com/datasets/moltean/fruits"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from glob import glob
from PIL import Image

# Re-size all images to this size
IMAGE_SIZE = [100, 100]  # Feel free to change depending on dataset

# Training configuration
epochs = 6
#In about 3 epochs, validation accuracy would be more than 97%. You can play with this.
batch_size = 32

# Paths to dataset
#Change them based on your needs
train_path = 'C:\\Users\\umroot\\Desktop\\Udemy Courses\\Advanced Computer Vision\\fruits-360_dataset_original-size\\fruits-360-original-size\\Training'
valid_path = 'C:\\Users\\umroot\\Desktop\\Udemy Courses\\Advanced Computer Vision\\fruits-360_dataset_original-size\\fruits-360-original-size\\Validation'

# Useful for getting the number of classes
train_folders = glob(train_path + '/*')
n_classes = len(train_folders)

# Data preprocessing and augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Datasets and Dataloaders
train_dataset = datasets.ImageFolder(train_path, data_transforms['train'])
valid_dataset = datasets.ImageFolder(valid_path, data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# Load the ResNet model pretrained on ImageNet
resnet = models.resnet50(pretrained=True)

# Freeze the model parameters
for param in resnet.parameters():
    param.requires_grad = False

# Modify the classifier to match the number of classes
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, n_classes)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

# Function to train the model
def train_model(model, criterion, optimizer, num_epochs=epochs):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(valid_loader.dataset)
        val_acc = val_corrects.double() / len(valid_loader.dataset)

        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc.item())

        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    return train_loss_history, val_loss_history, train_acc_history, val_acc_history

# Train the model
train_loss, val_loss, train_acc, val_acc = train_model(resnet, criterion, optimizer)

# Plot loss and accuracy
def plot_metrics(train_metric, val_metric, metric_name):
    plt.plot(train_metric, label=f'train {metric_name}')
    plt.plot(val_metric, label=f'val {metric_name}')
    plt.legend()
    plt.title(f'{metric_name} over epochs')
    plt.show()

plot_metrics(train_loss, val_loss, 'loss')
plot_metrics(train_acc, val_acc, 'accuracy')

# Confusion matrix
def get_confusion_matrix(loader):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return confusion_matrix(all_labels, all_preds)

train_cm = get_confusion_matrix(train_loader)
valid_cm = get_confusion_matrix(valid_loader)
print('Train Confusion Matrix:', train_cm)
print('Validation Confusion Matrix:', valid_cm)

# Plot confusion matrix (Assuming a util.py file exists to plot)
# from util import plot_confusion_matrix
# plot_confusion_matrix(train_cm, train_dataset.classes, title='Train confusion matrix')
# plot_confusion_matrix(valid_cm, valid_dataset.classes, title='Validation confusion matrix')