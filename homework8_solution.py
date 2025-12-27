"""
Machine Learning Zoomcamp - Homework 8
Deep Learning with PyTorch - Hair Classification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import subprocess

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# Step 1: Download and prepare the data
# ============================================================
print("\n" + "="*60)
print("Step 1: Downloading and preparing data")
print("="*60)

if not os.path.exists('data.zip'):
    print("Downloading dataset...")
    subprocess.run([
        'wget',
        'https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip'
    ])
    print("Unzipping dataset...")
    subprocess.run(['unzip', 'data.zip'])
else:
    print("Dataset already downloaded")

# ============================================================
# Step 2: Build the CNN Model
# ============================================================
print("\n" + "="*60)
print("Step 2: Building CNN Model")
print("="*60)

class HairClassifierCNN(nn.Module):
    def __init__(self):
        super(HairClassifierCNN, self).__init__()

        # Input shape: (3, 200, 200)
        # Convolutional layer: 32 filters, kernel size (3,3), padding=0, stride=1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=3, padding=0, stride=1)
        self.relu1 = nn.ReLU()

        # Max pooling: pool size (2, 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After conv: (32, 198, 198)
        # After pool: (32, 99, 99)
        # Flattened size: 32 * 99 * 99 = 313632

        # Fully connected layer with 64 neurons
        self.fc1 = nn.Linear(32 * 99 * 99, 64)
        self.relu2 = nn.ReLU()

        # Output layer with 1 neuron (binary classification)
        self.fc2 = nn.Linear(64, 1)
        # Note: We'll use BCEWithLogitsLoss which includes sigmoid,
        # so we don't add sigmoid here

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu2(x)

        x = self.fc2(x)

        return x

# Create model
model = HairClassifierCNN().to(device)

# ============================================================
# QUESTION 1: Which loss function to use?
# ============================================================
print("\n" + "="*60)
print("QUESTION 1: Which loss function to use?")
print("="*60)
print("Answer: nn.BCEWithLogitsLoss()")
print("\nExplanation:")
print("- This is a binary classification problem (straight vs curly hair)")
print("- BCEWithLogitsLoss combines sigmoid activation and BCE loss")
print("- It's numerically more stable than applying sigmoid + BCELoss separately")
print("- nn.CrossEntropyLoss() could also work if we had 2 output neurons")

criterion = nn.BCEWithLogitsLoss()

# ============================================================
# QUESTION 2: Total number of parameters
# ============================================================
print("\n" + "="*60)
print("QUESTION 2: Total number of parameters")
print("="*60)

# Manual calculation:
# Conv layer: (3 * 3 * 3 * 32) + 32 = 864 + 32 = 896
# FC1 layer: (313632 * 64) + 64 = 20072448 + 64 = 20072512
# FC2 layer: (64 * 1) + 1 = 64 + 1 = 65
# Total: 896 + 20072512 + 65 = 20073473

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Detailed breakdown
print("\nDetailed parameter breakdown:")
for name, param in model.named_parameters():
    print(f"{name}: {param.numel():,} parameters (shape: {list(param.shape)})")

print(f"\nAnswer: 20073473")

# ============================================================
# Step 3: Set up data loaders
# ============================================================
print("\n" + "="*60)
print("Step 3: Setting up data loaders")
print("="*60)

# Define transformations (without augmentation for now)
train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )  # ImageNet normalization
])

test_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load datasets
train_dataset = datasets.ImageFolder('data/train', transform=train_transforms)
validation_dataset = datasets.ImageFolder('data/test', transform=test_transforms)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(validation_dataset)}")
print(f"Classes: {train_dataset.classes}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=20, shuffle=False)

# ============================================================
# Step 4: Set up optimizer
# ============================================================
print("\n" + "="*60)
print("Step 4: Setting up optimizer")
print("="*60)

optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.8)
print("Optimizer: SGD with lr=0.002, momentum=0.8")

# ============================================================
# Step 5: Train the model (10 epochs without augmentation)
# ============================================================
print("\n" + "="*60)
print("Step 5: Training model (10 epochs without augmentation)")
print("="*60)

num_epochs = 10
history = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)  # Ensure labels are float and have shape (batch_size, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        # For binary classification with BCEWithLogitsLoss, apply sigmoid to outputs before thresholding for accuracy
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_train / total_train
    history['loss'].append(epoch_loss)
    history['acc'].append(epoch_acc)

    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / len(validation_dataset)
    val_epoch_acc = correct_val / total_val
    history['val_loss'].append(val_epoch_loss)
    history['val_acc'].append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

# ============================================================
# QUESTION 3: Median of training accuracy
# ============================================================
print("\n" + "="*60)
print("QUESTION 3: Median of training accuracy")
print("="*60)

median_train_acc = np.median(history['acc'])
print(f"Training accuracies: {[f'{acc:.4f}' for acc in history['acc']]}")
print(f"Median training accuracy: {median_train_acc:.4f}")
print(f"\nAnswer: {median_train_acc:.2f}")

# ============================================================
# QUESTION 4: Standard deviation of training loss
# ============================================================
print("\n" + "="*60)
print("QUESTION 4: Standard deviation of training loss")
print("="*60)

std_train_loss = np.std(history['loss'])
print(f"Training losses: {[f'{loss:.4f}' for loss in history['loss']]}")
print(f"Standard deviation of training loss: {std_train_loss:.4f}")
print(f"\nAnswer: {std_train_loss:.3f}")

# ============================================================
# Step 6: Add data augmentation and train for 10 more epochs
# ============================================================
print("\n" + "="*60)
print("Step 6: Adding data augmentation and training for 10 more epochs")
print("="*60)

# Define augmented transformations
train_transforms_augmented = transforms.Compose([
    transforms.RandomRotation(50),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Reload training dataset with augmentation
train_dataset_aug = datasets.ImageFolder('data/train', transform=train_transforms_augmented)
train_loader_aug = DataLoader(train_dataset_aug, batch_size=20, shuffle=True)

print("Training with data augmentation...")

num_epochs_aug = 10
history_aug = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

for epoch in range(num_epochs_aug):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader_aug:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset_aug)
    epoch_acc = correct_train / total_train
    history_aug['loss'].append(epoch_loss)
    history_aug['acc'].append(epoch_acc)

    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / len(validation_dataset)
    val_epoch_acc = correct_val / total_val
    history_aug['val_loss'].append(val_epoch_loss)
    history_aug['val_acc'].append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs_aug}, "
          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

# ============================================================
# QUESTION 5: Mean of test loss with augmentations
# ============================================================
print("\n" + "="*60)
print("QUESTION 5: Mean of test loss for all epochs with augmentation")
print("="*60)

mean_test_loss = np.mean(history_aug['val_loss'])
print(f"Test losses: {[f'{loss:.4f}' for loss in history_aug['val_loss']]}")
print(f"Mean test loss: {mean_test_loss:.4f}")
print(f"\nAnswer: {mean_test_loss:.2f}")

# ============================================================
# QUESTION 6: Average test accuracy for last 5 epochs
# ============================================================
print("\n" + "="*60)
print("QUESTION 6: Average test accuracy for last 5 epochs (6-10)")
print("="*60)

avg_test_acc_last5 = np.mean(history_aug['val_acc'][5:10])  # Epochs 6-10 are indices 5-9
print(f"Test accuracies for epochs 6-10: {[f'{acc:.4f}' for acc in history_aug['val_acc'][5:10]]}")
print(f"Average test accuracy (epochs 6-10): {avg_test_acc_last5:.4f}")
print(f"\nAnswer: {avg_test_acc_last5:.2f}")

# ============================================================
# Summary of all answers
# ============================================================
print("\n" + "="*60)
print("SUMMARY OF ALL ANSWERS")
print("="*60)
print(f"Question 1: nn.BCEWithLogitsLoss() (or nn.CrossEntropyLoss())")
print(f"Question 2: {total_params:,} parameters")
print(f"Question 3: Median training accuracy = {median_train_acc:.2f}")
print(f"Question 4: Std of training loss = {std_train_loss:.3f}")
print(f"Question 5: Mean test loss (augmented) = {mean_test_loss:.2f}")
print(f"Question 6: Avg test accuracy (epochs 6-10) = {avg_test_acc_last5:.2f}")
print("="*60)
