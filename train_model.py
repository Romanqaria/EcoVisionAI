
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Project settings
DATA_DIR = Path(__file__).resolve().parent / "trashnet-dataset-main"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

device = torch.device("cpu")
print("Using device:", device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Load labeled images from category folders
if not DATA_DIR.exists():
    raise FileNotFoundError(
        "Dataset folder not found. Run this script from EcoVisionAI."
    )

dataset = datasets.ImageFolder(
    root=str(DATA_DIR),
    transform=transform
)

if len(dataset.classes) < 2:
    raise ValueError(
        "At least two category folders containing images are required."
    )

print("Categories:", dataset.classes)
print("Total images:", len(dataset))

# Split into training, validation, and testing sets
total = len(dataset)
train_size = int(total * 0.70)
val_size = int(total * 0.15)
test_size = total - train_size - val_size

generator = torch.Generator().manual_seed(SEED)
train_set, val_set, test_set = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=generator
)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False
)

# Convolutional Neural Network
class WasteCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = WasteCNN(num_classes=len(dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

# Training and validation
best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            val_total += labels.size(0)

    train_accuracy = 100 * train_correct / max(train_total, 1)
    val_accuracy = 100 * val_correct / max(val_total, 1)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss / max(train_total, 1):.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Validation Loss: {val_loss / max(val_total, 1):.4f} | "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": dataset.classes,
            "image_size": IMAGE_SIZE
        }, MODEL_DIR / "waste_cnn.pth")

# Evaluate the saved best model on the held-out test set
checkpoint = torch.load(
    MODEL_DIR / "waste_cnn.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        test_correct += (
            outputs.argmax(1) == labels
        ).sum().item()
        test_total += labels.size(0)

test_accuracy = 100 * test_correct / max(test_total, 1)

print("\nTraining complete!")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print(f"Test accuracy: {test_accuracy:.2f}%")
print("Saved model to:", MODEL_DIR / "waste_cnn.pth")
