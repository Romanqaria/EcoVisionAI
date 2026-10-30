
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Project settings
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "trashnet-dataset-main"
MODEL_PATH = PROJECT_DIR / "Models" / "waste_cnn.pth"
IMAGE_SIZE = 128
BATCH_SIZE = 16

# Use the same CNN architecture as train_model.py
class WasteCNN(nn.Module):
    def __init__(self, num_classes):
        super(WasteCNN, self).__init__()

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


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run train_model.py first.")
        return

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        )
    ])

    dataset = datasets.ImageFolder(
        DATASET_PATH,
        transform=transform
    )

    # Use the same random split settings as training
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    _, _, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    classes = checkpoint["classes"]

    if dataset.classes != classes:
        print("Error: Dataset class order does not match the saved model.")
        return

    model = WasteCNN(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    correct = 0
    total = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1

                if predictions[i].item() == label:
                    class_correct[label] += 1

    if total == 0:
        print("No test images found.")
        return

    accuracy = 100 * correct / total

    print("\n===== EcoVision AI Evaluation =====")
    print("Total test images:", total)
    print(f"Overall accuracy: {accuracy:.2f}%")

    print("\nAccuracy by waste category:")

    for i, category in enumerate(classes):
        if class_total[i] > 0:
            class_accuracy = (
                100 * class_correct[i] / class_total[i]
            )
            print(
                f"{category}: {class_accuracy:.2f}% "
                f"({class_correct[i]}/{class_total[i]})"
            )
        else:
            print(f"{category}: No test images")


if __name__ == "__main__":
    main()
