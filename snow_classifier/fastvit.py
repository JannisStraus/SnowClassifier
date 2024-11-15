import random
from pathlib import Path

import cv2
import numpy as np
import torch
from timm import create_model
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from snow_classifier.utils import IMAGE_DIR, TRAIN_DIR


def infer(image_path: str | Path) -> str:
    # Define data transformation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # OpenCV handles resizing, so we only convert to tensor here
        ]
    )

    # Load the trained model with `weights_only=True`
    model = create_model("fastvit_t8", pretrained=False, num_classes=2)
    model.load_state_dict(
        torch.load(TRAIN_DIR / "best.pth", weights_only=True)
    )  # Explicitly set weights_only=True
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Class labels
    class_names = ["grass", "snow"]

    # Read the image using OpenCV
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")

    # Convert BGR (OpenCV format) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to 256x256
    image = cv2.resize(image, (366, 150))

    # Convert to tensor
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1).squeeze(
            0
        )  # Apply softmax and remove batch dimension

        # Print all classes with their probabilities
        for class_name, probability in zip(
            class_names, probabilities.cpu().numpy(), strict=False
        ):
            print(f"{class_name}: {probability:.4f}")

        # Optionally, print the predicted class
        _, predicted = torch.max(outputs, 1)
        return class_names[int(predicted.item())]


def train(epochs: int, seed: int | None = 42) -> None:
    if seed:
        # Set the seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # Ensure deterministic behavior for convolutional operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Define data transformations
    transform = transforms.Compose(
        [
            transforms.Resize((366, 150)),
            transforms.ToTensor(),
        ]
    )

    # Load datasets
    train_dataset = datasets.ImageFolder(root=IMAGE_DIR / "train", transform=transform)
    val_dataset = datasets.ImageFolder(root=IMAGE_DIR / "val", transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize FastViT model
    model = create_model("fastvit_t8", pretrained=False, num_classes=2)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Variable to keep track of the best model performance
    best_accuracy = 0.0  # or negative infinity if tracking loss

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inp, lbl in train_loader:
            inputs, labels = inp.to(device), lbl.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inp, lbl in val_loader:
                inputs, labels = inp.to(device), lbl.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Check if current model is better than the best saved model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                model.state_dict(),
                TRAIN_DIR / "best.pth",
                _use_new_zipfile_serialization=False,
            )

    # Save the final model
    torch.save(
        model.state_dict(),
        TRAIN_DIR / "final.pth",
        _use_new_zipfile_serialization=False,
    )
    print(f"Best model saved with accuracy: {best_accuracy:.4f}")
