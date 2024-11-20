import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from timm.models import create_model
from torch import nn, optim
from torch.types import Device
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from snow_classifier.utils import IMAGE_DIR, TRAIN_DIR


def infer(
    input_path: str | Path, return_img: bool = False
) -> dict[str, dict[str, Any]]:
    input_path = Path(input_path)

    # Load the model and device once
    model, device = load_model()

    result_dict: dict[str, dict[str, float]] = {}
    if input_path.is_dir():
        # Process all .jpg images in the directory
        for image_file in input_path.glob("*.jpg"):
            result_dict[image_file.name] = process_image(image_file, model, device)
    elif input_path.is_file():
        # Process the single image file
        result_dict[input_path.name] = process_image(input_path, model, device)
    else:
        raise FileNotFoundError(
            f"The path {input_path} is neither a file nor a directory."
        )
    return result_dict


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

    # Load datasets
    transform = transforms.Compose(
        [
            transforms.Resize((150, 366)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = datasets.ImageFolder(root=IMAGE_DIR / "train", transform=transform)
    val_dataset = datasets.ImageFolder(root=IMAGE_DIR / "val", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = create_model("fastvit_t8", pretrained=False, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_accuracy = 0.0  # or negative infinity if tracking loss
    for epoch in range(epochs):
        # Training
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
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Validation
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


def load_model() -> Any | Device:
    # Load the trained model
    model = create_model("fastvit_t8", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(TRAIN_DIR / "best.pth", weights_only=True))
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def process_image(
    image_path: str | Path, model: Any, device: torch.device
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    class_names = ["grass", "snow"]

    transform = transforms.Compose(
        [
            transforms.Resize((150, 366)),
            transforms.ToTensor(),
        ]
    )
    tensor_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_image)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

        result_dict: dict[str, Any] = {
            "result": predicted_class,
            "confidence": {
                class_name: float(prob)
                for class_name, prob in zip(
                    class_names, probabilities.cpu().numpy(), strict=False
                )
            },
            "image": image,
        }
    return result_dict
