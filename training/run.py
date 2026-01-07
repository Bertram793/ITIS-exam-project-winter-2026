import torch
from torch.utils.data import DataLoader, random_split

from training.dataset import FruitDatabase, get_transforms
from training.model import SimpleFruitClassifier
from training.train import train_model
from training.evaluate import evaluate_and_visualize


# ---------------- CONFIG ----------------
DATA_DIR = "/Users/bertramsillesen/Desktop/archive/Fruit-262"
BATCH_SIZE = 32
NUM_CLASSES = 2      # change to 262 later
NUM_EPOCHS = 5
# ---------------------------------------


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)

    transform = get_transforms()
    full_dataset = FruitDatabase(DATA_DIR, transform)

    print(f"Number of classes: {len(full_dataset.classes)}")
    print(f"Example classes: {full_dataset.classes[:5]}")

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleFruitClassifier(num_classes=NUM_CLASSES).to(device)

    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=NUM_EPOCHS
    )

    evaluate_and_visualize(
        model=model,
        dataloader=test_loader,
        dataset=full_dataset,
        device=device,
        max_images=16
    )


if __name__ == "__main__":
    main()