import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from training.dataset import build_datasets
from training.model import SimpleFruitClassifier
from training.train import train_model
from training.evaluate import evaluate_and_visualize


# ---- CONFIG ----
NUM_EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3

DATA_DIR = os.environ.get("DATA_DIR")
if DATA_DIR is None:
    raise RuntimeError(
        "DATA_DIR not set. "
        "Run with: export DATA_DIR=/work3/<user>/Fruit-262"
    )

# ---- DEVICE ----
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)

# ---- DATA ----
full_dataset, train_ds, val_ds, test_ds = build_datasets(DATA_DIR)
num_classes = len(full_dataset.classes)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Number of classes: {num_classes}")
print(f"Example classes: {full_dataset.classes[:5]}")

# ---- MODEL ----
model = SimpleFruitClassifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---- TRAIN ----
train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=NUM_EPOCHS
)

# ---- EVALUATE ----
evaluate_and_visualize(
    model=model,
    dataloader=test_loader,
    dataset=full_dataset,
    device=device,
    max_images=16
)