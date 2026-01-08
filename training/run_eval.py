import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from training.dataset import FruitDatabase
from training.model import SimpleFruitClassifier
from training.evaluate import evaluate_and_visualize

# ---- SAFETY FOR MAC OPENMP ----
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- PATHS ----
DATA_DIR = "/Users/bertramsillesen/Desktop/archive/Fruit-262"   # change if local
MODEL_PATH = "model.pt"
BATCH_SIZE = 32
NUM_CLASSES = 2

# ---- DEVICE ----
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)

# ---- TRANSFORMS ----
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ---- DATASET (FULL DATASET OR TEST SPLIT) ----
dataset = FruitDatabase(DATA_DIR, transform=transform)

test_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ---- MODEL ----
model = SimpleFruitClassifier(num_classes=NUM_CLASSES).to(device)

# ---- LOAD SAVED MODEL ----
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded from", MODEL_PATH)

# ---- EVALUATE ----
evaluate_and_visualize(
    model=model,
    dataloader=test_loader,
    dataset=dataset,
    device=device,
    max_images=16
)