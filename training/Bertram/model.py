import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# -------- Paths (USE RELATIVE PATHS) --------
TRAIN_PATH = "fruits_dataset/train/train"   # change if needed
TEST_PATH = "fruits_dataset/test/test"


BATCH_SIZE = 492
NUM_CLASSES = 33

# -------- Load and inspect ONE image --------
img_path = os.path.join(
    TRAIN_PATH,
    "Apple Braeburn",
    "Apple Braeburn_0.jpg"
)

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()   # (C, H, W) in [0,1]
])

img = Image.open(img_path).convert("RGB")
img_tensor = transform(img)

print(img_tensor.shape)   # torch.Size([3, 100, 100])