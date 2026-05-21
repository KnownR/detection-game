# train_3class.py
import json, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_FILE = "pose_data_3class.json"
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3
VAL_SPLIT = 0.2
IMG_SIZE = 224
MODEL_PATH = "pose_3class_resnet.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load
with open(DATA_FILE) as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")
for move in ['fireball', 'neutral', 'shield']:
    c = sum(1 for x in data if x['move'] == move)
    print(f"  {move}: {c}")

# Dataset
class PoseDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.connections = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),(11,13),(13,15),(15,17),(17,19),(19,15),(15,21),(11,12),(12,14),(14,16),(16,18),(18,20),(20,16),(16,22),(11,23),(12,24),(23,24)]
        self.label_map = {'fireball': 0, 'neutral': 1, 'shield': 2}

    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        s = self.data[i]
        img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (255,255,255))
        draw = ImageDraw.Draw(img)
        pts = [(x*IMG_SIZE, y*IMG_SIZE) for x,y in s['landmarks']]
        for a,b in self.connections:
            if a < len(pts) and b < len(pts):
                draw.line([pts[a], pts[b]], fill=(0,0,255), width=3)
        for x,y in pts:
            draw.ellipse([(x-6,y-6),(x+6,y+6)], fill=(255,0,0))
        if self.transform: img = self.transform(img)
        label = self.label_map[s['move']]
        return img, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

dataset = PoseDataset(data, transform)
train_len = int(len(dataset) * (1 - VAL_SPLIT))
train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for p in model.parameters(): p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# Train
best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    correct = total = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        _, pred = torch.max(out, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    train_acc = correct / total

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="val"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print("  → BEST MODEL SAVED")

print(f"Training done! Best Val Acc: {best_acc:.4f}")