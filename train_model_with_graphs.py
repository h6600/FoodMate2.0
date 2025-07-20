import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cnn_model import FoodCNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np

# === Configuration ===
DATA_DIR = 'data/food101_subset'
NUM_CLASSES = 80
EPOCHS = 5
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Data transforms ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# === Load dataset ===
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === Initialize model ===
model = FoodCNN(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# === Lists to store metrics ===
train_losses, val_losses = [], []
train_accs, val_accs = [], []
epoch_times, lrs = [], []

# === Training loop ===
for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    train_losses.append(train_loss / len(train_loader))
    train_accs.append(correct / total)

    # === Validation ===
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_accs.append(val_correct / val_total)

    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Acc: {train_accs[-1]:.2f}, Val Acc: {val_accs[-1]:.2f}, Time: {epoch_time:.2f}s")

# === Save model and labels ===
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/food_cnn.pth')
with open('models/class_names.txt', 'w') as f:
    for label in train_dataset.classes:
        f.write(f"{label}\n")

# === Make graphs ===
os.makedirs('graphs', exist_ok=True)

# 1. Loss curve
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('graphs/loss_curve.png')

# 2. Accuracy curve
plt.figure()
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('graphs/accuracy_curve.png')

# 3. Confusion Matrix
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix')
plt.savefig('graphs/confusion_matrix.png')

# 4. Class-wise Accuracy
report = classification_report(all_labels, all_preds, target_names=train_dataset.classes, output_dict=True)
classwise_acc = [report[label]['recall'] for label in train_dataset.classes]

plt.figure()
plt.bar(train_dataset.classes, classwise_acc, color='skyblue')
plt.title('Class-wise Accuracy')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('graphs/classwise_accuracy.png')

# 5. Epoch time
plt.figure()
plt.plot(epoch_times, marker='o', color='purple')
plt.title('Training Time per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.savefig('graphs/epoch_time.png')

# 6. Learning rate
plt.figure()
plt.plot(lrs, marker='o', color='orange')
plt.title('Learning Rate Over Time')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.savefig('graphs/learning_rate_curve.png')

print("✅ Training and graph generation complete. Check /graphs and /models folders.")
