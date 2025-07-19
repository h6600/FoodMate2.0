import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.cnn_model import FoodCNN
from torch.utils.data import DataLoader

# Config
DATA_DIR = 'data/food101_subset'
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Datasets
train_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
val_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# Model
model = FoodCNN(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'models/food_cnn.pth')
print("✅ Trained model saved as models/food_cnn.pth")

# Save class names
with open('models/class_names.txt', 'w') as f:
    for class_name in train_data.classes:
        f.write(class_name + "\n")
