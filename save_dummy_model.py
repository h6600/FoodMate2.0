import torch
from models.cnn_model import FoodCNN

# Create model instance with 5 food classes
model = FoodCNN(num_classes=5)

# Save untrained (dummy) model weights
torch.save(model.state_dict(), 'models/food_cnn.pth')

print("✅ Dummy model saved as 'models/food_cnn.pth'")
