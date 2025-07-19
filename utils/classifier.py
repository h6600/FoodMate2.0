import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import base64
from models.cnn_model import FoodCNN

# Load class names
with open('models/class_names.txt', 'r') as f:
    FOOD_CLASSES = [line.strip() for line in f.readlines()]

# Load model
model = FoodCNN(num_classes=len(FOOD_CLASSES))
model.load_state_dict(torch.load('models/food_cnn.pth', map_location=torch.device('cpu')))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def predict_food_tag(image_base64):
    """
    Accepts image_base64 (a base64 string without header),
    decodes it into a PIL image, and returns predicted class.
    """
    try:
        # Decode base64 to image
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            return FOOD_CLASSES[predicted.item()]
    except Exception as e:
        print("Prediction error:", e)
        return "Unknown"
