from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import numpy as np
from quantum_cnn import HybridQuantumCNN
import torchvision.transforms as transforms

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridQuantumCNN().to(device)
model.load_state_dict(torch.load('quantum_cnn_model.pth', map_location=device))
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Burn degree mapping
burn_degrees = {
    0: "First Degree",
    1: "Second Degree",
    2: "Third Degree"
}

# Recommendations for each burn degree
recommendations = {
    0: "1st-degree burns typically heal within 3-6 days. Keep the area clean, apply aloe vera or moisturizer, and protect from sun exposure.",
    1: "2nd-degree burns require medical attention. Keep the area clean, apply antibiotic ointment, and cover with a sterile bandage. Seek medical care if the burn is larger than 3 inches.",
    2: "3rd-degree burns are medical emergencies. Call emergency services immediately. Do not remove clothing stuck to the burn, and do not apply any ointments or home remedies."
}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get model prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = float(probabilities[0][predicted_class] * 100)
    
    # Prepare response
    result = {
        "degree": burn_degrees[predicted_class],
        "confidence": round(confidence, 2),
        "recommendations": recommendations[predicted_class]
    }
    
    return result 