import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pennylane as qml
from pennylane import numpy as np
from torchviz import make_dot
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

# Quantum circuit definition
def quantum_circuit(inputs, weights, n_qubits=4):
    # Encode classical data into quantum state
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Apply parameterized quantum gates
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
        qml.RZ(weights[i + n_qubits], wires=i)
    
    # Entangle qubits
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    
    # Measure expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Define the hybrid model
class HybridQuantumCNN(nn.Module):
    def __init__(self, n_qubits=4):
        super(HybridQuantumCNN, self).__init__()
        
        # Classical CNN layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Quantum layer setup
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(quantum_circuit, self.dev, interface="torch")
        
        # Initialize quantum weights
        self.quantum_weights = nn.Parameter(torch.rand(2 * n_qubits))
        
        # Final classical layers
        self.fc1 = nn.Linear(64 * 8 * 8 + n_qubits, 128)
        self.fc2 = nn.Linear(128, 3)
        
    def forward(self, x):
        # Classical CNN processing
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Flatten the CNN output
        cnn_features = x.view(-1, 64 * 8 * 8)
        
        # Prepare input for quantum circuit
        quantum_input = torch.mean(x, dim=[2, 3])  # Average pooling to reduce dimensions
        quantum_input = quantum_input.view(-1, self.n_qubits)
        
        # Process through quantum circuit
        quantum_output = torch.zeros((x.size(0), self.n_qubits))
        for i in range(x.size(0)):
            quantum_output[i] = torch.tensor(self.qnode(quantum_input[i], self.quantum_weights))
        
        # Combine classical and quantum features
        combined_features = torch.cat([cnn_features, quantum_output], dim=1)
        
        # Final classification
        x = torch.relu(self.fc1(combined_features))
        x = self.fc2(x)
        return x

# Reuse the same ImageDataset class from cnn_model.py
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = []
        self.labels = []
        
        # Pre-process files to filter out invalid ones
        for f in os.listdir(image_dir):
            if f.endswith('.jpg'):
                txt_file = f.replace('.jpg', '.txt')
                txt_path = os.path.join(image_dir, txt_file)
                
                # Check if corresponding txt file exists and has valid content
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r') as f_txt:
                            content = f_txt.read().strip()
                            if content:  # Check if file is not empty
                                label = int(float(content))
                                if label in [0, 1, 2]:  # Only accept valid labels
                                    self.image_files.append(f)
                                    self.labels.append(label)
                    except (ValueError, FileNotFoundError):
                        continue
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def visualize_model():
    # Create a dummy input
    dummy_input = torch.randn(1, 3, 64, 64)
    
    # Initialize model
    model = HybridQuantumCNN()
    
    # Forward pass
    output = model(dummy_input)
    
    # Create visualization
    dot = make_dot(output, params=dict(model.named_parameters()))
    
    # Save the visualization
    dot.render('model_architecture', format='png', cleanup=True)
    
    # Display the image
    img = mpimg.imread('model_architecture.png')
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('model_architecture_high_res.png', dpi=300, bbox_inches='tight')
    print("Model architecture visualization saved as 'model_architecture_high_res.png'")

def train_model():
    # Visualize model before training
    visualize_model()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to 64x64
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = ImageDataset('data_set', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = HybridQuantumCNN().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Save the model
    torch.save(model.state_dict(), 'quantum_cnn_model.pth')
    print('Model saved to quantum_cnn_model.pth')

if __name__ == '__main__':
    train_model() 