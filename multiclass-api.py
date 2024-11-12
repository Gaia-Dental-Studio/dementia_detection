from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

# Define the model architecture
class TunedCNN(nn.Module):
    def __init__(self):
        super(TunedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.drop1 = nn.Dropout(p=0.2)
        self.out = nn.Linear(128, 4)
    
    def forward(self, x):
        x = F.mish(self.conv1(x))
        x = self.pool1(x)
        x = self.batchnorm1(x)
        x = F.mish(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.drop1(x)
        x = self.out(x)
        return x

# Initialize Flask app and load model
app = Flask(__name__)
model = TunedCNN()
model.load_state_dict(torch.load("model/dementia_model-cpu.pt", map_location="cpu"))
model.eval()

# Define class names and preprocessing
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Preprocess the image
    image = Image.open(io.BytesIO(file.read())).convert('L')
    input_tensor = preprocess(image).unsqueeze(0)

    # Perform prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
        confidence_scores = {class_names[i]: probabilities[i].item() for i in range(len(class_names))}

    # Find the top predicted class
    predicted_idx = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_idx]
    predicted_confidence = probabilities[predicted_idx].item()

    # Return the result
    return jsonify({
        "predicted_class": predicted_class,
        "confidence": predicted_confidence,
        "confidence_scores": confidence_scores
    })

if __name__ == '__main__':
    app.run(debug=True)
