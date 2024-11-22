from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Load the trained model
model = load_model('model/dementia_model-cpu-v2.h5')

# Define class labels
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if not file:
        return jsonify({"error": "Invalid file"}), 400
    
    try:
         # Convert the FileStorage object to BytesIO
        img_bytes = io.BytesIO(file.read())

        # Load and preprocess the image
        img = image.load_img(img_bytes, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100
        
        # Build response
        response = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_scores": {class_labels[i]: float(predictions[0][i] * 100) for i in range(len(class_labels))}
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
