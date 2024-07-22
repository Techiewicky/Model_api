import logging
import json
import tensorflow as tf
import numpy as np
import os
import io
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'CNN model.h5')
model_now = tf.keras.models.load_model(model_path, compile=False)

def load_and_preprocess_image(photo_file):
    # Read the file into a BytesIO object
    img_bytes = io.BytesIO(photo_file.read())
    
    # Open the image using PIL
    with Image.open(img_bytes) as img:
        # Convert to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img)
    
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def get_pred(predictions):
    class_labels = ['normal', 'early mid', 'mid', 'late mid', 'late']
    predicted_probabilities = predictions[0]
    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class_label = class_labels[predicted_class_index]
    confidence_percentage = predicted_probabilities[predicted_class_index] * 100
    return predicted_class_label, confidence_percentage

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        # Check if there's any file in the request
        if len(request.files) > 0:
            # Get the first file, regardless of its key
            file = next(iter(request.files.values()))
        else:
            return jsonify({'error': 'No file part', 'received_data': str(request.files)})
    else:
        file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            img_array = load_and_preprocess_image(file)
            predictions = model_now.predict(img_array)
            predicted_class_label, confidence_percentage = get_pred(predictions)
            
            return jsonify({
                'predicted_class': predicted_class_label,
                'confidence': float(confidence_percentage)
            })
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
