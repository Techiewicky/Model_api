# api/index.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'CNN_model.h5')
model_now = tf.keras.models.load_model(model_path)

def load_and_preprocess_image(photo_file):
    img = image.load_img(photo_file, target_size=(256, 256))
    img_array = image.img_to_array(img)
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

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        img_array = load_and_preprocess_image(file)
        predictions = model_now.predict(img_array)
        predicted_class_label, confidence_percentage = get_pred(predictions)
        return jsonify({
            "predicted_class": predicted_class_label,
            "confidence_percentage": float(confidence_percentage)
        })
    
    return jsonify({"error": "File upload failed"}), 500

@app.route('/api/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({"status": "healthy"}), 200
