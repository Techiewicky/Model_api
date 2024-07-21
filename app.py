from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the model
model_now = tf.keras.models.load_model('CNN_model.h5')

# Define a function to preprocess the image
def load_and_preprocess_image(photo_file):
    img = image.load_img(photo_file, target_size=(256, 256))  # Adjust target_size to match your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    img_array = img_array / 255.0
    return img_array

# Define a function to interpret the predictions
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
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        img_array = load_and_preprocess_image(file)
        predictions = model_now.predict(img_array)
        predicted_class_label, confidence_percentage = get_pred(predictions)
        return jsonify({"predicted_class": predicted_class_label, "confidence_percentage": confidence_percentage})
    
    return jsonify({"error": "File upload failed"})

if __name__ == '__main__':
    app.run(debug=True)
