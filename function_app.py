# function_app.py
import azure.functions as func
import logging
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import io

app = func.FunctionApp()

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'CNN_model.h5')
model_now = tf.keras.models.load_model(model_path)

def load_and_preprocess_image(photo_file):
    img = image.load_img(io.BytesIO(photo_file.read()), target_size=(256, 256))
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

@app.route(route="predict", auth_level=func.AuthLevel.ANONYMOUS)
def predict(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        file = req.files['file']
        if not file:
            return func.HttpResponse(
                json.dumps({"error": "No file part"}),
                status_code=400,
                mimetype="application/json"
            )

        img_array = load_and_preprocess_image(file)
        predictions = model_now.predict(img_array)
        predicted_class_label, confidence_percentage = get_pred(predictions)

        return func.HttpResponse(
            json.dumps({
                "predicted_class": predicted_class_label,
                "confidence_percentage": float(confidence_percentage)
            }),
            mimetype="application/json"
        )

    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="healthcheck", auth_level=func.AuthLevel.ANONYMOUS)
def healthcheck(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps({"status": "healthy"}),
        status_code=200,
        mimetype="application/json"
    )
