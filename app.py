"""
Flask API that serves a Keras image-classification model
========================================================
POST /predict
-------------
Form-data key:  file  (JPEG / PNG)
Returns:        JSON {class: "...", probabilities: {label: prob, ...}}
"""

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import io
import logging


IMG_SIZE    = (128, 128)
MODEL_PATH  = r"C:\Users\Student\OneDrive\Desktop\Data Science\From Model to Production\TrainedModel"
LABELS_PATH = r"C:\Users\Student\PycharmProjects\pythonProject2\artifacts\class_labels.json"

app = Flask(__name__)
model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    CLASS_LABELS = json.load(f)

logging.basicConfig(level=logging.INFO)
app.logger.info(f"Loaded model with {len(CLASS_LABELS)} classes: {CLASS_LABELS}")


def prepare_image(img: Image.Image):
    """Convert PIL Image → model-ready numpy array (1, h, w, 3) scaled 0-1"""
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Fashion classifier is running!",
                    "predict_endpoint": "/predict"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Check input
    if "file" not in request.files:
        return jsonify({"error": "Use form-data with key 'file'"}), 400

    try:
        file = request.files["file"]
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Cannot read image: {e}"}), 400

    # 2. Pre-process & inference
    x = prepare_image(img)
    probs = model.predict(x)[0]
    idx   = int(np.argmax(probs))
    pred  = CLASS_LABELS[idx]

    # 3. Build JSON response
    response = {
        "class": pred,
        "probabilities": {lbl: float(p) for lbl, p in zip(CLASS_LABELS, probs)}
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
