import numpy as np
from ultralytics import YOLO
from flask import Flask, request, jsonify
import requests
from io import BytesIO
from PIL import Image
import sys

app = Flask(__name__)

# Load the YOLO model
mymodel = YOLO("grass_types_model.pt")

grass_classes = ['Bahia', 'Bentgrass', 'Bermuda', 'Buffalo Grass', 'Centipede', 'Fine Fescue', 'Kentucky Bluegrass', 'Ryegrass', 'St. Augustine', 'Tall Fescue', 'Zoysia']

@app.route('/')
def index():
    return 'Plant Grass Type Detection Homepage'

def process_image(image):
    result = mymodel.predict(image)
    cc_data = np.array(result[0].boxes.data)

    grass_type = None
    confidence = None

    if len(cc_data) != 0:
        _, _, _, _, conf, clas = cc_data[0]
        grass_type = grass_classes[int(clas)]
        confidence = np.round(conf * 100, 1)

    return grass_type, confidence

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        # Process the uploaded image file
        file = request.files['image']
        image = np.array(Image.open(file))
    elif 'url' in request.json:
        # Process the image from the URL
        url = request.json['url']
        response = requests.get(url)
        image = np.array(Image.open(BytesIO(response.content)))
    else:
        return jsonify({'error': 'No image or URL provided'}), 400

    grass_type, confidence = process_image(image)

    if grass_type is None:
        return jsonify({'error': 'No grass type detected'}), 404

    return jsonify({'grass_type': grass_type}),200

if __name__ == '__main__':
    print("Running app...", file=sys.stderr)
    app.run()
