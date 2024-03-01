from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

import cv2
from PIL import Image
from pytesseract import pytesseract
import pyttsx3
from collections import Counter
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def tesseract(image):
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract
    text = pytesseract.image_to_string(Image.fromarray(image))
    return text

def read_aloud(text):
    engine = pyttsx3.init()

    # Wait for the existing speech to finish (if any)
    engine.runAndWait()

    # Speak the new text
    engine.say(text)
    engine.runAndWait()

@app.route('/detect_text', methods=['POST'])
def detect_text():
    # Receive image from the HTML page
    image_file = request.files['image']
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Perform OCR on the image
    text = tesseract(image)

    # Read aloud the recognized text
    read_aloud(text)

    # Prepare JSON response
    response = {
        'recognized_text': text
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

