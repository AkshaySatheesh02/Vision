from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from werkzeug.utils import secure_filename

import cv2
from PIL import Image
from pytesseract import pytesseract
import pyttsx3
from collections import Counter
import numpy as np
from io import BytesIO
import os

import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes




def save_image(image_file,custom_filename):
    if image_file and custom_filename:
            filename = secure_filename(custom_filename)  # Secure the custom filename
            image_file.save(filename)
            return filename
    

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
    custom_filename = 'image_file.jpg'
    img=save_image(image_file,custom_filename)

    image = Image.open(img)

    genai.configure(api_key='AIzaSyDY81BFUYYffonYmQ_uvNSeJOmrHAsD8N4')
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content(image)
    text=''
    for chunk in response:
        text=text+chunk.text
    # Perform OCR on the image

    # Read aloud the recognized text
    read_aloud("The image seems to be"+text)

    # Prepare JSON response
    response = {
        'recognized_text': text
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

