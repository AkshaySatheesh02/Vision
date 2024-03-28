from flask import Flask, request, jsonify, render_template

import os
from werkzeug.utils import secure_filename
import google.generativeai as genai
import cv2
from PIL import Image
import re
import random
from pytesseract import pytesseract
import pyttsx3
import speech_recognition as sr
from openai import OpenAI
import datetime
import inflect
import re





app = Flask(__name__)

output_text = ""
inital_prompt=' you are a robot with eyes and the image above is what you see infront of you, describe the image you see in one or two sentences '
prompt='fail'
genai.configure(api_key='AIzaSyDY81BFUYYffonYmQ_uvNSeJOmrHAsD8N4')  # Replace with your actual API key

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

UPLOAD_FOLDER = './captured_images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILESIZE = 500000  # Maximum file size allowed (in bytes)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def record_audio():
    global prompt
    texts = ["Hello", "hi!", "yes, im listening", "whats up?"]
    random.shuffle(texts)  # Shuffle the list randomly
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        read_aloud(texts[0])
        audio = recognizer.listen(source)
        read_aloud("Recording completed.")
    
    return audio

def convert_audio_to_text(audio):
    global prompt
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    try:
        # Recognize the audio and convert it to text
        text = recognizer.recognize_google(audio)
        print("Prompt:", text)
        return text
    except sr.UnknownValueError:
        read_aloud("could not understand, please repeat")
        return("fail")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

def read_aloud(text):
    global output_text
    output_text = text
    filtered_text = re.sub(r'[^\w\s]', '', text)
    filtered_text = filtered_text.replace('\n', ' ')
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 150)
    # Wait for the existing speech to finish (if any)
    engine.runAndWait()

    # Speak the new text
    engine.say(filtered_text)
    engine.runAndWait()

def tesseract(image):
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract
    text = pytesseract.image_to_string(Image.fromarray(image))
    return text
def filter_special_characters(text):
    # Define a regex pattern to match any non-alphanumeric characters
    pattern = r'[^a-zA-Z0-9\s]'
    
    # Use re.sub() to replace all non-alphanumeric characters with an empty string
    filtered_text = re.sub(pattern, '', text)
    
    return filtered_text
# Define the route to run the Python script

@app.route('/get_output_text', methods=['GET'])
def get_output_text():
    global output_text
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%H:%M:%S")
    

    return jsonify({'output_text': output_text})




@app.route('/conversation', methods=['POST'])
def conversation():
    try:
        global  chat
        prompt='fail'
        inital_conversation_prompt='you are a conversational AI for blind named VISION developed by Abhishek, Akshay, Ananya and Akhil. Limit your answers to one or two sentences. answer in paragraph only when asked to. only if the person says a greeting word for the first time you are supposed to introduce yourself as vision or just say the answer directly. if a question is asked answer to the question precisely instead of giving all details.'
        YOUR_API_KEY = "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"
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")
        p = inflect.engine()
        if current_time.hour == 0:
            hour_text = "twelve"
        elif current_time.hour <= 12:
            hour_text = p.number_to_words(current_time.hour)
        else:
            hour_text = p.number_to_words(current_time.hour - 12)
            
        if current_time.minute == 0:
            minute_text = "o'clock"
        else:
            minute_text = p.number_to_words(current_time.minute)

        # Determine whether it's AM or PM
        if current_time.hour < 12:
            period_text = "a.m."
        else:
            period_text = "p.m."

        # Construct the time in text format
        time_text = f"It's {hour_text}  {minute_text}  {period_text}."
        if request.method == 'POST':
            if 'value' in request.form:
                received_value = request.form['value']
                print("Received value:", received_value)
                if(received_value=='1'):
                    while(prompt=='fail'):
                        audio = record_audio()
                        prompt = convert_audio_to_text(audio)
                        messages = [
                                {
                                    "role": "system",
                                    "content": (
                                        "you are a conversational AI for blind named VISION developed by Abhishek, Akshay, Ananya and Akhil. "
                                        " Limit your answers to one or two sentences. answer in paragraph only when asked to."
                                        "only if the person says a greeting word for the first time you are supposed to introduce yourself as vision or just say the answer directly."
                                        " if a question is asked answer to the question precisely instead of giving all details."    
                                        "You are located in bangalore ,India . When asked date or time or any location specific questions answer based on this location, and in one or two words"
                                        "The time is  "+(time_text)+"\n"
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        prompt
                                    ),
                                },
                            ]
                    try:    
                        response_stream = client.chat.completions.create(
                                               model="sonar-small-online",
                                                messages=messages,
                                            )
                        
                    except Exception as e:
                        print("An error occurred:", e)
                    text = response_stream.choices[0].message.content

                    '''for chunk in response_stream:
                        
                        print(chunk)'''
                    read_aloud(text)
                    print("response= ",text)
                prompt='fail'
                
                return "Received value: " + received_value
            else:
                return "No value received"
    except Exception as e:
        print("error occured",e)
        read_aloud("sorry, there was an error")
        return "none"
    
@app.route('/post_images', methods=['POST'])
def upload_file():
    if 'imageFile' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['imageFile']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check file size
        if os.stat(filepath).st_size > MAX_FILESIZE:
            return jsonify({'error': 'File size exceeds maximum allowed size'})
        
        file.save(filepath)
        other_data = file.filename
        print(other_data)
        if(other_data=='image(1).jpg'):
            global  inital_prompt
            model = genai.GenerativeModel('gemini-pro-vision')
            image = Image.open('captured_images/image1.jpg')
                
            response = model.generate_content([inital_prompt, image])
            text = ''

            for chunk in response:
                text = text + chunk.text
            read_aloud(text)

            out=text
            return out
        elif(other_data=='image(2).jpg'):
            try:
        # Receive image from the HTML page
                image = cv2.imread("captured_images/image2.jpg")
                
                if image is None:
                    return "Error: Unable to read the image."

                # Convert the image to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Apply Gaussian blur to reduce noise
                blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

                # Save the processed image with contours
                save_path = "processed_image_with_contours.jpg"
                cv2.imwrite(save_path, blurred_image)

                # Perform OCR on the processed image
                text = tesseract(blurred_image) 
                text=filter_special_characters(text)
                # You can choose to use the blurred image or original grayscale image for OCR
                read_aloud(text)

                return text
            except Exception as e:
                return f"Error: {str(e)}"
        return jsonify({'message': 'File successfully uploaded', 'filename': filename})
    else:
        return jsonify({'error': 'File type not allowed'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
