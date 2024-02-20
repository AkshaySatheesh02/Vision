from flask import Flask, request, jsonify
import cv2
from super_gradients.training import models
import pyttsx3

app = Flask(__name__)

# Load YOLO NAS model
yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# COCO class names
coco_class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # Receive image from the HTML page
    image_file = request.files['image']
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Perform object detection on the image
    results = yolo_nas_l.predict(image, conf=0.25)

    # Extract information about detected objects
    detected_objects = results[0].prediction

    # Prepare the text for voice feedback
    voice_feedback = "Detected objects:"

    # Count the number of objects for each class
    objects_count = {}
    for label in detected_objects.labels:
        class_name = coco_class_names[int(label)]
        objects_count[class_name] = objects_count.get(class_name, 0) + 1

    # Add object counts to the voice feedback
    for class_name, count in objects_count.items():
        voice_feedback += f" {count} {class_name},"

    # Speak the voice feedback
    engine.say(voice_feedback)
    engine.runAndWait()

    # Prepare JSON response
    response = {
        'text_output': voice_feedback,
        'detected_objects': detected_objects.to_json()
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
