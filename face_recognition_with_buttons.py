import cv2
import numpy as np
import face_recognition
import os
import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()


path = 'face_attendance/images'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodelistknown = findEncodings(images)

cap = cv2.VideoCapture(0)

mode = None  # Variable to track current mode
engine.say("Press 1 to register new face, Press 2 to recognize")
engine.runAndWait()

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    if mode == 'register':
        # Registration mode
        cv2.putText(img, "Registration Mode (Press '2' to switch)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Perform face registration

    elif mode == 'recognize':
        # Recognition mode
        cv2.putText(img, "Recognition Mode (Press '1' to switch)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodelistknown, encodeFace)
            faceDis = face_recognition.face_distance(encodelistknown, encodeFace)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex]
                engine.say("the person I see is " + name)
                engine.runAndWait()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)

    key = cv2.waitKey(1)
    if key == ord('q'):  # If 'q' key is pressed
        break
    elif key == ord('1'):  # If '1' key is pressed
        mode = 'register' if mode != 'register' else None  # Toggle mode
        print("Switched to registration mode")
        # Capture an image
        ret, frame = cap.read()
        # Convert captured image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the captured image
        cv2.imshow('Captured Image', gray)
        
        # Use speech recognition to get the file name from audio input
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Speak the file name:")
            engine.say("Speak the person's name:")
            engine.runAndWait()
            audio = recognizer.listen(source)
        
        try:
            # Recognize speech using Google Speech Recognition
            file_name = recognizer.recognize_google(audio)
            print("File name:", file_name)
            engine.say("Image saved successfully.")
            engine.runAndWait()
            # Save the captured image with the recognized file name
            cv2.imwrite(os.path.join(path, f"{file_name}.jpg"), frame)
            print("Image saved successfully.")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio.")
            engine.say("Sorry, I could not understand.")
            engine.runAndWait()
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            engine.say("Could not recognize speech. Please try again.")
            engine.runAndWait()
    elif key == ord('2'):  # If '2' key is pressed
        mode = 'recognize' if mode != 'recognize' else None  # Toggle mode
        print("Switched to recognition mode")

cap.release()
cv2.destroyAllWindows()
