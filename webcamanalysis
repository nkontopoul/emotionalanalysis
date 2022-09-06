import cv2
import sys
import numpy as np
import logging as log
import datetime as dt
from time import sleep
import imutils
from deepface import DeepFace

cam = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
log.basicConfig(filename='webcam.log',level=log.INFO)
b,g,r,a = 0,255,0,0
video_capture = cv2.VideoCapture(0)
anterior = 0
while True:

    ret, frame = cam.read()
    print(ret)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    emotRes = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection=False)
    emotions=str(emotRes['emotion']).split(",")
    
    cv2.putText(frame, "Emotional Probability", (0, 50), font, .75, (255, 0, 255), 2, cv2.LINE_4)
    cv2.putText(frame, emotions[0], (0, 100), font, .75, (0, 0, 255), 2, cv2.LINE_4)
    cv2.putText(frame, emotions[1], (0, 200), font, .75, (0, 0, 255), 2, cv2.LINE_4)
    cv2.putText(frame, emotions[2], (0, 300), font, .75, (0, 0, 255), 2, cv2.LINE_4)
    cv2.putText(frame, emotions[3], (0, 400), font, .75, (0, 0, 255), 2, cv2.LINE_4)
    cv2.putText(frame, emotions[4], (0, 500), font, .45, (0, 0, 255), 2, cv2.LINE_4)
    cv2.putText(frame, "FPS: {}".format(cam.get(cv2.CAP_PROP_FPS)), (10,30), font, 0.75, (0, 255, 255), 2, cv2.LINE_4) 
    cv2.putText(frame, 'Press q to quit', (200,450), font, 1, (0, 255, 255), 2, cv2.LINE_4)
    cv2.imshow("Emotion Recognition System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
quit()
