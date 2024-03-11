import cv2
import numpy as np
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# # Check if the face cascade file has been loaded
# if haar_cascade.empty():
#     raise IOError('Unable to load the face cascade classifier xml file')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Define the scaling factor
scaling_factor = 0.8

while True:
# Capture the current frame and resize it
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = haar_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    cv2.imshow('Face Detector', frame)
    # Check if Esc key has been pressed
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()