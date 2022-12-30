import cv2
import time

# Loading pre-trained OpenCV data.
trainedFrontalFace = cv2.CascadeClassifier(
    'data\haarcascades\haarcascade_frontalface_default.xml')
trainedEye = cv2.CascadeClassifier('data/haarcascade/haarcascade_eye.xml')

# Create a VideoCapture object
defaultCaptureSource = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not defaultCaptureSource.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # Read the next frame from the webcam
    successfulFrameRead, frame = defaultCaptureSource.read()

    # Check if the frame was read successfully
    if not successfulFrameRead:
        break

    # Convert the frame to greyscale
    greyscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trainedFrontalFace.detectMultiScale(greyscaled_frame)

    # Loop through all the detected faces and draw a rectangle around each one
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Next frame
    cv2.waitKey(1)

    # Quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
