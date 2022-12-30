import cv2
import time

# Loading pre-trained OpenCV data for different features
trainedFrontalFace = cv2.CascadeClassifier(
    'data\haarcascades\haarcascade_frontalface_default.xml')
trainedEyes = cv2.CascadeClassifier(
    'data\haarcascades\haarcascade_eye.xml')
trainedProfileFace = cv2.CascadeClassifier(
    'data\haarcascades\haarcascade_profileface.xml')
trainedFullBody = cv2.CascadeClassifier(
    'data\haarcascades\haarcascade_fullbody.xml')

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

    # Detect frontal faces and draw a green rectangle around them
    face_coordinates = trainedFrontalFace.detectMultiScale(greyscaled_frame)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

    # Detect eyes and draw a blue rectangle around them
    eye_coordinates = trainedEyes.detectMultiScale(greyscaled_frame)
    for (x, y, w, h) in eye_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

    # Detect profile faces and draw a yellow rectangle around them
    profile_face_coordinates = trainedProfileFace.detectMultiScale(
        greyscaled_frame)
    for (x, y, w, h) in profile_face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 5)

    # Detect full bodies and draw a red rectangle around them
    full_body_coordinates = trainedFullBody.detectMultiScale(greyscaled_frame)
    for (x, y, w, h) in full_body_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Next frame
    cv2.waitKey(1)

    # Quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
