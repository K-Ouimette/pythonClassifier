import cv2

# Loading pre-trained OpenCV frontal face data.
trainedFrontalFace = cv2.CascadeClassifier(
    'data\haarcascades\haarcascade_frontalface_default.xml')

# Source to detect from.
img = cv2.imread('staticFaceTest.webp')

# Convert the image to greyscale
greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trainedFrontalFace.detectMultiScale(greyscaled_img)

# Loop through all the detected faces and draw a rectangle around each one
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)

# Shows the image.
cv2.imshow('faceDetector.py', img)

# Closes image on keypress
cv2.waitKey()

# Make sure the code works with no errors.
print('If you see this, you did NOT fuck up the code!  Good job! :D')
