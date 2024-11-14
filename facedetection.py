import cv2
haar_file = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file) #load the cascade
webcam = cv2.VideoCapture(0)# To capture video from webcam. 
while True:
    (_, im) = webcam.read()# Read the frame
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)# Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)# Detect the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)# Draw the rectangle around each face
    cv2.imshow('FaceDetection', im)# Display
    key = cv2.waitKey(10)# Stop if escape key is pressed
    if key == 27:
        break
webcam.release()# Release the VideoCapture object
cv2.destroyAllWindows()
