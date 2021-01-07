import numpy as np
import cv2
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("DATA/haarcascades/haarcascade_frontalface_default.xml")
def detect_face(img):
    de_face= np.copy(img)
    face_dim=face_cascade.detectMultiScale(de_face,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in face_dim:
        cv2.rectangle(de_face,(x,y),(x+w,y+h),(0,0,255),10)
    return(de_face)


while True:
    ret,frame=cap.read()
    face=detect_face(frame)
    cv2.imshow('frame',face)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()