
import encodings
import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime

path = 'Train'
encodings = []
names = []

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

def findEcoding(img):

    image = face_recognition.load_image_file(img)
    cv2.imshow('AV CV- Winter Wonder Sharpened', image)
    sharped_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    cv2.imshow('AV CV- Winter Wonder Sharpened', sharped_image)
    
#   face_locations = face_recognition.face_locations(image)
    
    face_detection = face_recognition.face_encodings(image, model="small")
    for faceing in face_detection:
         faceing = np.array(faceing).ravel()
         encodings.append(faceing)
         names.append(os.path.basename(img).split('.')[0])

def detectFace(path , encodings , names):
    name = []
    unknown_picture = face_recognition.load_image_file(path)
    unknown_picture = cv2.cvtColor(unknown_picture, cv2.COLOR_BGR2RGB)
    #unknown_picture = cv2.resize(unknown_picture, (0,0), None, 0.25,0.25)
    #face_locations = face_recognition.face_locations(unknown_picture)            

    unknown_face_encoding = face_recognition.face_encodings(unknown_picture, model="small")
    unknown_faces = []

    for faces in unknown_face_encoding:
        unknown_faces.append(np.array(faces).ravel())

    for unknown_face in unknown_faces: 
        results = face_recognition.compare_faces(encodings, unknown_face, tolerance=0.5)
        best_face_destination = face_recognition.face_distance(encodings, unknown_face)
        index_of_best_face_destination = np.argmin(best_face_destination) # index of best distance (encodings)
        if results[index_of_best_face_destination]:
            name.append(names[index_of_best_face_destination])
        else:
            name.append("UNKOWN PERSON!")
    return name         

if __name__ == "__main__":
    j = 0 

    image = cv2.imread('Mohammed Alaa.jpg', flags=cv2.IMREAD_COLOR)
    cv2.imshow('AV CV- Winter Wonder Sharpened', image)
    sharped_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    cv2.imshow('AV CV- Winter Wonder Sharpened', sharped_image)