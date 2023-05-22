import face_recognition
import os
import numpy as np

def getFace(img):
    image = face_recognition.load_image_file(img)
  
    return image

def findEcoding(img):
    encodings = []
    names = []

    image = getFace(img)

    face_locations = face_recognition.face_locations(image, model="hog")
    face_detection = face_recognition.face_encodings(image,known_face_locations=face_locations)
   

    for faceing in face_detection:
        faceing = np.array(faceing).ravel()
        encodings.append(faceing)
        names.append(os.path.basename(img).split('.')[0])

    return encodings,names


def load_data(path):
    Xdata = []
    yLabel = []
    j = 0 

    for img in os.listdir(path):
        j+=1
        print("Training/Testing.... Person : ",j)
       
        data , name = findEcoding(f'{path}/{img}') 
        
        for da,na in zip(data,name) :
            cnt+=1
            Xdata.append(da) 
            yLabel.append(na)

    return Xdata,yLabel
