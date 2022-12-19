
import encodings
import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime

path = 'Train'
encodings = []
names = []

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'{name}, Time: {time}, date: {date}')
            f.writelines("\n")

def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)

def findEcoding(img):

    image = face_recognition.load_image_file(img)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (0,0), None, 0.5,0.5)
#    face_locations = face_recognition.face_locations(image)
    
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
    unknown_persons = []

    for faces in unknown_face_encoding:
        unknown_faces.append(np.array(faces).ravel())
    
    for unknown_face in unknown_faces: 
        results = face_recognition.compare_faces(encodings, unknown_face, tolerance=0.44)
        best_face_destination = face_recognition.face_distance(encodings, unknown_face)
        index_of_best_face_destination = np.argmin(best_face_destination) # index of best distance (encodings)
        if results[index_of_best_face_destination]:
            name.append(names[index_of_best_face_destination])
            markAttendance(names[index_of_best_face_destination])
          
        else:
            if len(unknown_persons) == 0: 
                unknown_persons.append(np.array(unknown_face).ravel())    
                name.append("UNKOWN PERSON!")
            else:
                results = face_recognition.compare_faces(unknown_persons, unknown_face, tolerance=0.44)
                best_face_destination = face_recognition.face_distance(unknown_persons, unknown_face)
                index_of_best_face_destination = np.argmin(best_face_destination) # index of best distance (encodings)
                if results[index_of_best_face_destination] == False:
                        unknown_persons.append(np.array(unknown_face).ravel())    
                        name.append("UNKOWN PERSON!")

    return name         
count = 0

sample_number = 1  
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"
measures = np.zeros(sample_number, dtype=np.float64)
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
def antiSpoof(img):
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    faces3 = net.forward()
    measures[count%sample_number]=0
    height, width = img.shape[:2]
    text = "True"
    for i in range(faces3.shape[2]):
        confidence = faces3[0, 0, i, 2]
        if confidence > 0.5:
            box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            # cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 5)
            roi = img[y:y1, x:x1]

            point = (0,0)
            
            img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
            img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)
    
            ycrcb_hist = calc_hist(img_ycrcb)
            luv_hist = calc_hist(img_luv)

            ycrcb_hist = calc_hist(img_ycrcb)
            luv_hist = calc_hist(img_luv)
    
            feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
            feature_vector = feature_vector.reshape(1, len(feature_vector))
    
            prediction = clf.predict_proba(feature_vector)
            prob = prediction[0][1]
    
            measures[count % sample_number] = prob
    
            cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
    
            point = (x, y-5)
    
            print (measures, np.mean(measures))
            if 0 not in measures:
                if np.mean(measures) >= 0.7:
                    text = "False"
                else:
                    text = "True"

    return text
            

def camModel(encodings , names):
    
    cap  = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        imgS = cv2.resize(img, (0,0), None, 0.3,0.3)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        unknown_faces_location = face_recognition.face_locations(imgS)
        unknown_faces_encodeings = face_recognition.face_encodings(imgS, unknown_faces_location)
 
        
        for encode_face, faceloc in zip(unknown_faces_encodeings,unknown_faces_location):
            matches = face_recognition.compare_faces(encodings, encode_face , tolerance=0.5)
            faceDist = face_recognition.face_distance(encodings, encode_face)
            matchIndex = np.argmin(faceDist)
        
            if matches[matchIndex]:
                name = names[matchIndex]
                print(name)
                
                y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                markAttendance(name)

                # since we scaled down by 4 times
               

        cv2.imshow('webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    clf = None
    j = 0 
    known_cnt = 0
    unknown_cnt = 0
    print(datetime.now().strftime('%I:%M:%S:%p')) 
    for img in os.listdir(path):
        j+=1
        print("Encoding.... Person num: ",j)
        findEcoding(f'{path}/{img}') 

    print("The testing has been started... ")
    
    camModel(encodings , names)