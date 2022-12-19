import encodings
import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime

Train_path = 'Train'
Test_path = 'Test'
encodings = []
names = []
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
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
    #Pre_Proccessing
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    
    face_detection = face_recognition.face_encodings(image, model="small")
    for faceing in face_detection:
         faceing = np.array(faceing).ravel()
         encodings.append(faceing)
         names.append(os.path.basename(img).split('.')[0])
def detectFace(path , encodings , names):
    name = []
    unknown_picture = face_recognition.load_image_file(path)
    unknown_picture = cv2.cvtColor(unknown_picture, cv2.COLOR_BGR2RGB)
    unknown_picture = cv2.filter2D(src=unknown_picture, ddepth=-1, kernel=kernel)


    unknown_face_encoding = face_recognition.face_encodings(unknown_picture, model="small")
    unknown_faces = []

    for faces in unknown_face_encoding:
        unknown_faces.append(np.array(faces).ravel())
    total = 0
    for unknown_face in unknown_faces: 
        results = face_recognition.compare_faces(encodings, unknown_face, tolerance=0.5)
        best_face_destination = face_recognition.face_distance(encodings, unknown_face)
        index_of_best_face_destination = np.argmin(best_face_destination) # index of best distance (encodings)
        if results[index_of_best_face_destination]:
            name.append(names[index_of_best_face_destination])
            markAttendance(names[index_of_best_face_destination])
        else:
            name.append("UNKOWN PERSON!")
        total+=1
        if total == 4:
            break
    return name   

def getAccuracy(img , i):
    wrong_Samples = 0
    total = 0
    test_name = os.path.basename(img).split('.')[0]
    
    namesA = detectFace(img,encodings,names)
    print("result of ", i,"th test" ,namesA)

    for name in namesA:
        if name != test_name:
            wrong_Samples+=1
        total+=1
        if total == 4:
            break
    return total,wrong_Samples    

    print("a")
def camModel(encodings , names):
    cap  = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
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
                y1,x2,y2,x1 = faceloc
                # since we scaled down by 4 times
                y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                markAttendance(name)
        cv2.imshow('webcam', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    j = 0 
    print(datetime.now().strftime('%I:%M:%S:%p')) 
    for img in os.listdir(Train_path):
        j+=1
        print("Encoding.... Person num: ",j)
        findEcoding(f'{Train_path}/{img}') 

    print("The testing has been started... ")
    print("GET ACCURACY...")
    print(datetime.now().strftime('%I:%M:%S:%p')) 
    total = 0
    wrong_samples = 0
    i = 1
    for img in os.listdir(Test_path):
        t,w = getAccuracy(f'{Test_path}/{img}',i) 
        total+=t
        wrong_samples+=w
        i+=1
    accuracy = (1-(wrong_samples/total))*100
    print("Accuracy = ", accuracy)
    #camModel(encodings , names)
