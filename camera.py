# -*- coding: utf-8 -*-
"""
Created on Sun May  3 20:50:56 2020
@author: Ilias Sachpazidis
"""

import cv2
import face_recognition

print(cv2.__version__)
cap = cv2.VideoCapture(0)

if not (cap.isOpened()):
    print("Could not open video device")

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,800)

while(True):    
    # Capture frame-by-frame    
    ret, frame = cap.read() 
    frame = cv2.resize(frame, (0,0), None, 0.25,0.25)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
    unknown_faces_location = face_recognition.face_locations(frame)
    # Display the resulting frame    
    # cv2.imshow('preview',frame)    
    for faceloc in unknown_faces_location :
        y1,x2,y2,x1 = faceloc    
        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(frame, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
        cv2.putText(frame,"Mohammed", (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #cv2.imshow('frame', gray)

    cv2.imshow('frame', frame)
    
    #Waits for a user input to quit the application    
    if cv2.waitKey(1) & 0xFF == ord('q'):    
        break
    
cap.release()
cv2.destroyAllWindows()