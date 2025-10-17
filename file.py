import cv2
import csv
import mediapipe as mp
import os
import io
from PIL import Image
import numpy as np
# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mpDraw = mp.solutions.drawing_utils

f= open("PData.CSV", "w",newline='')

writer = csv.writer(f)
# https://stackoverflow.com/questions/58887056/resize-frame-of-cv2-videocapture
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
# https://docs.opencv.org/4.x/
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# https://www.geeksforgeeks.org/python/writing-csv-files-in-python/
# https://docs.python.org/3/tutorial/datastructures.html#more-on-lists
# https://www.geeksforgeeks.org/python/python-loop-through-folders-and-files-in-directory/
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
# https://stackoverflow.com/questions/3348460/csv-file-written-with-python-has-blank-lines-between-each-row
# https://www.geeksforgeeks.org/machine-learning/saving-a-machine-learning-model/

def getdata(path,name,count):
    video = os.path.join(path,name) 
    # print(f"{name[0]}{name[len(name)-5]}")
    cap = cv2.VideoCapture(video)
    while True:
        
        part = name[len(name)-11:len(name)-8]
        if part == "050":  
            part="002"
        else : part = "001"
        spose = [part]
        spose.append(count)
        key = cv2.waitKey(1) & 0xFF
        success,img = cap.read()
        if success != True:
            break
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                spose.extend([cx,cy])
            writer.writerow(spose)
        # cv2.imshow("Image", img)
        
        
            
path = 'C:/Users/ninja/Desktop/Capstone/subset/'
count = 0
for name in os.listdir(path):
    count+=1
    print(name)
    getdata(path,name,count)

f.close()
cv2.destroyAllWindows()
