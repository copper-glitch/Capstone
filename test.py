import cv2
import mediapipe as mp
import os
import numpy as np
import sys
import pickle

# https://stackoverflow.com/questions/58887056/resize-frame-of-cv2-videocapture
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
# https://docs.opencv.org/4.x/
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# https://www.geeksforgeeks.org/python/writing-csv-files-in-python/
# https://docs.python.org/3/tutorial/datastructures.html#more-on-lists
# https://www.geeksforgeeks.org/python/python-loop-through-folders-and-files-in-directory/
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
# https://stackoverflow.com/questions/3348460/csv-file-written-with-python-has-blank-lines-between-each-row
# https://numpy.org/doc/stable/reference/generated/numpy.argmax.html


MAX_F = 60
FEATURES = 66

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mpDraw = mp.solutions.drawing_utils

def getdata(model): 
    
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    
    frames = []
    while True:
        spose = []
        success,img = cap.read()
        
        if success != True:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                spose.extend([cx,cy])
        cv2.imshow("Image", img)
        frames.append(spose)
        if len(frames)==MAX_F:
            data = np.expand_dims(frames, axis=0).astype(np.float32)
            output = model(data)
            guess = np.argmax(output)
            print(guess)
            frames=[]
            
            
            
        
def main():
    if len(sys.argv)!=1:
        with open(sys.argv[1], 'rb') as file:  
            model = pickle.load(file)
        getdata(model)
        cv2.destroyAllWindows()
    else:
        print("Need Command Line arguement: Model pkl file")
main()
