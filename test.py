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

def getdata(path,name): 
    video = os.path.join(path,name) 
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        spose = []
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
        cv2.imshow("Image", img)
        key = cv2.waitKey(1) & 0xFF
        frames.append(spose)
    return frames
def pad(data):
    if len(data)>MAX_F:
        return data[:MAX_F]
    elif len(data)<MAX_F:
        pad = []
        i=0
        while i<MAX_F-len(data):
            temp = []
            x = 0
            while x<FEATURES:
                temp.append(0)
                x+=1
            pad.append(temp)
            i+=1
        pad=np.array(pad)
        
        # ___________________________________________
        return np.vstack((data,pad))
        # ____________________________________________
    else:
        return data   
def main():
    path = 'C:/Users/ninja/Desktop/Capstone/mvp/'
    
    with open(sys.argv[1], 'rb') as file:  
            model = pickle.load(file)

    for name in os.listdir(path):

        print(name)
        data = getdata(path,name)
        data=np.array(pad(data))
        #_________________________________________________________  
        data = np.expand_dims(data, axis=0).astype(np.float32)
        #_________________________________________________________
        
        output = model(data)
        guess = np.argmax(output)
        print(guess)
    
    cv2.destroyAllWindows()
main()