import cv2
import csv
import mediapipe as mp
import os
import io
from PIL import Image
import numpy as np
import threading
import queue
from multiprocessing import Pool as Processpool


path = 'C:/Users/ninja/Desktop/Capstone/mvp/'
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
# https://www.geeksforgeeks.org/python/multithreading-python-set-1/
# https://medium.com/codex/reading-files-fast-with-multi-threading-in-python-ff079f40fe56
# https://vuamitom.github.io/2019/12/13/fast-iterate-through-video-frames.html
# Tam Vu is a legend
# 

global_queue = queue.Queue()
file_queue = queue.Queue()

def producer_thread(workerId):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mpDraw = mp.solutions.drawing_utils
    
    while file_queue.empty()!=True:
        
        [Video_path,Video_name,Id]=file_queue.get()
        video = os.path.join(Video_path,Video_name) 
        cap = cv2.VideoCapture(video)
        
        while True:
            part = Video_name[len(Video_name)-11:len(Video_name)-8]
            if part == "001":  
                part="001"
            else : part = "002"
            spose = [part]
            spose.append(Id)
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
                global_queue.put(spose)
                
        cap.release()
        file_queue.task_done()
        print(f"Worker {workerId} finished {part} Id {Id} fql: {file_queue.qsize()}\n")
    print(f"Worker {workerId} is done\n")

def consumer_thread():
    f= open("PData.CSV", "w",newline='')
    writer = csv.writer(f)
    while True:
        frame=global_queue.get()
        if frame is None:
            break 
        writer.writerow(frame)
    print("CSV is Completed\n")
    f.close()
    
count = 0
print("ADDING all videos to the queue")
for name in os.listdir(path):
    count+=1
    # print(f"{name} added to the queue")
    file_queue.put((path,name,count))

consumer = threading.Thread(target=consumer_thread)
consumer.start()
threads = []

for i in range(os.cpu_count()):
    thready = threading.Thread(target=producer_thread,args=(i+1,))
    thready.start()
    threads.append(thready)
for t in threads:
    t.join()
global_queue.put(None)
consumer.join()

cv2.destroyAllWindows()
