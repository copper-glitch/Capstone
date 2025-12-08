import cv2
import csv
import os
import mediapipe as mp
import io
from PIL import Image
import numpy as np
import threading
import queue
from multiprocessing import Pool as Processpool


path = 'C:/Users/ninja/Desktop/Capstone/subset/'
# Reference links for documentation and related resources used during development.
# (Your comments preserved exactly)

global_queue = queue.Queue()   # Queue for processed pose data (producer → consumer)
file_queue = queue.Queue()     # Queue of videos to be processed (main → producer threads)

def normalize_landmarks(spose):
    # Normalize pose landmarks so models see consistent coordinates

    left_hip = spose[25]        # Landmark index for left hip
    right_hip = spose[26]       # Landmark index for right hip
    hip_center = (left_hip + right_hip) / 2

    left_shoulder = spose[13]   # Landmark index for left shoulder
    right_shoulder = spose[14]  # Landmark index for right shoulder

    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)

    if shoulder_dist < 1e-6:    # Prevent division by zero
        return spose

    # Translate and scale to normalize pose
    spose -= hip_center
    spose /= shoulder_dist

    return spose


def producer_thread(workerId):
    # Thread function: reads videos, extracts pose landmarks, pushes to global_queue

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mpDraw = mp.solutions.drawing_utils
    
    while file_queue.empty() != True:
        
        [Video_path,Video_name,Id] = file_queue.get()
        video = os.path.join(Video_path,Video_name)
        cap = cv2.VideoCapture(video)
        
        while True:
            # Determine part label from filename
            part = Video_name[1]
            if part == "0":  
                part = "0"
            elif part == "p":
                part = "1"
            elif part == "m":
                part = "2"
            elif part == "f":
                part == "3"

            spose = [part]       # First element is the class/part label
            spose.append(Id)     # Second element is video ID being processed

            key = cv2.waitKey(1) & 0xFF
            
            success, img = cap.read()
            if success != True:
                break

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Run MediaPipe pose detection
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Append pose landmark coordinates
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    spose.extend([lm.x, lm.y])

                # Normalize landmark coordinates
                pose_array = np.array(spose[2:]).reshape(-1, 2)  # skip part + Id
                spose[2:] = normalize_landmarks(pose_array).flatten().tolist()

                # Push processed frame to global queue
                global_queue.put(spose)
                
        cap.release()
        file_queue.task_done()
        print(f"Worker {workerId} finished {part} Id {Id} fql: {file_queue.qsize()}\n")

    print(f"Worker {workerId} is done\n")


def consumer_thread():
    # Writes all processed pose sequences to CSV

    f = open("PData.CSV", "w", newline='')
    writer = csv.writer(f)

    while True:
        frame = global_queue.get()

        if frame is None:   # End signal from main thread
            break 

        writer.writerow(frame)

    print("CSV is Completed\n")
    f.close()   


count = 0
print("ADDING all videos to the queue")

# Fill queue with all video paths to be processed
for name in os.listdir(path):
    count += 1
    file_queue.put((path, name, count))

# Start consumer thread
consumer = threading.Thread(target=consumer_thread)
consumer.start()

threads = []

# Start one producer thread per CPU core
for i in range(os.cpu_count()):
    thready = threading.Thread(target=producer_thread, args=(i+1,))
    thready.start()
    threads.append(thready)

# Wait for all producers to finish
for t in threads:
    t.join()

# Signal consumer to stop
global_queue.put(None)
consumer.join()

cv2.destroyAllWindows()
