import cv2
import os
import mediapipe as mp
import numpy as np
import sys
import time
from tensorflow.keras.models import load_model
from threading import Thread

# Multiple reference links for resources used in this project.
MAX_F = 60          # Maximum number of frames to store before prediction
FEATURES = 66       # Number of pose features expected (not used directly below)

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mpDraw = mp.solutions.drawing_utils
def normalize_landmarks(spose):
    # Normalize pose landmarks so models see consistent coordinates
    spose = np.array(spose).reshape(33, 2)
    left_hip = spose[23]        # Landmark index for left hip
    right_hip = spose[24]       # Landmark index for right hip
    hip_center = (left_hip + right_hip) / 2

    left_shoulder = spose[11]   # Landmark index for left shoulder
    right_shoulder = spose[12]  # Landmark index for right shoulder

    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)

    if shoulder_dist < 1e-6:    # Prevent division by zero
        return spose

    # Translate and scale to normalize pose
    spose -= hip_center
    spose /= shoulder_dist

    return spose
class WebcamVideoStream:
    # Custom webcam class to read frames in a separate thread for higher FPS
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        # Setting webcam capture properties
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        # Start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Continuously read frames until stopped
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()
        self.stream.release()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Flag the thread to stop
        self.stopped = True


def getdata(model): 
    # Capture pose data from webcam and perform real-time predictions
    
    vs = WebcamVideoStream(src=0).start()  # Start threaded video stream
    frames = []                            # Stores recent pose frames
    getpuncch = False                      # Unused flag (could be for future logic)
    prev_time = time.time()                # For FPS calculation
    counter = 0                            # Small delay counter before predictions

    while True:
        spose = []                         # Store this frame’s pose vector
        img = vs.read()                    # Get current frame
        if img is None:
            continue

        # Resize and convert frame for MediaPipe
        small = cv2.resize(img, (640, 360))
        smallRGB = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Get pose landmarks
        results = pose.process(smallRGB)

        # FPS calculation
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        # Display frame count and FPS on window
        cv2.putText(img, f'Frame: {len(frames)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Image", img)

        # Exit on 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('g'):
            getpuncch = True
        if key == ord('q'):
            break
        # If pose landmarks detected
        if getpuncch: 
            if results.pose_landmarks:  
                for lm in results.pose_landmarks.landmark:
                    # Extract normalized landmark coordinates
                    h, w, c = img.shape
                    cx, cy = lm.x, lm.y
                    spose.extend([cx, cy])

                frames.append(normalize_landmarks(spose).flatten().tolist())   # Add frame's pose vector

                if len(frames) > MAX_F:
                    frames.pop(0)      # Keep only last MAX_F frames

                counter += 1

        # When enough frames collected and small delay passed, predict
        if len(frames) == MAX_F:
            data = np.expand_dims(frames, axis=0).astype(np.float32)
            output = model(data)           # Get model prediction
            guess = np.argmax(output)      # Pick highest probability class
            if guess != 0:
                getpuncch= False
                print(f"Predicted class: {guess} with confidence {output[0][guess]:.4f}")
                if guess == 1:
                    print("Perfect cross punch detected! Nice job")
                elif guess == 2:
                    print("you swung your elbow outward too much, try keeping your eblow in and ")
                elif guess == 3:
                    print("you swung your arm downward, like a swat in stead of a straight punch, don't think of punching down, think of punching straight ahead")
                frames = []
                # Reset frames for next prediction
                
                      

    vs.stop()  # Stop webcam thread
     
def main():
    # Expecting model path as command-line argument
    if len(sys.argv) != 1:
        model = load_model(sys.argv[1])
        getdata(model)
        cv2.destroyAllWindows()
    else:
        print("Need Command Line arguement: Model pkl file")

main()
