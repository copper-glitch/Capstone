# import cv2
# import os
# import numpy as np

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
# https://www.geeksforgeeks.org/python/python-opencv-capture-video-from-camera/


# Initialize MediaPipe Pose and Drawing utilities
import cv2
import sys
import time
# Open the default camera
if len(sys.argv)!=1:
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    cam.set(cv2.CAP_PROP_FPS, 30)
    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    read = False
    wait = False
    waitcount = 30
    count = 0
    vidnum = 0
    # prev_time = time.time()
    while True:
        
        ret, frame = cam.read()
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key== ord('g'):
            wait = True
            vidnum+=1
            out = cv2.VideoWriter(f'myrecord/{sys.argv[1]}{vidnum}.avi', fourcc, 30.0, (frame_width, frame_height))
        if wait == True:
            waitcount -= 1
            if waitcount == 0:
                read = True
                wait = False
                waitcount = 30
        # Write the frame to the output file
        if read == True:
            out.write(frame)
            count+=1
        if count == 60:
            count = 0
            read = False
            out.release()
        # Display the captured frame
        # now = time.time()
        # fps = 1 / (now - prev_time)
        # prev_time = now
        cv2.putText(frame, f'Frame: {count} waitcount: {waitcount} vidnum {vidnum}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera', frame)


    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()           
else:
    print("Need Command Line arguement: Recording Name")            
        