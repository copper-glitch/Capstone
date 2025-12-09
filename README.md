Coach AI
**AI-powered motion analysis for a Right Straight**

## 📖 Overview
**Coach AI** is a computer vision project that analyzes human body mechanics in real time using a webcam.  
It focuses on evaluating **a right straight punch**, providing **instant feedback** on technique, balance, and alignment.

---

## 🎯 Features
- 📸 **Webcam-based motion tracking** – No special equipment required.  
- 🧠 **Pose estimation & angle detection** – Uses AI models (e.g., MediaPipe or OpenPose) to identify key joints and body angles.  
- 🥊 **Punch analysis** – Evaluates the mechanics of a right straight punch: shoulder rotation, hip alignment, and follow-through.  
- 💬 **Instant feedback** – Visual or audio cues guide users on what to adjust.  
---

## 🧩 Tech Stack
- **Language:** Python  
- **Frameworks/Libraries:**  
  - [MediaPipe](https://google.github.io/mediapipe/) or [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)  
  - [OpenCV](https://opencv.org/) for video capture and visualization  
  - NumPy / Pandas for data analysis  
To Run, I provided an example dataset to create the model, you might need to change some of the path variables to make it work. Run the multifile on the videos, wouldn't recommend downloading it's a good amount of videos. I provided the pdata which you can run the cnn on, then just pass the name of the model to the test.py
