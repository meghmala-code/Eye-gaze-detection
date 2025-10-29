# 👁️ Eye Gaze Detection using MediaPipe

This project detects whether a user is looking **toward** or **away from the screen** in real-time using their webcam.  
It uses **MediaPipe FaceMesh** to track eye landmarks and estimate gaze direction based on iris positions.

---

## 🚀 Features
- Real-time webcam feed with gaze tracking.
- Detects when the user looks away for more than a given threshold (default 5 seconds).
- Displays on-screen alerts when the user loses focus.
- Adjustable alert timer from Streamlit sidebar.
- Lightweight and runs locally — no heavy deep learning model required.

---

## 🧠 How It Works
The app uses **MediaPipe’s FaceMesh** model to get facial landmarks (468 points on the face).  
It focuses on:
- **Iris landmarks** for both eyes (`LEFT_IRIS`, `RIGHT_IRIS`).
- **Eye corners** to compute a gaze ratio.  

If both eyes’ ratios fall within a central range, the app assumes the user is looking at the screen.  
Otherwise, it counts how long they’ve been looking away and triggers a warning if it exceeds the threshold.

---

## 🖥️ Tech Stack
- **Python**
- **Streamlit**
- **OpenCV**
- **MediaPipe**
- **NumPy**

---
