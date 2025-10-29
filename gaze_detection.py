import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

# Setup MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

# App title
st.title("👁️ Real-Time Eye Gaze Detection")
st.write("Detects if you’re looking away from the screen using your webcam.")

# Sidebar controls
LOOK_AWAY_THRESHOLD = st.sidebar.slider("Look-away alert time (seconds)", 2.0, 10.0, 5.0)
GAZE_CENTER_RANGE = (0.35, 0.65)

# Helper functions
def get_coord(landmark, shape):
    """Convert normalized landmark coordinates to pixel coordinates"""
    return int(landmark.x * shape[1]), int(landmark.y * shape[0])

def calculate_eye_ratio(iris_center, outer_corner, inner_corner):
    """Calculate gaze ratio (0–1) where 0.5 ≈ center"""
    eye_width = inner_corner[0] - outer_corner[0]
    iris_offset = iris_center[0] - outer_corner[0]
    return iris_offset / eye_width if eye_width != 0 else 0.5

def is_looking_center(left_ratio, right_ratio):
    """Check if both eyes are directed roughly at the center"""
    return (
        GAZE_CENTER_RANGE[0] <= left_ratio <= GAZE_CENTER_RANGE[1]
        and GAZE_CENTER_RANGE[0] <= right_ratio <= GAZE_CENTER_RANGE[1]
    )

# Streamlit display placeholders
status_placeholder = st.empty()
frame_window = st.image([])

# Start webcam button
run = st.sidebar.checkbox("Start Camera")

look_away_start_time = None
alert_active = False

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not access webcam.")
    else:
        st.info("✅ Webcam started. Press the checkbox again to stop.")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Failed to capture frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                get_landmark = lambda i: get_coord(face.landmark[i], (h, w))

                left_center = tuple(np.mean([get_landmark(i) for i in LEFT_IRIS], axis=0).astype(int))
                right_center = tuple(np.mean([get_landmark(i) for i in RIGHT_IRIS], axis=0).astype(int))

                left_outer, left_inner = get_landmark(LEFT_EYE[0]), get_landmark(LEFT_EYE[1])
                right_outer, right_inner = get_landmark(RIGHT_EYE[0]), get_landmark(RIGHT_EYE[1])

                left_ratio = calculate_eye_ratio(left_center, left_outer, left_inner)
                right_ratio = calculate_eye_ratio(right_center, right_outer, right_inner)

                center = is_looking_center(left_ratio, right_ratio)
                now = time.time()

                if not center:
                    if look_away_start_time is None:
                        look_away_start_time = now
                    elapsed = now - look_away_start_time

                    if elapsed > LOOK_AWAY_THRESHOLD and not alert_active:
                        st.warning(f"⚠️ You’ve been looking away for {elapsed:.1f} seconds!")
                        alert_active = True

                    status = f"🔴 Looking away ({elapsed:.1f}s)"
                else:
                    look_away_start_time = None
                    alert_active = False
                    status = "🟢 Looking at screen"

                status_placeholder.markdown(f"### {status}")

                # Draw face mesh points (optional visualization)
                for pt in [left_center, right_center]:
                    cv2.circle(frame, pt, 3, (0, 255, 255), -1)

                cv2.putText(
                    frame,
                    f"L:{left_ratio:.2f} R:{right_ratio:.2f}",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                status_placeholder.markdown("### 🚫 No face detected")

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
else:
    st.info("👆 Check 'Start Camera' in sidebar to begin gaze detection.")

