
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
from io import BytesIO

# Load the trained model
model = load_model("modelnew.h5")

# Set Streamlit page configuration
st.set_page_config(layout="wide")
st.title("Violence Detection System")

# Function to detect violence in a frame
def detect_violence(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction[0][0]

# Function to process uploaded video and overlay detection results
def process_uploaded_video_with_overlay(input_path, output_path="processed_output.mp4"):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Try a more compatible codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 'avc1' for mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        confidence = detect_violence(frame)
        violence_text = "Violence: True" if confidence > 0.5 else "Violence: False"
        color = (0, 0, 255) if confidence > 0.5 else (0, 255, 0)

        cv2.putText(frame, f"{violence_text} ({confidence*100:.2f}%)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()


# Sidebar for choosing mode
option = st.sidebar.radio("Choose Mode:", ["Live Camera", "Upload Video"])

# --- Live Camera Mode ---
if option == "Live Camera":
    st.subheader("Live Violence Detection")

    start_webcam = st.button("Start Live Video")
    stop_webcam = st.button("Stop Live Video")
    stframe = st.empty()

    if start_webcam:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Could not access the webcam.")
                break

            frame = cv2.resize(frame, (640, 480))
            confidence = detect_violence(frame)
            violence_text = "Violence: True" if confidence > 0.5 else "Violence: False"
            color = (0, 0, 255) if confidence > 0.5 else (0, 255, 0)

            cv2.putText(frame, f"{violence_text} ({confidence*100:.2f}%)", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", width=640)

            if stop_webcam:
                break

        cap.release()
        cv2.destroyAllWindows()
        st.success("Live video stopped.")

# --- Upload Video Mode ---
if option == "Upload Video":
    st.subheader("Upload Video for Violence Detection")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        input_video_path = tfile.name

        # Prepare output path
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

        # Process video and add violence overlay
        process_uploaded_video_with_overlay(input_video_path, output_path)

        st.subheader("Processed Output Video")
        # Read and show processed video
        with open(output_path, 'rb') as f:
            video_bytes = f.read()
            st.video(output_path)

            # Download button
            st.download_button("Download Processed Video", video_bytes, file_name="violence_output.mp4", mime="video/mp4")


