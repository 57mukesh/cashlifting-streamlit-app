import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO
import uuid
import ffmpeg

# Load model
@st.cache_resource
def load_model():
    model_path = r"C:\Users\mukes\cashlifting_poc\runs\detect\cashlifting_augmented_v4b\weights\best.pt"
    return YOLO(model_path)

model = load_model()

st.title("📦 Cashlifting Detection")
st.markdown("Upload your `.mp4` video. We’ll detect suspicious actions and show the output video.")

uploaded_file = st.file_uploader("📤 Upload video", type=["mp4"])

if uploaded_file:
    # Save uploaded file
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_file.read())
    input_video_path = temp_input.name

    st.subheader("Original Video")
    st.video(input_video_path)

    # Prepare for writing output
    cap = cv2.VideoCapture(input_video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Step 1: Write with OpenCV to temp raw file
    raw_output_path = os.path.join(tempfile.gettempdir(), f"raw_{uuid.uuid4().hex}.mp4")
    out = cv2.VideoWriter(raw_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    st.info("🔍 Detecting... Please wait...")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]
        annotated = results.plot()
        out.write(annotated)
        frame_count += 1

    cap.release()
    out.release()

    st.success(f"✅ Detection complete! Processed {frame_count} frames.")

    # Step 2: Use ffmpeg to re-encode for browser playback
    final_output_path = os.path.join(tempfile.gettempdir(), f"final_{uuid.uuid4().hex}.mp4")
    ffmpeg.input(raw_output_path).output(final_output_path, vcodec='libx264', crf=23).run(overwrite_output=True)

    # Step 3: Display and download final video
    with open(final_output_path, "rb") as vid_file:
        video_bytes = vid_file.read()

    st.subheader("🎬 Processed Video")
    st.video(video_bytes)
    st.download_button("📥 Download Result Video", data=video_bytes, file_name="cashlifting_result.mp4")
