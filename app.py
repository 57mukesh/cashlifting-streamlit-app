import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO
import uuid
import ffmpeg
import gdown  # Added for Google Drive download

# Model download from Google Drive if not already present
@st.cache_resource
def load_model():
    model_path = "model/best.pt"  # Use relative path for deployment

    if not os.path.exists(model_path):
        st.warning("🔄 Downloading model from Google Drive... Please wait...")
        os.makedirs("model", exist_ok=True)
        file_id = "1cNsMOayjiGrbR53XlC721x8TzV88RXmV"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

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

    import time

    # Step 2: Use ffmpeg to re-encode for browser playback (after ensuring file exists)
    final_output_path = os.path.join(tempfile.gettempdir(), f"final_{uuid.uuid4().hex}.mp4")

    # Wait briefly to ensure the file system syncs
    time.sleep(1)

    if os.path.exists(raw_output_path):
        ffmpeg.input(raw_output_path).output(final_output_path, vcodec='libx264', crf=23).run(overwrite_output=True)
    else:
        st.error("⚠️ Failed to process video: raw output file not found.")
        st.stop()


    # Step 3: Display and download final video
    with open(final_output_path, "rb") as vid_file:
        video_bytes = vid_file.read()

    st.subheader("🎬 Processed Video")
    st.video(video_bytes)
    st.download_button("📥 Download Result Video", data=video_bytes, file_name="cashlifting_result.mp4")
