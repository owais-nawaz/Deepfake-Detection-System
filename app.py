# app.py

import streamlit as st
import torch
from helper import (
    load_model,
    ValidationDataset,
    ValidationDataset_raw,
    train_transforms,
    get_prediction_confidence,
)
import tempfile
from torchvision import transforms
import os

models_directory = "./models"  # Update this path if needed

# Get a list of available model files
available_models = [f for f in os.listdir(models_directory)]


# Main app code
def main():
    st.title("Deepfake Video Detection")
    st.write("Upload a video to check if it contains deepfake content.")

    # Video upload
    uploaded_video = st.file_uploader(
        "Choose a video file", type=["mp4", "avi", "mov", "mkv"]
    )
    # Sequence length input
    sequence_length = st.number_input(
        "Enter sequence length for prediction", min_value=1, value=20, step=1
    )

    # Model selection dropdown
    model_name = st.selectbox("Select a model for prediction", available_models)

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if uploaded_video is not None:
        # Display the uploaded video
        # st.video(uploaded_video)

        temp_dir = "temp_video"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, uploaded_video.name)
        # Save uploaded video to a temporary location

        with open(temp_file, "wb") as f:
            f.write(uploaded_video.getbuffer())

        # st.video(temp_file)

        st.write("Processing video...")

        try:

            # Load and process video frames
            video_dataset = ValidationDataset_raw(
                temp_file,
                sequence_length=sequence_length,
            )
            face_frames = video_dataset.get_face_extracted_frames()
            st.image(
                face_frames,
                width=100,
                caption=[f"Face Frame {i+1}" for i in range(len(face_frames))],
            )

            video_dataset = ValidationDataset(
                temp_file,
                sequence_length=sequence_length,
                transform=train_transforms,
            )
            frames = next(iter(video_dataset))
            frames = frames.to(device)

            # Load model and move to the same device
            model = load_model(model_name)
            model = model.to(device)

            # Make prediction
            _, logits = model(frames)  # Adding batch dimension if needed
            prediction, confidence = get_prediction_confidence(logits)

            # Display results
            st.write(f"Prediction: **{prediction}**")
            st.write(f"Confidence: **{confidence:.2f}%**")

        finally:
            # Cleanup temporary video file
            os.remove(temp_file)


if __name__ == "__main__":
    main()
