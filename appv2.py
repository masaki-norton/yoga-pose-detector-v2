import streamlit as st
import numpy as np
import joblib
import mediapipe as mp
import tensorflow as tf
import warnings
import av
import cv2
from queue import Queue
from get_landmarks import get_landmarks_simple, get_landmarks_from_pose
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

tf.get_logger().setLevel("ERROR")
warnings.filterwarnings('ignore', message="X does not have valid feature names")

# Load models
model = tf.keras.models.load_model("models/nn_v1_1.h5")
pipeline = joblib.load("models/pipeline_v1_1.pkl")

# Define possible poses
poses = ['downdog', 'goddess', 'plank', 'tree_chest', 'tree_up', 'warrior2_left', 'warrior2_right']

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Queue to hold results for updating the UI
results_queue = Queue()

class VideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.pose = mp_pose.Pose(static_image_mode=False,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = get_landmarks_simple(img, self.pose)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lmks = get_landmarks_from_pose(results)
            lmks_transformed = pipeline.transform(np.array([lmks]))
            pred = model.predict(lmks_transformed, verbose=0)

            # Put the results in the queue
            results_queue.put({
                "pose": poses[np.argmax(pred)],
                "confidence": np.max(pred.round(2))
            })

        return av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="rgb24")

# Initialize frames
st.title("Yoga Pose Estimation")


RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

output_box = st.empty()

# Function to update the UI based on queue data
def update_ui():
    while True:
        if not results_queue.empty():
            result = results_queue.get()
            st.session_state.output_text = (
                f"Predicted Pose: {result['pose']} "
                f"Pred. Confidence: {result['confidence']}"
            )
            output_box.markdown(
                f"<div style='text-align: center; font-size: 24px;'>{st.session_state.output_text}</div>",
                unsafe_allow_html=True
            )
        else:
            break

if "output_text" not in st.session_state:
    st.session_state.output_text = ""

# Update the UI in an event-driven manner
st.experimental_rerun = True  # This should be used to trigger the rerun in an event-driven way

if st.experimental_rerun:
    update_ui()
