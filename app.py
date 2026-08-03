import streamlit as st
import numpy as np
import cv2
import random
from ultralytics import YOLO

st.set_page_config(page_title="Project Leopard Dashboard", layout="wide")
st.title("🐆 Project Leopard - Web Autonomous Dashboard")

# Load YOLO Model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Simulation Environment
class Environment:
    def __init__(self):
        self.car_speed = 40.0
        self.oncoming_cars = [{'x': -1.5, 'z': 100, 'speed': 50}]
        self.road_debris = [{'x': 0.2, 'z': 60}]
        self.rear_cars = [{'x': 1.5, 'z': -15, 'speed': 60}]

    def update(self):
        for car in self.oncoming_cars:
            car['z'] -= (self.car_speed + car['speed']) * 0.05
            if car['z'] < 2:
                car['z'] = 120
                car['x'] = random.choice([-1.8, -1.2])

        for debris in self.road_debris:
            debris['z'] -= self.car_speed * 0.05
            if debris['z'] < 1:
                debris['z'] = 80
                debris['x'] = random.uniform(-0.8, 0.8)

env = Environment()

# Streamlit UI
col1, col2 = st.columns([3, 1])

with col1:
    frame_placeholder = st.empty()

with col2:
    st.subheader("📊 Telemetry Data")
    speed_metric = st.empty()
    status_metric = st.empty()

# Run Simulation
if st.button("🚀 Start Simulation Stream"):
    for _ in range(300): # Simulation frames
        env.update()

        front_cam = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.fillPoly(front_cam, [np.array([[100, 360], [270, 180], [370, 180], [540, 360]])], (60, 60, 60))
        cv2.line(front_cam, (320, 180), (320, 360), (255, 255, 255), 2)

        # Debris (Obstacles)
        for obj in env.road_debris:
            if obj['z'] > 2:
                scale = max(0.1, 30 / obj['z'])
                cx, cy = int(320 + (obj['x'] * scale * 100)), int(180 + (scale * 80))
                size = int(20 * scale)
                cv2.rectangle(front_cam, (cx - size, cy - size), (cx + size, cy + size), (0, 100, 200), -1)

        # Oncoming Cars
        for car in env.oncoming_cars:
            if car['z'] > 2:
                scale = max(0.1, 30 / car['z'])
                cx, cy = int(320 + (car['x'] * scale * 100)), int(180 + (scale * 80))
                w, h = int(40 * scale), int(30 * scale)
                cv2.rectangle(front_cam, (cx - w, cy - h), (cx + w, cy + h), (0, 0, 255), -1)

        # YOLO AI Detection
        results = model(front_cam, verbose=False)
        front_annotated = results[0].plot()

        # Rear Camera View
        rear_cam = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.putText(rear_cam, f"REAR CAR: {round(abs(env.rear_cars[0]['z']), 1)}m", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        final_dash = np.zeros((540, 640, 3), dtype=np.uint8)
        final_dash[0:360, 0:640] = front_annotated
        final_dash[360:540, 160:480] = cv2.resize(rear_cam, (320, 180))

        # Streamlit Image Display
        frame_rgb = cv2.cvtColor(final_dash, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        speed_metric.metric("Speed", f"{env.car_speed} KM/H")
        status_metric.info("System Status: AUTO-PILOT ACTIVE")
