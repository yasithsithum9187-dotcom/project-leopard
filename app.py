import cv2
import numpy as np
from ultralytics import YOLO
import time
import streamlit as st
import math
import tempfile

st.set_page_config(page_title="Project Leopard - Dual Autonomous Engine", layout="wide")
st.title("🐆 Project Leopard - Integrated Vision & Navigation Dashboard")

@st.cache_resource
def load_model():
    return YOLO('yolov8n-seg.pt')

model = load_model()

# Sidebar Controls
st.sidebar.header("🕹️ Project Leopard Controls")
video_source = st.sidebar.radio("Select Driving Feed:", ["Demo Video (ride_test.mp4)", "Upload Custom Video"])

if video_source == "Upload Custom Video":
    uploaded_file = st.sidebar.file_uploader("Upload Driving Video (.mp4)", type=['mp4', 'avi', 'mov'])
else:
    uploaded_file = None

target_x = st.sidebar.slider("Target Destination X", 20, 380, 320)
target_y = st.sidebar.slider("Target Destination Y", 20, 380, 60)

run_sim = st.sidebar.checkbox("🚀 Launch Integrated Autonomous Simulation", value=False)

# Top Metrics
col1, col2, col3, col4 = st.columns(4)
speed_metric = col1.metric("Target Speed", "0 Km/h")
steering_metric = col2.metric("Steering Angle", "0 DEG")
dist_metric = col3.metric("Distance to Target", "0 m")
status_metric = col4.metric("System Status", "OFFLINE")

# Side-by-Side Displays (Left: Vision, Right: Radar Map)
v_col, m_col = st.columns(2)
vision_placeholder = v_col.empty()
map_placeholder = m_col.empty()

if 'car_pos' not in st.session_state:
    st.session_state.car_pos = [60.0, 340.0]
    st.session_state.car_heading = -90.0

if run_sim:
    # Determine video stream source
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
    else:
        cap = cv2.VideoCapture("ride_test.mp4")

    while run_sim and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize for smooth performance
        frame = cv2.resize(frame, (640, 480))
        
        # 1. AI Vision Processing
        results = model(frame, conf=0.35, verbose=False)
        annotated_frame = results[0].plot()

        # 2. 2D Radar Map Canvas
        map_canvas = np.zeros((480, 480, 3), dtype=np.uint8)
        
        # Grid lines for visual appeal
        for i in range(0, 480, 40):
            cv2.line(map_canvas, (i, 0), (i, 480), (25, 25, 25), 1)
            cv2.line(map_canvas, (0, i), (480, i), (25, 25, 25), 1)

        # Draw Target
        cv2.circle(map_canvas, (int(target_x), int(target_y)), 12, (0, 0, 255), -1)
        cv2.putText(map_canvas, "TARGET", (int(target_x)-25, int(target_y)-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # Draw Obstacles
        obstacles = [(220, 220, 30), (140, 150, 25), (300, 280, 35)]
        for ox, oy, orad in obstacles:
            cv2.circle(map_canvas, (ox, oy), orad, (0, 255, 255), -1)
            cv2.putText(map_canvas, "OBSTACLE", (ox-30, oy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Kinematic Steering Logic
        cx, cy = st.session_state.car_pos
        dx = target_x - cx
        dy = target_y - cy
        dist_to_target = math.hypot(dx, dy)

        if dist_to_target > 15:
            desired_angle = math.degrees(math.atan2(dy, dx))
            steering_angle = int(desired_angle - st.session_state.car_heading)
            steering_angle = (steering_angle + 180) % 360 - 180

            for ox, oy, orad in obstacles:
                if math.hypot(ox - cx, oy - cy) < orad + 40:
                    steering_angle += 45

            target_speed = 35 if abs(steering_angle) < 15 else 15
            status_str = "AUTONOMOUS NAVIGATING"

            st.session_state.car_heading += steering_angle * 0.08
            rad = math.radians(st.session_state.car_heading)
            st.session_state.car_pos[0] += math.cos(rad) * (target_speed * 0.07)
            st.session_state.car_pos[1] += math.sin(rad) * (target_speed * 0.07)
        else:
            target_speed = 0
            steering_angle = 0
            status_str = "🎯 DESTINATION REACHED"

        # Draw Ego Vehicle
        car_x, car_y = int(st.session_state.car_pos[0]), int(st.session_state.car_pos[1])
        cv2.circle(map_canvas, (car_x, car_y), 10, (0, 255, 0), -1)
        cv2.line(map_canvas, (car_x, car_y), (int(target_x), int(target_y)), (100, 100, 100), 1)
        
        # Vehicle Direction Indicator
        head_rad = math.radians(st.session_state.car_heading)
        cv2.line(map_canvas, (car_x, car_y),
                 (int(car_x + math.cos(head_rad)*25), int(car_y + math.sin(head_rad)*25)),
                 (255, 255, 255), 3)

        # Convert colors & Display Side-by-Side
        vision_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        map_rgb = cv2.cvtColor(map_canvas, cv2.COLOR_BGR2RGB)

        vision_placeholder.image(vision_rgb, caption="📷 Real-Time AI Camera Vision (YOLO Perception)", use_container_width=True)
        map_placeholder.image(map_rgb, caption="🗺️ 2D Navigation Radar & Path Planner", use_container_width=True)

        speed_metric.metric("Target Speed", f"{int(target_speed)} Km/h")
        steering_metric.metric("Steering Angle", f"{int(steering_angle)} DEG")
        dist_metric.metric("Distance to Target", f"{int(dist_to_target)} m")
        status_metric.metric("System Status", status_str)

        time.sleep(0.03)

    cap.release()
