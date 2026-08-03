import cv2
import numpy as np
from ultralytics import YOLO
import time
import streamlit as st
import math

# Streamlit UI Config
st.set_page_config(page_title="Project Leopard - Autonomous Navigation Engine", layout="wide")
st.title("🐆 Project Leopard - Autonomous Navigation & Vision Engine")

@st.cache_resource
def load_model():
    return YOLO('yolov8n-seg.pt')

model = load_model()

# Sidebar Setup
st.sidebar.header("🎯 Target & Control Panel")
target_x = st.sidebar.slider("Target X Location (m)", 10, 390, 350)
target_y = st.sidebar.slider("Target Y Location (m)", 10, 390, 50)

run_sim = st.sidebar.checkbox("🚀 Start Autonomous Navigation", value=False)

# UI Layout
col1, col2, col3, col4 = st.columns(4)
speed_metric = col1.metric("Target Speed", "0 Km/h")
steering_metric = col2.metric("Steering Angle", "0 DEG")
dist_metric = col3.metric("Distance to Target", "0 m")
status_metric = col4.metric("System Status", "OFFLINE")

map_placeholder = st.empty()

# Vehicle Initial State
if 'car_pos' not in st.session_state:
    st.session_state.car_pos = [50.0, 350.0]  # Start Position (X, Y)
    st.session_state.car_heading = -90.0      # Heading angle in degrees (facing up)

if run_sim:
    log_file = open("auto_train_data.txt", "a", encoding="utf-8")
    
    # Simulation Loop
    while run_sim:
        # Create 2D Map Canvas (400x400)
        map_canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Draw Target Location (Red Circle)
        cv2.circle(map_canvas, (int(target_x), int(target_y)), 10, (0, 0, 255), -1)
        cv2.putText(map_canvas, "TARGET", (int(target_x) - 20, int(target_y) - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Draw Dynamic Obstacles (Yellow Circles)
        obstacles = [(200, 200, 25), (150, 120, 20), (280, 250, 30)]
        for ox, oy, orad in obstacles:
            cv2.circle(map_canvas, (ox, oy), orad, (0, 255, 255), -1)
            cv2.putText(map_canvas, "OBSTACLE", (ox - 25, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Current Car Coordinates
        cx, cy = st.session_state.car_pos
        
        # Calculate Distance to Target
        dx = target_x - cx
        dy = target_y - cy
        dist_to_target = math.hypot(dx, dy)
        
        # Navigation & Steering Logic
        if dist_to_target > 10:
            desired_angle = math.degrees(math.atan2(dy, dx))
            steering_angle = int(desired_angle - st.session_state.car_heading)
            
            # Normalize Angle (-180 to 180)
            steering_angle = (steering_angle + 180) % 360 - 180
            
            # Obstacle Avoidance Offset
            for ox, oy, orad in obstacles:
                dist_to_obs = math.hypot(ox - cx, oy - cy)
                if dist_to_obs < orad + 35:
                    steering_angle += 45  # Avoid Obstacle
                    
            # Vehicle Speed Control
            target_speed = 30 if abs(steering_angle) < 15 else 12
            traffic_status = "NAVIGATING TO LOCATION"
            
            # Kinematic Update (Move Vehicle)
            st.session_state.car_heading += steering_angle * 0.1
            rad = math.radians(st.session_state.car_heading)
            st.session_state.car_pos[0] += math.cos(rad) * (target_speed * 0.08)
            st.session_state.car_pos[1] += math.sin(rad) * (target_speed * 0.08)
        else:
            target_speed = 0
            steering_angle = 0
            traffic_status = "🎯 DESTINATION REACHED!"

        # Draw Vehicle Path Line
        cv2.line(map_canvas, (int(cx), int(cy)), (int(target_x), int(target_y)), (100, 100, 100), 1)

        # Draw EGO Vehicle (Green Arrow / Rect)
        car_x, car_y = int(st.session_state.car_pos[0]), int(st.session_state.car_pos[1])
        cv2.circle(map_canvas, (car_x, car_y), 8, (0, 255, 0), -1)
        
        # Vehicle Heading Direction Vector
        head_rad = math.radians(st.session_state.car_heading)
        head_x = int(car_x + math.cos(head_rad) * 20)
        head_y = int(car_y + math.sin(head_rad) * 20)
        cv2.line(map_canvas, (car_x, car_y), (head_x, head_y), (255, 255, 255), 2)
        cv2.putText(map_canvas, "MY CAR", (car_x - 20, car_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Display Navigation Output
        rgb_map = cv2.cvtColor(map_canvas, cv2.COLOR_BGR2RGB)
        map_placeholder.image(rgb_map, caption="2D Live Autonomous Map & Kinematic Simulation", width=600)
        
        # Update Telemetry Metrics
        speed_metric.metric("Target Speed", f"{int(target_speed)} Km/h")
        steering_metric.metric("Steering Angle", f"{int(steering_angle)} DEG")
        dist_metric.metric("Distance to Target", f"{int(dist_to_target)} m")
        status_metric.metric("System Status", traffic_status)
        
        log_file.write(f"{time.time()},{target_speed},{steering_angle},{traffic_status}\n")
        time.sleep(0.05)
        
    log_file.close()
