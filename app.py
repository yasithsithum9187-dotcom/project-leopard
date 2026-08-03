import cv2
import numpy as np
from ultralytics import YOLO
import time
import streamlit as st
import tempfile
import os

# Streamlit UI Config
st.set_page_config(page_title="Project Leopard - Web Dashboard", layout="wide")
st.title("🐆 Project Leopard - Autonomous Vision Dashboard")

# Load YOLO Model (Auto downloads if missing)
@st.cache_resource
def load_model():
    return YOLO('yolov8n-seg.pt')

model = load_model()

# Sidebar Controls
st.sidebar.header("🕹️ Control Panel")
video_source = st.sidebar.radio("Select Video Source:", ("Upload Video File", "Use Camera / Demo Video"))

uploaded_file = None
if video_source == "Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload Driving Video (.mp4 / .avi)", type=['mp4', 'avi', 'mov'])

run_sim = st.sidebar.checkbox("🚀 Run Autonomous Simulation", value=False)

# UI Layout Placeholders
col1, col2, col3 = st.columns(3)
speed_metric = col1.metric("Target Speed", "0 Km/h")
steering_metric = col2.metric("Steering Angle", "0 DEG")
status_metric = col3.metric("System Status", "OFFLINE")

video_placeholder = st.empty()

CAMERA_HEIGHT = 1.4  
PITCH_ANGLE = 0.22   

if run_sim:
    # Handle Video Capture Source
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap_front = cv2.VideoCapture(tfile.name)
    else:
        # Fallback to local file if present, else default webcam
        video_path = 'ride_test.mp4' if os.path.exists('ride_test.mp4') else 0
        cap_front = cv2.VideoCapture(video_path)

    log_file = open("auto_train_data.txt", "w", encoding="utf-8")
    log_file.write("timestamp,target_speed,steering_angle,traffic_status,emergency_brake\n")

    prev_objects = {} 
    last_time = time.time()

    while cap_front.isOpened() and run_sim:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        if dt == 0: dt = 0.03

        ret, frame_front = cap_front.read()
        if not ret:
            st.warning("Video stream ended or unable to open.")
            break
            
        height, width, _ = frame_front.shape
        camera_center_x = int(width / 2)
        
        # 🖼️ 360° Virtual CAR Dashboard Canvas
        dash_360 = np.zeros((400, 400, 3), dtype=np.uint8)
        ego_car_x, ego_car_y = 200, 200
        cv2.rectangle(dash_360, (ego_car_x - 15, ego_car_y - 25), (ego_car_x + 15, ego_car_y + 25), (255, 255, 255), -1)
        cv2.putText(dash_360, "MY CAR", (180, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            
        # Road Lane Segmentation (HSV)
        hsv = cv2.cvtColor(frame_front, cv2.COLOR_BGR2HSV)
        road_mask = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 40, 200]))
        road_mask[int(height * 0.80):height, :] = 0  
        
        # AI Object Detection & Tracking
        results = model.track(frame_front, persist=True, device='cpu', conf=0.45, verbose=False)
        obstacle_mask = np.zeros((height, width), dtype=np.uint8)
        
        target_speed = 40  
        traffic_status = "CAR VISION: ACTIVE"
        emergency_brake = False
        steering_angle = 0  
        avoidance_offset = 0
        stop_sign_detected = False
        current_frame_objects = {}
        
        if results[0].masks is not None:
            boxes = results[0].boxes
            masks = results[0].masks.xy
            
            for idx, (mask, box) in enumerate(zip(masks, boxes)):
                obj_id = int(box.id[0]) if box.id is not None else idx
                cls = int(box.cls[0])
                points = np.array(mask, dtype=np.int32)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                obj_width = x2 - x1
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                screen_y_normalized = (height - y2) / height
                actual_distance = max(0.5, round(CAMERA_HEIGHT / np.tan(PITCH_ANGLE + (screen_y_normalized * 0.5)), 1))
                
                real_speed_kmh = 0
                if obj_id in prev_objects:
                    distance_moved = prev_objects[obj_id]['distance'] - actual_distance
                    real_speed_kmh = max(0, round((distance_moved / dt) * 3.6, 1))
                
                current_frame_objects[obj_id] = {'distance': actual_distance, 'width': obj_width, 'center': cx}
                
                if cls == 9: # Traffic Light
                    crop = frame_front[y1:y2, x1:x2]
                    if crop.size > 0:
                        r_mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), np.array([0, 100, 100]), np.array([10, 255, 255]))
                        if np.sum(r_mask) > 100:
                            traffic_status = "TRAFFIC LIGHT: RED"
                            emergency_brake = True
                
                elif cls == 11: # Stop Sign
                    if actual_distance < 15.0:
                        stop_sign_detected = True
                        traffic_status = "🛑 STOP SIGN: SCANNING JUNCTION"
                        target_speed = 0

                if cls in [0, 2, 3, 5, 7, 15, 16, 17, 18, 19]:
                    if actual_distance < 3.0 and (x1 < camera_center_x + 150 and x2 > camera_center_x - 150):
                        traffic_status = "🚨 DANGER: TOO CLOSE!"
                        emergency_brake = True
                    
                    elif 3.0 <= actual_distance < 10.0 and (x1 < camera_center_x + 100 and x2 > camera_center_x - 100):
                        cv2.fillPoly(obstacle_mask, [points], 255)
                        target_speed = min(target_speed, 15)
                        
                        if cx < camera_center_x:
                            avoidance_offset = 150  
                            traffic_status = "🔄 AVOIDING: STEERING RIGHT"
                        else:
                            avoidance_offset = -150 
                            traffic_status = "🔄 AVOIDING: STEERING LEFT"

                elif cls not in [0, 1, 2, 3, 5, 7, 11] and obj_width < 80:
                    target_speed = min(target_speed, 15)
                else:
                    cv2.fillPoly(obstacle_mask, [points], 255)
                
                color = (0, 255, 255) if cls not in [0,2,3,5,7,15,16,17,18,19] else (255, 0, 0)
                cv2.rectangle(frame_front, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_front, f"{actual_distance}m", (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                            
        prev_objects = current_frame_objects.copy()
        
        # Path Planning
        safe_road_mask = cv2.bitwise_and(road_mask, cv2.bitwise_not(obstacle_mask))
        annotated_frame = frame_front.copy()
        
        if emergency_brake:
            cv2.putText(annotated_frame, "🚨 EMERGENCY BRAKE 🚨", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 2)
            target_speed = 0
        elif stop_sign_detected:
            cv2.putText(annotated_frame, "🛑 YIELDING AT JUNCTION", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 165, 255), 2)
        else:
            M = cv2.moments(safe_road_mask)
            if M["m00"] != 0:
                path_center_x = int(M["m10"] / M["m00"])
                path_center_y = int(M["m01"] / M["m00"])
                left_side_target_x = (path_center_x - 120) + avoidance_offset 
            else:
                left_side_target_x = camera_center_x
                path_center_y = int(height * 0.5)

            deviation = left_side_target_x - camera_center_x
            steering_angle = int(deviation / 5)
            
            if abs(steering_angle) > 15: target_speed = min(target_speed, 20)
            
            cv2.circle(annotated_frame, (left_side_target_x, path_center_y), 12, (0, 255, 0), -1)
            cv2.line(annotated_frame, (camera_center_x, height), (left_side_target_x, path_center_y), (0, 255, 255), 4)
            
            text = f"STEER RIGHT: {steering_angle} DEG" if deviation > 50 else f"STEER LEFT: {abs(steering_angle)} DEG" if deviation < -50 else "LANE KEEP: STRICT LEFT"
            cv2.putText(annotated_frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Telemetry displays
        cv2.putText(annotated_frame, f"CAR SPEED: {target_speed} Km/h", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"SYSTEM: {traffic_status}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        overlay = annotated_frame.copy()
        overlay[safe_road_mask > 0] = [0, 255, 0]
        annotated_frame = cv2.addWeighted(overlay, 0.25, annotated_frame, 0.75, 0)
        
        # Dashboard Display Stack
        dash_resized = cv2.resize(dash_360, (int(height * (400/400)), height))
        final_output = np.hstack((annotated_frame, dash_resized))
        
        # Convert BGR to RGB for Web Streaming
        rgb_frame = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)
        
        # Update Web UI Components
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        speed_metric.metric("Target Speed", f"{target_speed} Km/h")
        steering_metric.metric("Steering Angle", f"{steering_angle} DEG")
        status_metric.metric("System Status", traffic_status)
        
        log_file.write(f"{time.time()},{target_speed},{steering_angle},{traffic_status},{emergency_brake}\n")

    cap_front.release()
    log_file.close()
