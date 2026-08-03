import gradio as gr
import numpy as np
import cv2
import math
import random
from ultralytics import YOLO

# Cloud AI Model Loading
model = YOLO('yolov8n.pt')

class Environment:
    def __init__(self):
        self.car_speed = 40.0
        self.oncoming_cars = [{'x': -1.5, 'z': 100, 'speed': 50}]
        self.rear_cars = [{'x': 1.5, 'z': -15, 'speed': 60}]
        self.road_debris = [{'x': 0.2, 'z': 60}]
        
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

def run_simulation_frame():
    env.update()
    
    # 1. Front Camera View Generation
    front_cam = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.fillPoly(front_cam, [np.array([[100, 360], [270, 180], [370, 180], [540, 360]])], (60, 60, 60))
    cv2.line(front_cam, (320, 180), (320, 360), (255, 255, 255), 2)
    
    # Render Obstacles & Vehicles
    for obj in env.road_debris:
        if obj['z'] > 2:
            scale = max(0.1, 30 / obj['z'])
            cx, cy = int(320 + (obj['x'] * scale * 100)), int(180 + (scale * 80))
            size = int(20 * scale)
            cv2.rectangle(front_cam, (cx - size, cy - size), (cx + size, cy + size), (0, 100, 200), -1)

    for car in env.oncoming_cars:
        if car['z'] > 2:
            scale = max(0.1, 30 / car['z'])
            cx, cy = int(320 + (car['x'] * scale * 100)), int(180 + (scale * 80))
            w, h = int(40 * scale), int(30 * scale)
            cv2.rectangle(front_cam, (cx - w, cy - h), (cx + w, cy + h), (0, 0, 255), -1)

    # YOLO AI Vision Processing
    results = model(front_cam, verbose=False)
    front_annotated = results[0].plot()

    # 2. Mirror / Rear Cam View
    rear_cam = np.zeros((180, 320, 3), dtype=np.uint8)
    for r_car in env.rear_cars:
        dist = abs(r_car['z'])
        cv2.putText(rear_cam, f"REAR VEHICLE: {round(dist, 1)}m", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Combined Dashboard Output
    final_dash = np.zeros((540, 640, 3), dtype=np.uint8)
    final_dash[0:360, 0:640] = front_annotated
    final_dash[360:540, 160:480] = cv2.resize(rear_cam, (320, 180))
    
    cv2.putText(final_dash, f"PROJECT LEOPARD - SPEED: {env.car_speed} KM/H", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return cv2.cvtColor(final_dash, cv2.COLOR_BGR2RGB)

# Gradio Web App Setup
with gr.Blocks(title="Project Leopard Cloud Simulator") as demo:
    gr.Markdown("# 🐆 Project Leopard - Web Autonomous Cloud Simulator")
    img_display = gr.Image(label="Live Camera Stream")
    demo.load(fn=run_simulation_frame, inputs=None, outputs=img_display, every=0.1)

demo.launch()