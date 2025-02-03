import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO

def measure_sharpness(image):
    """ Mede a nitidez do frame usando a variância do Laplaciano """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def select_best_frame(frames):
    """ Seleciona o frame mais nítido da lista """
    return max(frames, key=lambda f: measure_sharpness(f))

output_dir = "./bestFrame/bestFramepaste"
os.makedirs(output_dir, exist_ok=True)

model = YOLO("./checkpoints/checkpoints/v2/weights/YOLO-MEDIUM/runs/segment/train/weights/best.pt")

video_path = "./bestFrame/out_video/output_video.mp4"  
cap = cv2.VideoCapture(video_path)

detected_frames = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    
    frame_resized = cv2.resize(frame, (640, 640))  
    
    
    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0  
    frame_tensor = frame_tensor.unsqueeze(0)  

    
    results = model(frame_tensor)  
    
    for result in results:
        for box in result.boxes:
            conf = box.conf.item()
            if conf > 0.8:  
                detected_frames.append(frame)

    
    if len(detected_frames) >= 5:
        best_frame = select_best_frame(detected_frames)
        output_path = os.path.join(output_dir, f"best_frame_{frame_count}.jpg")
        cv2.imwrite(output_path, best_frame)  
        frame_count += 1
        detected_frames.clear()  

cap.release()  
