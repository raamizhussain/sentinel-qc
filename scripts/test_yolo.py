from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# Load a pretrained YOLOv10 model (downloads automatically, ~6MB)
model = YOLO("yolov10n.pt")

# Point to one defect image from MVTec
image_path = r"C:\Users\User\sentinel\data\mvtec\bottle\test\broken_large\000.png"

# Run detection
results = model(image_path)

# Show the result
results[0].save(filename="output.jpg")
print("Detection done! Check scripts/output.jpg")