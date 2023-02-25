from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("/Users/.../Desktop/bottleTest/best.pt")#edit path according to your setup
model.predict(source="0", show=True, conf=0.5)
