import time

#basic yolo detector
from yolov5 import detect

#detect.run(weights="best.pt", source="../demos/vid.mp4", view_img="True", imgsz=[640,640])

detect.run(weights="best.pt", source="../demos/vid.mp4", view_img="True", imgsz=[640,640])

time.sleep(20)