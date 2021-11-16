import time

#basic yolo detector
import detect

#detect.run(weights="best.pt", source="../demos/vid.mp4", view_img="True", imgsz=[640,640])

detect.run(weights="./best.pt", source="../../yolov5/DC.mp4", view_img="True", imgsz=[640,640], save_txt="true")


#