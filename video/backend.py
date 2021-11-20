import time

#basic yolo detector
import detect

#detect.run(weights="best.pt", source="../demos/vid.mp4", view_img="True", imgsz=[640,640])

detect.run(weights="./best.pt", source=0, view_img="True", imgsz=[640,480], save_txt="true")


#