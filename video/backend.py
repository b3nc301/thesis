from yolov5 import detect

detect.run(weights="best.pt",source="../demos/vid.mp4", view_img="True")