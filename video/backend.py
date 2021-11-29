import time

#basic yolo detector
import detect

#detect.run(weights="best.pt", source="rtsp://192.168.1.207:8080/h264_ulaw.sdp", stream_img="True", imgsz=[640,640])

detect.run(weights="./best.pt", source="../demos/IMG_2566.MP4", stream_img="True", imgsz=[640,420],)


#