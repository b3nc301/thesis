# YOLOv5 🚀 by Ultralytics, GPL-3.0 license


#mysql connector
import mysql.connector
import subprocess


mydb = mysql.connector.connect(
  host="localhost",
  user="laravel",
  password="laravel",
  database="laravel"
)

rtmp_url = "rtmp://127.0.0.1:1935/stream"


mycursor = mydb.cursor()
#mysql connector end

import time
import datetime

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn


#changepath for imports
path = './yolov5'

os.chdir(path)


FILE = Path(path).resolve()
ROOT = FILE.parents[0] # YOLOv5 root directory
if str(ROOT) not in sys.path:
   sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



#module imports

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, strip_optimizer, xyxy2xywh, LOGGER
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

#fall back to current script directory
os.chdir("../")



@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=True,  # use FP16 half-precision inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    '''
    if webcam:  # video
        fps, width, height = 30, imgsz[0], imgsz[0]
    else:  # stream
        cap=cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = 640
    height = 640
    command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(width, height),
           '-r', str(fps),
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           #'-flvflags', 'no_duration_filesize',
           rtmp_url]
    #p = subprocess.Popen(command, stdin=subprocess.PIPE)
    '''


    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    #custom
    start_time = datetime.datetime.now()
    sql = "INSERT INTO videos (videoName,videoDate,videoURL,videoAvailable) VALUES (%s,%s,%s, %s)"
    val = (time.strftime("%Y%m%d%H%M%S", time.localtime())+source, time.strftime("%Y-%m-\%d %H:%M:%S", time.localtime()),'not ready', 0)
    mycursor.execute(sql, val)
    mydb.commit()
    videoID=mycursor.lastrowid;
    predictionList = np.zeros((4,10,4))
    frameCounter = 0
    predictionID=0
    frameNum = 0
    #end custom
    for path, img, im0s, vid_cap, s in dataset:
        frameCounter += 1
        if(frameCounter >= 4):
            predictionList[0]=predictionList[1]
            predictionList[1]=predictionList[2]
            predictionList[2]=predictionList[3]
            predictionList[3]=np.zeros((10,4))
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(img, augment=augment, visualize=visualize)[0]

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)








        # Process predictions
        for i, det in enumerate(pred):  # per image
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            #Detection processing 
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                detectionCounter = 0
                for *xyxy, conf, cls in reversed(det):

                    if save_txt and int(cls)!=1 :  # Write to file
                        predictionID += 1
                        detectionCounter+=1
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        clss = cls.clone().detach().view(1).tolist()
                        current_time = time.localtime()
                        distance=0.11
                        predID=predictionID

                        frameRepeatCounter=0
                        if frameCounter > 4 :
                            for j in range(0,3):
                                for k in predictionList[j]:
                                    if(k[0]!=0 and k[1]!=0):
                                        if(abs(k[0]-xywh[0])<=distance and abs(k[1]-xywh[1])<=distance):
                                            frameRepeatCounter+=1
                                            predID = k[2]
                                            frameNum=k[3]+1
                                            print(frameRepeatCounter)
                        if frameCounter<4 :
                            if detectionCounter<10 :
                                predictionList[frameCounter][detectionCounter][0] = xywh[0]
                                predictionList[frameCounter][detectionCounter][1] = xywh[1]
                                predictionList[frameCounter][detectionCounter][2] = predID
                                predictionList[frameCounter][detectionCounter][3] = frameNum
                        else:
                            if detectionCounter<10 :
                                predictionList[3][detectionCounter][0] = xywh[0]
                                predictionList[3][detectionCounter][1] = xywh[1]
                                predictionList[3][detectionCounter][2] = predID
                                predictionList[3][detectionCounter][3] = frameNum

                        if(frameRepeatCounter>=1):
                            predictionID-=1
                        if(frameRepeatCounter >=3):
                            print("KUTYA", + predID)
                            if frameNum>3:
                                sql = "UPDATE events SET classid=%s,time=%s,frames=%s,videoID=%s,level=%s,predID=%s WHERE videoID = %s AND PredID= %s"
                                val = (int(clss[0]), time.strftime("%Y-%m-\%d %H:%M:%S", current_time),frameNum, videoID,0, predID,videoID,predID)
                                mycursor.execute(sql, val)
                                mydb.commit()
                            else:
                                sql = "INSERT INTO events (classid,time,frames,videoID,level,predID) VALUES (%s, %s,%s, %s,%s, %s)"
                                val = (int(clss[0]), time.strftime("%Y-%m-\%d %H:%M:%S", current_time),frameNum, videoID,0, predID)
                                mycursor.execute(sql, val)
                                mydb.commit()


                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print completed inference(debug only)
            LOGGER.info(f'{s}Done.')
            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                #p.stdin.write(img)
                cv2.waitKey(1)  # 1 millisecond
            # Save results
            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                vid_writer[i] = cv2.VideoWriter(save_path + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(im0)

        time_spent= datetime.datetime.now()-start_time
        if time_spent.total_seconds() < -1:
            sql = "UPDATE videos SET videoURL=%s, videoAvailable=%s WHERE id=%s"
            val = (save_path+".webm", 1, videoID)
            mycursor.execute(sql, val)
            mydb.commit()
            vid_writer[i].release()
            os.system("ffmpeg -i "+ save_path + ".mp4 -c:v libvpx-vp9 -crf 30 -b:v 0 "+ save_path+".webm")
            break
