# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

#rendszer importok
import time
import datetime
import subprocess
import math



import argparse
import os
import signal
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn


#mappaváltás az importokhoz
path = './yolov5'

os.chdir(path)

FILE = Path(path).resolve()
ROOT = FILE.parents[0] # YOLOv5 root directory
if str(ROOT) not in sys.path:
   sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

#yolov5 module importálások

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, check_imshow, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, scale_coords, xyxy2xywh, LOGGER
from utils.plots import Annotator, colors
from utils.torch_utils import  select_device

#vissza a jelenlegi mappába
os.chdir("../")

#mysql connector definíció
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="laravel",
  password="laravel",
  database="laravel"
)
proc1 = None
proc2 = None
videoID = "0"


mycursor = mydb.cursor()
#mysql connector definíció vége

#RTMP stream URL
rtmp_url = "rtmp://localhost:1935/live/stream"

#alap futó függvüny
@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        stream_img=True,  # stream and show results
        save_video=True,  # save results to video
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=True,  # use FP16 half-precision inference
        ):
    global proc1
    global proc2
    #forrásvizsgálat, hogy videó-e vagy stream
    source = str(source)
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Mappák
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # Inkrementális mappalétrehozás
    save_dir.mkdir(parents=True, exist_ok=True)  # Mappa létrehozása

    # CPU/Videókártya inicializálása
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Modell betöltése
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '']

    check_suffix(w, suffixes)  # A megfelelő kiterjesztés ellenőrzése
    stride, names = 64, [f'class{i}' for i in range(1000)]  # Alapértékek beállítása

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # Osztályok neveinek eltárolása
    if half:
        model.half()  # to FP16
    model2 = attempt_load("./yolov5m.pt", map_location=device)
    stride2 = int(model2.stride.max())  # model stride
    names2 = model2.module.names if hasattr(model, 'module') else model2.names  # Osztályok neveinek eltárolása
    if half:
        model2.half()  # to FP16
    imgsz = check_img_size(imgsz, s=stride2)  # check image size

    # Dataloader, itt tölti be a videókat/streamet képkockákba
    if webcam:
        stream_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    #sql
    if save_video:
        global videoID
        start_time = datetime.datetime.now() # a videó kezdeti időpontja
        start_localtime = time.localtime()
        sql = "INSERT INTO videos (videoName,videoDate,videoURL,videoAvailable) VALUES (%s,%s,%s, %s)" # SQL insert kérés
        val = (time.strftime("%Y%m%d%H%M%S", start_localtime)+source, time.strftime("%Y-%m-\%d %H:%M:%S", time.localtime()),'not ready', 0)
        mycursor.execute(sql, val) # SQL kérés végrehajtása
        mydb.commit() # SQL kérés lezárása
        videoID=mycursor.lastrowid;# Az éppen mentendő videó ID-ja
        save_path = str(save_dir) +"/"+time.strftime("%Y%m%d%H%M%S", start_localtime)+".mp4"
    #változók beállítása
    predictionList = np.zeros((4,10,5)) #az észleléseket tároló vektor
    #(4 képkockán max 10 észlelés, minden észleléshez tartozik 4 adat, x,y koordináta egy prediction ID, ami összeköti az észleléseket és egy frame number, hogy hány képkockán keresztül tartott az esemény)
    frameCounter = 0 #képkocka számláló 
    predictionID=0
    frameNum = 0 
    # a hálózat alkalmazása a képkockákon
    if device.type != 'cpu':
        model2(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once

    for path, img, im0s, vid_cap, s in dataset:

        frameCounter += 1

        #A képek betöltése és átalakítása
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # A hálózatba betáplált képkocka végigfuttatása
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(img, augment=augment, visualize=visualize)[0]
        pred2 = model2(img, augment=augment, visualize=visualize)[0]

        # Nem-maximum vágás (aktiváció)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=10)
        pred2 = non_max_suppression(pred2, conf_thres, iou_thres, 0, agnostic_nms, max_det=10)
        # A megtalált becslések feldolgozása
        for i, det in enumerate(pred):  # per image
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            #save_path = str(save_dir / p.name)
            s += '%gx%g ' % img.shape[2:]  # kiíró string(debug)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))



            #Detektálások feldolgozása 
            if len(det):
                # A befoglaló geometriák átméretezése img_sizeról im0 sizera
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Az eredmények kiírása(debug)
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                #észlelések elmozgatása 1 képkockával az észleléseket tartalmazó vektorban
                if(frameCounter >= 4):
                    predictionList[0]=predictionList[1]
                    predictionList[1]=predictionList[2]
                    predictionList[2]=predictionList[3]
                    predictionList[3]=np.zeros((10,5))




                # Az eredmények összegyűjtése
                detectionCounter = 0
                for *xyxy, conf, cls in reversed(det):
                    #Ha még nincs 4 frame akkor csak feltöltés következik
                    if frameCounter<4 :
                        if detectionCounter<10 :
                            predictionList[frameCounter][detectionCounter][0] = xyxy[0]
                            predictionList[frameCounter][detectionCounter][1] = xyxy[1]
                            predictionList[frameCounter][detectionCounter][2] = -1
                            predictionList[frameCounter][detectionCounter][3] = 1
                            predictionList[frameCounter][detectionCounter][4] = int(cls)
                    #Ha van már 4 frame akkor csak az utolsó képkocka adatainak feltöltése zajlik
                    else:
                        if detectionCounter<10 :
                            predictionList[3][detectionCounter][0] = xyxy[0]
                            predictionList[3][detectionCounter][1] = xyxy[1]
                            predictionList[3][detectionCounter][2] = -1
                            predictionList[3][detectionCounter][3] = 1
                            predictionList[3][detectionCounter][4] = int(cls)
                    detectionCounter+=1 #detektálás számláló növelése(hogy max 10 detektálás legyen)
                    if stream_img:  #(befoglaló geometria képre mentése)
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                distance = 30              

                if frameCounter > 4 :
                    for curr in predictionList[3]:
                        if(curr[0]!=0 and curr[1]!=0 ):
                            for j in range(0,3):
                                for k in predictionList[j]:
                                    if(math.sqrt(math.pow((curr[0]-k[0]),2) + math.pow((curr[1]-k[1]),2))<=distance and (curr[4] != 1)):
                                        curr[3]=k[3]+1
                                        if(k[2] == -1):
                                            k[2] = predictionID
                                            curr[2] = predictionID
                                            predictionID+=1
                                        else:
                                            curr[2]=k[2]
                        if(curr[3]>3):
                            sql = "UPDATE events SET classid=%s,frames=%s,videoID=%s,level=%s,predID=%s WHERE videoID = %s AND PredID= %s"
                            val = (curr[4],curr[3], videoID,1, curr[2],videoID,curr[2])
                            mycursor.execute(sql, val)
                            mydb.commit()
                        elif(curr[3]==3):
                            sql = "INSERT INTO events (classid,time,frames,videoID,level,predID) VALUES (%s, %s,%s, %s,%s, %s)"
                            val = (curr[4], time.strftime("%Y-%m-\%d %H:%M:%S", time.localtime()),curr[3], videoID,1, curr[2])
                            mycursor.execute(sql, val)
                            mydb.commit()                                  
            centers = []
            #emberek megtalálása
            det2 = pred2[0]
            if len(det2):
                # A befoglaló geometriák átméretezése img_sizeról im0 sizera
                det2[:, :4] = scale_coords(img.shape[2:], det2[:, :4], im0.shape).round()

                # Az eredmények kiírása(debug)
                for c in det2[:, -1].unique():
                    n = (det2[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names2[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Az eredmények kiítása
                for *xyxy, conf, cls in reversed(det2):
                    mask=False
                    predID=None
                    if frameCounter>4 :
                        for k in predictionList[3]:
                            if(xyxy[0]<k[0] and xyxy[2] > k[0] and xyxy[1] < k[1] and xyxy[3] > k[1]):
                                mask = True
                                predID = k[2]
                                frameNum = k[3]

                    center=[(xyxy[0]+xyxy[2])/2, xyxy[3], (0,255,0), mask, predID]
                    centers.append(center)
                    setdist=300


                    if stream_img:  #(befoglaló geometria képre mentése)
                        c = int(cls)  # integer class
                        label2 = None if hide_labels else (names2[c] if hide_conf else f'{names2[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label2, color=colors(c, True))
            violated= []
            for c in centers:
                for c1 in centers:
                    dist = math.sqrt(math.pow((c1[0]-c[0]),2) + math.pow((c1[1]-c[1]),2))
                    if(dist<setdist and c != c1):
                        # távolságon belül vannak, sql művelet, és maszkvizsgálat kell
                        c1[2] = (0,0,255)
                        c[2] = (0,0,255)
                        if c not in violated:
                            violated.append(c)
                        if c1 not in violated:
                            violated.append(c1)

            ''' violatedPrediction = False
            for v in violated:
                for p in predictionList[3]:
                    if v[3] == p[2]:
                        if p[2] != -1:
                            violatedPrediction = True
                            if(p[3]>=3):
                                sql = "UPDATE events SET classid=%s,frames=%s,videoID=%s,level=%s,predID=%s WHERE videoID = %s AND PredID= %s"
                                val = (p[4],p[3], videoID,4, p[2],videoID,p[2])
                                mycursor.execute(sql, val)
                                mydb.commit()
                    if(not violatedPrediction):
            '''

                        

            # Print completed inference(debug only)
            LOGGER.info(f'{s}Done.')



            # Eredmények rajzolása
            im0 = annotator.result()
            for c in centers:
                cv2.circle(im0, (int(c[0]),int(c[1])), 3, c[2], 3)
            #eredmények közvetítése
            if stream_img:
                if proc1 == None:
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    command = ['ffmpeg',
                            '-re',
                        '-y',
                        '-f', 'rawvideo',
                        '-pix_fmt', 'bgr24',
                        '-s', "{}x{}".format(str(w), str(h)),
                        '-r', str(fps),
                        '-i', '-',
                        '-tune', 'zerolatency',
                        '-crf', '18',
                        '-vcodec', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-f', 'flv',
                        rtmp_url]
                    #ffmpeg plugin indítása
                    proc1 = subprocess.Popen(command, stdin=subprocess.PIPE)
                cv2.imshow(str(p), im0)
                proc1.stdin.write(im0.tobytes())
                cv2.waitKey(1)  # vár 1 millisecond
            
            #videó mentése
            if save_video:
                #ha a videómentő nem fut indítsa el
                if proc2 == None:
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    command2 = ['ffmpeg',
                        '-re',
                        '-y',
                        '-f', 'rawvideo',
                        '-pix_fmt', 'bgr24',
                        '-s', "{}x{}".format(str(w), str(h)),
                        '-r', str(fps),
                        '-i', '-',
                        '-tune', 'zerolatency',
                        '-crf', '18',
                        '-vcodec', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        str(save_path)]
                    proc2 = subprocess.Popen(command2, stdin=subprocess.PIPE)
                else: 
                    #videó mentése
                    proc2.stdin.write(im0.tobytes())
            
                time_spent= datetime.datetime.now()-start_time
                #ha legalább 1 órája megy a videó zárja le a fájlt, mentse el SQL-ben, és írjon új videót
                if time_spent.total_seconds() >= 60*60:
                    sql = "UPDATE videos SET videoURL=%s, videoAvailable=%s WHERE id=%s"
                    val = (save_path, 1, videoID)
                    mycursor.execute(sql, val)
                    mydb.commit()
                    #új videó definiálása
                    start_localtime = time.localtime()
                    save_path = str(save_dir) +"/"+time.strftime("%Y%m%d%H%M%S", start_localtime)+".mp4"
                    sql = "INSERT INTO videos (videoName,videoDate,videoURL,videoAvailable) VALUES (%s,%s,%s, %s)" # SQL insert kérés
                    val = (time.strftime("%Y%m%d%H%M%S", start_localtime)+source, time.strftime("%Y-%m-\%d %H:%M:%S", time.localtime()),'not ready', 0)
                    mycursor.execute(sql, val) # SQL kérés végrehajtása
                    mydb.commit() # SQL kérés lezárása
                    videoID=mycursor.lastrowid;# Az éppen mentendő videó ID-ja
                    #videówriter kill
                    proc2.terminate()
                    proc2 = None
                    start_time = datetime.datetime.now() # a videó kezdeti időpontja


