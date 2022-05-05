# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

#rendszer importok
import time
import datetime
import subprocess
import math

import yaml


import argparse
import os
import signal
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from detect import getDistance, compute_perspective_transform, compute_point_perspective_transformation

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
def getCoord(matrix,p1,d_w,d_h,im):
    transformed_downoids_p1 = compute_point_perspective_transformation(matrix,(p1[0],p1[1]))
    h = transformed_downoids_p1[0][0]
    w = transformed_downoids_p1[0][1]
    if h>0 and w>0 :
        im1= cv2.circle(im,(int(h),int(w)), 1, (0,255,0),10)
    else:
        im1=im
    #h = abs(p2[1]-p1[1])
    #w = abs(p2[0]-p1[0])
    dis_w = float((w/d_w)*100)
    dis_h = float((h/d_h)*100)
    return dis_w, dis_h, im1, transformed_downoids_p1[0].copy()
#alap futó függvüny
@torch.no_grad()
def run(weights=ROOT / './yolov5m.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=[640,480],  # inference size (pixels)
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
        min=150
        ):
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
    imgsz = check_img_size(imgsz, s=stride)  # check image size

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


    #madártávlati konfig beolvasása
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    width_og, height_og = 0,0
    corner_points = []
    for section in cfg:
        corner_points.append(cfg["image_parameters"]["p1"])
        corner_points.append(cfg["image_parameters"]["p2"])
        corner_points.append(cfg["image_parameters"]["p3"])
        corner_points.append(cfg["image_parameters"]["p4"])
        width_og = int(cfg["image_parameters"]["width_og"])
        height_og = int(cfg["image_parameters"]["height_og"])
        size_frame = cfg["image_parameters"]["size_frame"]
        baseCoord = cfg["image_parameters"]["pz"]
        xCoord = cfg["image_parameters"]["px"]
        yCoord = cfg["image_parameters"]["py"]
    im1=cv2.imread("static_frame_from_video.jpg")
    matrix,imgOutput = compute_perspective_transform(corner_points,width_og,height_og,im1)
    img = imgOutput
    height,width,_ = imgOutput.shape
    blank_image = np.zeros((height,width,3), np.uint8)
    height = blank_image.shape[0]
    width = blank_image.shape[1]
    dim = (width, height)
    bird_view_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #változók beállítása
    prevViolated = list() #a korábbi mozgásokat tároló lista
    # a hálózat alkalmazása a képkockákon
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    prevCenters = list() # középpontok

    for path, img, im0s, vid_cap, s in dataset:
        #A képek betöltése és átalakítása
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # A hálózatba betáplált képkocka végigfuttatása
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred2 = model(img, augment=augment, visualize=visualize)[0]

        # Nem-maximum vágás (aktiváció)
        pred2 = non_max_suppression(pred2, conf_thres, iou_thres, 0, agnostic_nms, max_det=10)
        # A megtalált becslések feldolgozása
        for i, det in enumerate(pred2):  # per image
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
            centers = list() # középpontok
            #emberek megtalálása és középpontok eltárolása
            det2 = pred2[0]
            if len(det2):
                # A befoglaló geometriák átméretezése img_sizeról im0 sizera
                det2[:, :4] = scale_coords(img.shape[2:], det2[:, :4], im0.shape).round()
                # Az eredmények kiírása(debug)
                for c in det2[:, -1].unique():
                    n = (det2[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                for *xyxy, conf, cls in reversed(det2):
                    mask=False
                    print(xyxy)
                    center=[int((xyxy[0]+xyxy[2])/2),int(xyxy[3]), path]
                    centers.append(center)
                    annotator.box_label(xyxy, "AAAAAAAAAAAAA", color=colors(c, True))
                print("person")
                print(centers)        
            
            #print(centers)
            #perspektíva transzformáció
            matrix,imgOutput = compute_perspective_transform(corner_points,width_og,height_og,im0)
            img1 = imgOutput
            height,width,_ = imgOutput.shape
            blank_image = np.zeros((height,width,3), np.uint8)
            height = blank_image.shape[0]
            width = blank_image.shape[1]
            dim = (width, height)
            bird_view_img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
            #cv2.imshow("img", bird_view_img)


            #relative távolság számolása
            dist_point = compute_point_perspective_transformation(matrix,(baseCoord,xCoord,yCoord))
            distance_w = np.sqrt((dist_point[0][0] - dist_point[1][0]) ** 2 + (dist_point[0][1] - dist_point[1][1]) ** 2)
            distance_h = np.sqrt((dist_point[0][0] - dist_point[2][0]) ** 2 + (dist_point[0][1] - dist_point[2][1]) ** 2)
            for c in centers:
                print("Coordinates:")
                d_w,d_h,bird_view_img,td=getCoord(matrix, c, distance_w, distance_h,bird_view_img)
                cv2.circle(im0,(int(c[0]),int(c[1])), 1, (255,0,0),10)
                print(d_w,d_h)
            print(td)
            cv2.circle(bird_view_img1,(int(td[0]),int(td[1])), 1, (255,0,0),5)
            cv2.circle(bird_view_img1,(0,0), 1, (0,255,0),5)
            cv2.imshow("img1", bird_view_img1)
            cv2.imshow("img2", im0)
            cv2.waitKey(1000000)  # vár 1 millisecond
            prevCenters = centers.copy()
            print('\n')



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='dir')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--min', type=float, default=150, help='Minimum distance')
    opt = parser.parse_args()
    #print_args(FILE.stem, opt)
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
