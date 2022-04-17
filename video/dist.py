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



# https://deepnote.com/@deepnote/A-social-distancing-detector-using-a-Tensorflow-object-detection-model-Python-and-OpenCV-KBcEvWejRjGyjy2YnxiP5Q
def compute_perspective_transform(corner_points,width,height,image):
	""" Compute the transformation matrix
	@ corner_points : 4 corner points selected from the image
	@ height, width : size of the image
	return : transformation matrix and the transformed image
	"""
	# Create an array out of the 4 corner points
	corner_points_array = np.float32(corner_points)
	# Create an array with the parameters (the dimensions) required to build the matrix
	img_params = np.float32([[0,0],[width,0],[0,height],[width,height]])
	# Compute and return the transformation matrix
	matrix = cv2.getPerspectiveTransform(corner_points_array,img_params) 
	img_transformed = cv2.warpPerspective(image,matrix,(width,height))
	return matrix,img_transformed


def compute_point_perspective_transformation(matrix,list_downoids):
	""" Apply the perspective transformation to every ground point which have been detected on the main frame.
	@ matrix : the 3x3 matrix 
	@ list_downoids : list that contains the points to transform
	return : list containing all the new points
	"""
	# Compute the new coordinates of our points
	list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
	transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
	# Loop over the points and add them to the list that will be returned
	transformed_points_list = list()
	for i in range(0,transformed_points.shape[0]):
		transformed_points_list.append([transformed_points[i][0][0],transformed_points[i][0][1]])
	return transformed_points_list

def getDistance(matrix,p1,p2,d_w,d_h):
    transformed_downoids_p1 = compute_point_perspective_transformation(matrix,(p1[0],p1[1]))
    transformed_downoids_p2 = compute_point_perspective_transformation(matrix,(p2[0],p2[1]))
    h = abs(p2[1]-p1[1])
    w = abs(p2[0]-p1[0])
    dis_w = float((w/d_w)*200)
    dis_h = float((h/d_h)*200)
    dist = math.sqrt(math.pow((dis_h),2) + math.pow((dis_w),2))
    return dist

#RTMP stream URL
rtmp_url = "rtmp://localhost:1935/live/stream"
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

    #változók beállítása
    predictionList = np.zeros((4,10,5)) #az észleléseket tároló vektor
    prevViolated = list() #a korábbi mozgásokat tároló lista
    violateID=0
    #(4 képkockán max 10 észlelés, minden észleléshez tartozik 4 adat, x,y koordináta egy prediction ID, ami összeköti az észleléseket és egy frame number, hogy hány képkockán keresztül tartott az esemény)
    frameCounter = 0 #képkocka számláló 
    predictionID=0
    frameNum = 0 
    violateFrames=0
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
                    center=[int((xyxy[0]+xyxy[2])/2), int(xyxy[3]), (0,255,0), mask, frameNum,-1]
                    centers.append(center)
                print(centers)
                print(prevCenters)
                print("\n")
            
            
            #print(centers)
            #perspektíva transzformáció
            matrix,imgOutput = compute_perspective_transform(corner_points,width_og,height_og,im0)
            #relative távolság számolása
            dist_point = compute_point_perspective_transformation(matrix,(baseCoord,xCoord,yCoord))
            distance_w = np.sqrt((dist_point[0][0] - dist_point[1][0]) ** 2 + (dist_point[0][1] - dist_point[1][1]) ** 2)
            distance_h = np.sqrt((dist_point[0][0] - dist_point[2][0]) ** 2 + (dist_point[0][1] - dist_point[2][1]) ** 2)
            for c in centers:
                for c1 in prevCenters:
                    print(getDistance(matrix, c, c1, distance_w, distance_h))
                    if(getDistance(matrix, c, c1, distance_w, distance_h)<min and c != c1):
                        c1[2] = (0,0,255)
                        c[2] = (0,0,255)
                        print(getDistance(matrix, c, c1, distance_w, distance_h))
            prevCenters = centers.copy()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='dir')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--min', type=float, default=150, help='Minimum distance')
    opt = parser.parse_args()
    #print_args(FILE.stem, opt)
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
