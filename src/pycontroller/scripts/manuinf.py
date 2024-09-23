#!/usr/bin/env python3

import os
import time
import argparse

import cv2
print(cv2.__version__)
# import pycuda.autoinit

import math
import numpy as np

from utils.yolo_classes import get_cls_dict
# from utils.camera import add_camera_args, Camera, camera_args
from utils.display import show_fps
from utils.visualization import BBoxVisualization
from utils.mjpeg import MjpegServer
# from utils.yolo_with_plugins import TrtYOLO


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d wbmode=0 awblock=true gainrange='8 8' ispdigitalgainrange='4 4' exposuretimerange='2000000 2000000' aelock=true !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


# control vars
res_x = 640.0
res_y = 480.0

max_lost = 40
goto_zero = 0.01    
stop_zone = 0.20

offset_in_x = 0.13
offset_in_y = 0.15

lost_count = 0
ball_lock = False


# YOLO PARAMS

category_num = 2
model = "yolov3-tiny-low"
letter_box = False
mjpeg_port = 8090

cam = None
trt_yolo = None
conf_th = 0.3 
vis = None
mjpeg_server = None 


class Tracking:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.m = 100.0

    def set(self, tar):
        self.x = tar.x
        self.y = tar.y
        self.m = tar.m
tic = 0
def detect(track_ball):

    global tic
    global lost_count
    global cam
    global trt_yolo
    global vis
    global ball_lock

    tic += 1

    _,img = cam.read()
    if img is None:
        return None
    # boxes, confs, clss = trt_yolo.detect(img, conf_th)
    # json = vis.get_json(boxes, confs, clss)
    # img = vis.draw_bboxes(img, boxes, confs, clss)

    # dns = cv2.fastNlMeansDenoisingColored(img,None,40,40,7,21)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 196, 56])
    upper = np.array([12, 255, 255]) 

    mask = cv2.inRange(hsv, lower, upper)

    result = cv2.bitwise_and(img, img, mask=mask)
    gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # dns_gray = cv2.fastNlMeansDenoising(gray_image,None,10,7,21)

    ret,thresh = cv2.threshold(gray_image,1,255,cv2.THRESH_BINARY)
    nonZero = cv2.countNonZero(thresh)
    

    found = False
    
    cX = 0
    CY = 0

    if(nonZero > 50):
        found = True
        M = cv2.moments(thresh)

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # if(tic%2 == 0):

        cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
        # cv2.putText(img, "Ball", (cX - 25, cY - 25),cv2.FONT+_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # if(tic%2 == 0):
        # mjpeg_server.send_img(img)
        # print(nonZero)
            # name = 'dts18-'+str(counter)+'.jpg'
            # cv2.imwrite(name, img)


    # ============      Control       ============
    closest = Tracking()
    # closest_c = 0.0
    # d_closest = 100.0

    # for i_box, i_class, conf_c in zip(boxes, clss, confs):
    #     if i_class == 0:
    #         tracked = Tracking() 
    #         size_x = (i_box[2] - i_box[0]) /res_x
    #         size_y = (i_box[3] - i_box[1]) /res_y
    #         tracked.x = ((size_x / 2.0 + i_box[0]) / res_x) - 0.5 + offset_in_x
    #         tracked.y = ((size_y / 2.0 + i_box[1]) / res_y) - 0.5 + offset_in_y
    #         tracked.m = math.sqrt((size_x*size_x) + (size_y*size_y))

    #         d_x = tracked.x - track_ball.x
    #         d_y = tracked.y - track_ball.y

    #         dist = math.sqrt((d_x*d_x) + (d_y*d_y))

    #         if dist < d_closest:
    #             d_closest = dist
    #             closest_c = conf_c
    #             closest = tracked
    #             found = True

    if found:
        closest.x = cX/res_x*2 - 1
        closest.y = cY/res_y*2 - 1
        closest.y = -closest.y
        track_ball.set(closest)
        # print(str(closest.x) + " : " + str(closest.y))
        lost_count = 0
        ball_lock = True
    else:
        lost_count += 1

        if track_ball.x < -stop_zone:
            track_ball.x += goto_zero
        if track_ball.x > stop_zone:
            track_ball.x -= goto_zero
        if track_ball.y < -stop_zone:
            track_ball.y += goto_zero
        if track_ball.y > stop_zone:
            track_ball.y -= goto_zero    

        if lost_count >= max_lost:
            ball_lock = False
            lost_count = max_lost
            track_ball.x = 0.0
            track_ball.y = 0.0


def startInference():

    global cam
    global mjpeg_server
    global vis
    global trt_yolo

    cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cam.isOpened():
        return -1
    
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    global model
    global category_num
    global letter_box
    global mjpeg_port

    # cls_dict = get_cls_dict(category_num)
    # vis = BBoxVisualization(cls_dict)
    # trt_yolo = TrtYOLO(model, category_num, letter_box)

    mjpeg_server = MjpegServer(port=mjpeg_port)
    print('MJPEG server started...')
    return 0

def inferenceLoop(track_ball):
    while(True):
        detect(track_ball)


def shutdown():
    global mjpeg_server
    global cam
    mjpeg_server.shutdown()
    cam.release()