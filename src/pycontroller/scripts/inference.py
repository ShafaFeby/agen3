#!/usr/bin/env python3

import os
import time
import argparse

import cv2
print(cv2.__version__)
import pycuda.autoinit

import math

from utils.yolo_classes import get_cls_dict
# from utils.camera import add_camera_args, Camera, camera_args
from utils.display import show_fps
from utils.visualization import BBoxVisualization
from utils.mjpeg import MjpegServer
from utils.yolo_with_plugins import TrtYOLO


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=360,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d wbmode=0 awblock=true gainrange='8 8' ispdigitalgainrange='4 4' exposuretimerange='2000000 2000000' aelock=false !"
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

max_lost = 120
goto_zero = 0.01    
stop_zone = 0.20

offset_in_x = 0.13
offset_in_y = 0.15

lost_count = 0

ball_lock = False

# YOLO PARAMS

category_num = 2
model = "yolov3-tiny_last"
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
        self.found = False

    def set(self, tar):
        self.x = tar.x
        self.y = tar.y
        self.m = tar.m
        self.found = False

def detect(track_ball):

    global tic
    global lost_count
    global cam
    global trt_yolo
    global vis
    global ball_lock

    _,img = cam.read()
    if img is None:
        return None
    boxes, confs, clss = trt_yolo.detect(img, conf_th)

    # json = vis.get_json(boxes, confs, clss)

    img = vis.draw_bboxes(img, boxes, confs, clss)

    mjpeg_server.send_img(img)

    # ============      Control       ============
    closest = Tracking()
    closest_c = 0.0
    d_closest = 100.0

    found = False

    for i_box, i_class, conf_c in zip(boxes, clss, confs):
        if i_class == 0:
            tracked = Tracking()
            size_x = (i_box[2] - i_box[0]) /res_x
            size_y = (i_box[3] - i_box[1]) /res_y
            tracked.x = ((size_x / 2.0 + i_box[0]) / res_x) - 0.5 + offset_in_x
            tracked.y = ((size_y / 2.0 + i_box[1]) / res_y) - 0.5 + offset_in_y
            tracked.m = math.sqrt((size_x*size_x) + (size_y*size_y))

            d_x = tracked.x - track_ball.x
            d_y = tracked.y - track_ball.y

            dist = math.sqrt((d_x*d_x) + (d_y*d_y))

            if dist < d_closest:
                d_closest = dist
                closest_c = conf_c
                closest = tracked
                found = True

    if found:
        track_ball.set(closest)
        lost_count = 0
        ball_lock = True
        print("detext")
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

        if lost_count > max_lost:
            lost_count = max_lost
            ball_lock = False
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

    global model
    global category_num
    global letter_box
    global mjpeg_port

    cls_dict = get_cls_dict(category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(model, category_num, letter_box)

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