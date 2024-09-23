#!/usr/bin/env python3

# -1 to 1
# center is 0,0
# right, up = +, -

import cv2 
import torch
import random
import time
import numpy as np
import tensorrt as trt
from collections import OrderedDict,namedtuple
from yolov10.utilsNAS import letterbox_v8, postprocess_v8, blob_v8, TRTModule

import math

import streamer as streamer

#DEF YOLO START
weight_file = '/home/name/models/best1.trt'
device = torch.device('cuda:0')

bindings = None
binding_addrs = None
context = None
YOLO = None

color_ball = (0, 0, 255)
color_goal = (255, 0, 0)
COLORS = [(255,255,255), (40,200,40), (200,40,40)]


class v8():
    def __init__(self):
        self.Engine = TRTModule(weight_file, device)
        self.H, self.W = self.Engine.inp_info[0].shape[-2:]

        self.Engine.set_desired(['num', 'boxes', 'scores', 'classes'])
    
    def infer(self, bgr):
        start2 = time.perf_counter()

        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox_v8(bgr, (self.W, self.H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob_v8(rgb) /255
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        start = time.perf_counter()
        # inference
        data = self.Engine(tensor)

        bboxes, scores, labels = postprocess_v8(data)
        if bboxes.numel() == 0:
            return [], bgr
        bboxes -= dwdh
        bboxes /= ratio

        packs = []

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            color = COLORS[cls_id]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            packs.append((bbox[0],bbox[1],bbox[2],bbox[3],cls_id,score))

        return packs, draw

# names = ['bola':0, 'gawang':1]
#DEF YOLO END

videocap = None

res = None

lost_count = 0
ball_lock = False

max_lost = 30
goto_zero_x = 0.01
goto_zero_y = 0.01
stop_zone = 0.20

class Tracking:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.m = 100.0

    def set(self, tar):
        self.x = tar.x
        self.y = tar.y
        self.m = tar.m

class Detection:
    def __init__(self):
        self.balls = []
        self.goals = []

def detect(track_ball, dets):

    global videocap

    img = videocap.read_in()

    dets, draw = YOLO.infer(img)

    is_found = False
    d_closest = 1000.0
    closest = Tracking()

    detected_goals = []

    for det in dets:
        cl = det[4]
        box = det[:4]

        if cl == 1:
            x = int((box[0] + box[2]) / 2)
            y = int((box[1] + box[3]) / 2)
            detected_goals.append([x, max(box[3], box[1])])
        else:
            x = int((box[0] + box[2]) / 2)
            y = int((box[1] + box[3]) / 2)

            x = (x / res[0] - 0.5) * 2
            y = (y / res[1] - 0.5) * 2
            
            d_x = x - track_ball.x
            d_y = y - track_ball.y

            dist = math.sqrt((d_x*d_x) + (d_y*d_y))

            if dist < d_closest:
                d_closest = dist
                closest.x = x
                closest.y = y
                is_found = True

    # goal processing
    dets.goals += detected_goals

    # ball processing
    global ball_lock
    global lost_count

    if is_found:
        track_ball.x = closest.x
        track_ball.y = closest.y
        lost_count = 0
        ball_lock = True
    else:
        lost_count += 1

        if track_ball.x < -stop_zone:
            track_ball.x += goto_zero_x
        if track_ball.x > stop_zone:
            track_ball.x -= goto_zero_x
        if track_ball.y < -stop_zone:
            track_ball.y += goto_zero_y
        if track_ball.y > stop_zone:
            track_ball.y -= goto_zero_y    

        if lost_count >= max_lost:
            lost_count = max_lost
            ball_lock = False
            track_ball.x = 0.0
            track_ball.y = 0.0

    videocap.store_out(draw)


def startInference():

    global videocap
    global YOLO
    global res

    YOLO = v8
    videocap = streamer.VideoCapture()
    track = streamer.VideoOpencvTrack(videocap)
    streamer.video = track

    res = track.get_frame_size()

    print('inference is running...')
    return 0

def inferenceLoop(track_ball):
    while(True):
        detect(track_ball)


def shutdown():
    videocap.release()