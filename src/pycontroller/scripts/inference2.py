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


import math

import streamer as streamer

#DEF YOLO START
w = '/home/name/models/tc.trt'
device = torch.device('cuda:0')

bindings = None
binding_addrs = None
context = None

color_ball = (0, 0, 255)
color_goal = (255, 0, 0)


# Infer TensorRT Engine
def init_tensorrt():
    global bindings
    global binding_addrs
    global context

    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")
    with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    bindings = OrderedDict()
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        shape = tuple(model.get_binding_shape(index))
        data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    context = model.create_execution_context()

        # warmup for 10 times
    for _ in range(10):
        tmp = torch.randn(1,3,480,320).to(device)
        binding_addrs['images'] = int(tmp.data_ptr())
        context.execute_v2(list(binding_addrs.values()))

def letterbox(im, new_shape=(480, 320), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def postprocess(boxes,r,dwdh):
    dwdh = torch.tensor(dwdh*2).to(boxes.device)
    boxes -= dwdh
    boxes /= r
    return boxes

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

    global bindings
    global binding_addrs
    global context

    global videocap

    img = videocap.read_in()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)

    im = torch.from_numpy(im).to(device)
    im/=255

    start = time.perf_counter()
    binding_addrs['images'] = int(im.data_ptr())
    context.execute_v2(list(binding_addrs.values()))
    exec_cost = time.perf_counter()-start

    nums = bindings['num_dets'].data
    boxes = bindings['det_boxes'].data
    boxes = bindings['det_boxes'].data
    classes = bindings['det_classes'].data

    boxes = boxes[0,:nums[0][0]]
    classes = classes[0,:nums[0][0]]

    is_found = False
    d_closest = 1000.0
    closest = Tracking()

    detected_goals = []

    for box,cl in zip(boxes,classes):
        box = postprocess(box,ratio,dwdh).round().int().tolist()

        if cl == 1:
            cv2.rectangle(img,tuple(box[:2]),tuple(box[2:]),color_goal,2)
            x = int((box[0] + box[2]) / 2)
            y = int((box[1] + box[3]) / 2)
            detected_goals.append([x, max(box[3], box[1])])
        else:
            x = int((box[0] + box[2]) / 2)
            y = int((box[1] + box[3]) / 2)
            cv2.circle(img, (x,y), 4, color_ball, -1)
            
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

        if lost_count > max_lost:
            lost_count = max_lost
            ball_lock = False
            track_ball.x = 0.0
            track_ball.y = 0.0

    videocap.store_out(img)


def startInference():

    global videocap

    global res

    init_tensorrt()
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