#!/usr/bin/env python3

import cv2 
import torch
import time
import numpy as np

import math
import time

from utils.utilsNAS import letterbox_v8, postprocess_v8, blob_v8, TRTModule
from utils.utilsv7 import letterbox_v7, init_tensorrt_v7, postprocess_v7

NAS_str = '/home/name/samba/models/nas_shoes1.trt'
v8_str = '/home/name/samba/models/v71.trt'
v7_str = '/home/name/samba/models/v81.trt'

device = torch.device('cuda:0')

CLASSES = ['ball', 'circle', 'float', 'shoe' ]
COLORS = [(255,255,255), (40,200,40), (200,40,40), (0,40,0)]

performances = []
performances2 = []

class NAS():
    def __init__(self):
        self.Engine = TRTModule(NAS_str, device)
        self.H, self.W = self.Engine.inp_info[0].shape[-2:]

        self.Engine.set_desired(['num_dets', 'det_boxes', 'det_scores', 'det_classes'])
    
    def infer(self, bgr):
        start2 = time.perf_counter()

        draw = bgr.copy()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob_v8(rgb)
        
        tensor = torch.asarray(tensor, device=device)
        start = time.perf_counter()
        data = self.Engine(tensor)
        performances.append(time.perf_counter()-start)

        bboxes, scores, labels = postprocess_v8(data)
        if bboxes.numel() == 0:
            return [], bgr

        packs = []

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            color = COLORS[cls_id]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            packs.append((bbox[0],bbox[1],bbox[2],bbox[3],cls_id,score))

        performances2.append(time.perf_counter()-start2)
        return packs, draw
    
class v7():
    def __init__(self):
        self.bindings = None
        self.binding_addrs = None
        self.context = None

        self.bindings, self.binding_addrs, self.context = init_tensorrt_v7(v7_str, device)
    
    def infer(self, bgr):
        start2 = time.perf_counter()
        image = bgr.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, ratio, dwdh = letterbox_v7(image)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im = torch.from_numpy(im).to(device)
        im/=255

        self.binding_addrs['images'] = int(im.data_ptr())

        start = time.perf_counter()
        self.context.execute_v2(list(self.binding_addrs.values()))
        
        performances.append(time.perf_counter()-start)

        nums = self.bindings['num_dets'].data
        boxes = self.bindings['det_boxes'].data
        classes = self.bindings['det_classes'].data
        scores = self.bindings['det_scores'].data

        boxes = boxes[0,:nums[0][0]]
        classes = classes[0,:nums[0][0]]
        scores = scores[0,:nums[0][0]]

        packs = []

        for (bbox, score, label) in zip(boxes, scores, classes):
            bbox = postprocess_v7(bbox,ratio,dwdh).round().int().tolist()
            cls_id = int(label)
            packs.append((bbox[0],bbox[1],bbox[2],bbox[3],cls_id,score))
            cv2.rectangle(bgr,(bbox[0],bbox[1]),(bbox[2],bbox[3]), COLORS[cls_id], 2)

        performances2.append(time.perf_counter()-start2)
        
        return packs, bgr
    
class v8():
    def __init__(self):
        self.Engine = TRTModule(v8_str, device)
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
        performances.append(time.perf_counter()-start)

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

        performances2.append(time.perf_counter() - start2)
        return packs, draw
    
def get_perf():
    total_p = 0.0
    for p in performances:
        total_p += p
    infer_only = total_p/len(performances)

    total_p = 0.0
    for p in performances2:
        total_p += p
    pipeline = total_p/len(performances2)

    return infer_only, pipeline

def get_yolo(yolo_str):
    if yolo_str == "nas":
        return NAS()
    elif yolo_str == "v8":
        return v8()
    elif yolo_str == "v7":
        return v7()
    else: return NAS()

def draw_border(canvas, color):
    return cv2.rectangle(canvas, (2,2), (638, 478), color, 3)