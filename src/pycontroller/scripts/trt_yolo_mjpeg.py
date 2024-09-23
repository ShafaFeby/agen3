#!/usr/bin/env python

"""trt_yolo_mjpeg.py
MJPEG version of trt_yolo.py.
"""

 
import os
import time
import argparse
import serial
import math
from simple_pid import PID

# websocket
import threading
from simple_websocket_server import WebSocketServer, WebSocket

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import show_fps
from utils.visualization import BBoxVisualization
from utils.mjpeg import MjpegServer
from utils.yolo_with_plugins import TrtYOLO

clients = []


control_angle = 0
control_speed = 0
control_mode = 0
control_angle_time = 0
control_speed_time = 0
control_timeout = 100

out_scale = 0.2
servo_scale = 100
motor_speed = 290 # /500
turn_diff = 300 # /500
turn_max_t = 10
turn_cooldown = 3
move_max_t = 10
move_cooldown = 3

offset_in_x = 0.13
offset_in_y = 0.15

servo_x = 1500
servo_y = 1500

# ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0)
pid_x = PID(2, 0.5, 0.05, setpoint=0)
pid_y = PID(2, 0.5, 0.05, setpoint=0)

pid_x.output_limits = (-1.3, 1.3)
pid_y.output_limits = (-1.3, 1.3)


# print(ser.name)

class Tracking:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.m = 100.0

track_ball = Tracking()

class WS(WebSocket):
    def handle(self):

        global control_angle
        global control_speed
        global control_mode
        global control_angle_time
        global control_speed_time
        global control_timeout

        global out_scale
        global servo_scale
        global motor_speed # /500
        global turn_diff # /500
        global turn_max_t
        global turn_cooldown
        global move_max_t
        global move_cooldown
        
        global offset_in_x
        global offset_in_y

        global servo_x
        global servo_y

        global track_ball
        global pid_x
        global pid_y


        print(self.address, 'receive data: ', self.data)

        pointer_c = 0

        is_decoy = True

        for c in self.data:
            if(c == 'T' or c == 'C' or c == 'M'):
                is_decoy = False
                break
            else:
                pointer_c += 1

        if is_decoy:
            return

        if(self.data[pointer_c] == 'T'):
            if(self.data[pointer_c + 1] == 'P'):
                if(self.data[pointer_c + 2] == 'X'):
                    pid_x.Kp = float(self.data[pointer_c + 3:len(self.data)])
                elif(self.data[pointer_c + 2] == 'Y'):
                    pid_y.Kp = float(self.data[pointer_c + 3:len(self.data)])
                print(self.address, 'Kp x :', pid_y.Kp)
            
            elif(self.data[pointer_c + 1] == 'I'):
                if(self.data[pointer_c + 2] == 'X'):
                    pid_x.Ki = float(self.data[pointer_c + 3:len(self.data)])
                elif(self.data[pointer_c + 2] == 'Y'):
                    pid_y.Ki = float(self.data[pointer_c + 3:len(self.data)])
                print(self.address, 'Ki x :', pid_y.Ki)
            
            elif(self.data[pointer_c + 1] == 'D'):
                if(self.data[pointer_c + 2] == 'X'):
                    pid_x.Kd = float(self.data[pointer_c + 3:len(self.data)])
                elif(self.data[pointer_c + 2] == 'Y'):
                    pid_y.Kd = float(self.data[pointer_c + 3:len(self.data)])
                print(self.address, 'Kd x :', pid_y.Kd)
        
        elif(self.data[pointer_c] == 'C'):
            if(self.data[pointer_c + 1] == 'S'):
                if(self.data[pointer_c + 2] == '1'):
                    control_mode = 1
                elif(self.data[pointer_c + 2] == '0'):
                    control_mode = 0
                print(self.address, 'Control mode :', control_mode)
            elif(self.data[pointer_c + 1] == 'F'):
                if(self.data[pointer_c + 2] == '1'):
                    control_speed = 1
                elif(self.data[pointer_c + 2] == '0'):
                    control_speed = 0
                print(self.address, 'Control speed :', control_speed)
            elif(self.data[pointer_c + 1] == 'B'):
                if(self.data[pointer_c + 2] == '1'):
                    control_speed = -1
                elif(self.data[pointer_c + 2] == '0'):
                    control_speed = 0
                print(self.address, 'Control speed :', control_speed)
            elif(self.data[pointer_c + 1] == 'L'):
                if(self.data[pointer_c + 2] == '1'):
                    control_angle = -1
                elif(self.data[pointer_c + 2] == '0'):
                    control_angle = 0
                print(self.address, 'Control angle :', control_angle)
            elif(self.data[pointer_c + 1] == 'R'):
                if(self.data[pointer_c + 2] == '1'):
                    control_angle = 1
                elif(self.data[pointer_c + 2] == '0'):
                    control_angle = 0
                print(self.address, 'Control angle :', control_angle)
        
        elif(self.data[pointer_c] == 'M'):
            if(self.data[pointer_c + 1] == '1'):
                out_scale = float(self.data[pointer_c + 2:len(self.data)])
                print(self.address, 'out_scale :', out_scale)
            elif(self.data[pointer_c + 1] == '2'):
                servo_scale = float(self.data[pointer_c + 2:len(self.data)])
                print(self.address, 'servo_scale :', servo_scale)
            elif(self.data[pointer_c + 1] == '3'):
                motor_speed = float(self.data[pointer_c + 2:len(self.data)])
                print(self.address, 'motor_speed :', motor_speed)
            elif(self.data[pointer_c + 1] == '4'):
                turn_diff = float(self.data[pointer_c + 2:len(self.data)])
                print(self.address, 'turn_diff :', turn_diff)
            elif(self.data[pointer_c + 1] == '5'):
                turn_max_t = float(self.data[pointer_c + 2:len(self.data)])
                print(self.address, 'turn_max_t :', turn_max_t)
            elif(self.data[pointer_c + 1] == '6'):
                turn_cooldown = float(self.data[pointer_c + 2:len(self.data)])
                print(self.address, 'turn_cooldown :', turn_cooldown)
            elif(self.data[pointer_c + 1] == '7'):
                move_max_t = float(self.data[pointer_c + 2:len(self.data)])
                print(self.address, 'move_max_t :', move_max_t)
            elif(self.data[pointer_c + 1] == '8'):
                move_cooldown = float(self.data[pointer_c + 2:len(self.data)])
                print(self.address, 'move_cooldown :', move_cooldown)
            elif(self.data[pointer_c + 1] == 'Q'):
                offset_in_x = float(self.data[pointer_c + 2:len(self.data)])
                print(self.address, 'offset_in_x :', offset_in_x)
            elif(self.data[pointer_c + 1] == 'W'):
                offset_in_y = float(self.data[pointer_c + 2:len(self.data)])
                print(self.address, 'offset_in_y :', offset_in_y)
            elif(self.data[pointer_c + 1] == 'E'):
                new_limit = float(self.data[pointer_c + 2:len(self.data)])
                pid_x.output_limits = (-new_limit, new_limit)
                print(self.address, 'limit_x :', new_limit)
            elif(self.data[pointer_c + 1] == 'R'):
                new_limit = float(self.data[pointer_c + 2:len(self.data)])
                pid_y.output_limits = (-new_limit, new_limit)
                print(self.address, 'limit_y :', new_limit)
            elif(self.data[pointer_c + 1] == '0'):
                servo_x = 1500
                servo_y = 1500
                print(self.address, 'reset_servo')

        #for client in clients:
        #    client.send_message(self.address[0] + u' - ' + self.data)

    def connected(self):
        print(self.address, 'connected')
        for client in clients:
            client.send_message(self.address[0] + u' - connected')
        clients.append(self)


    def handle_close(self):
        clients.remove(self)
        print(self.address, 'closed')
        for client in clients:
            client.send_message(self.address[0] + u' - disconnected')

def forever_ws(num):
    server = WebSocketServer('', 8077, WS)
    print("Websocket is running...")
    server.serve_forever()


t1 = threading.Thread(target=forever_ws, args=(10,))

def parse_args():
    """Parse input arguments."""
    desc = 'MJPEG version of trt_yolo'
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument(
        '-p', '--mjpeg_port', type=int, default=8090,
        help='MJPEG server port [8090]')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis, mjpeg_server):
    """Continuously capture images from camera and do object detection.
    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
      mjpeg_server
    """
    fps = 0.0
    tic = time.time()

    # control vars
    res_x = 640.0
    res_y = 480.0

    lost_count = 0
    max_lost = 120
    goto_zero = 0
    stop_zone = 0.15

    turn_time = 0
    move_time = 0


    global track_ball
    global pid_x
    global pid_y

    global control_angle
    global control_speed
    global control_mode
    global control_angle_time
    global control_speed_time
    global control_timeout

    global out_scale
    global servo_scale
    global motor_speed # /500
    global turn_diff # /500
    global turn_max_t
    global turn_cooldown
    global move_max_t
    global move_cooldown
    
    global offset_in_x
    global offset_in_y

    global servo_x
    global servo_y
    


    # MAIN LOOP
    while True:
        img = cam.read()
        if img is None:
            break
        boxes, confs, clss = trt_yolo.detect(img, conf_th)

        json = vis.get_json(boxes, confs, clss)
        for client in clients:
            client.send_message(json)

        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        mjpeg_server.send_img(img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)


        # ============      manual        ============
        if control_angle != 0:
            control_angle_time += 1
            if control_angle_time > control_timeout:
                control_angle_time = 0
                control_angle = 0
            
        if control_speed != 0:
            control_speed_time += 1
            if control_speed_time > control_timeout:
                control_speed_time = 0
                control_speed = 0



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
            print(" ")
            print("inference time : ", (toc - tic))
#            print("confidence : ", closest_c)
            track_ball = closest
            lost_count = 0
        else:
            lost_count += 1

#            if track_ball.x < -stop_zone:
#                track_ball.x += goto_zero
#            if track_ball.x > stop_zone:
#                track_ball.x -= goto_zero
#            if track_ball.y < -stop_zone:
#                track_ball.y += goto_zero
#            if track_ball.y > stop_zone:
#                track_ball.y -= goto_zero    

            if lost_count > max_lost:
                lost_count = 0
                track_ball.x = 0.0
                track_ball.y = 0.0
        tic = toc

        # PID

        out_x = pid_x(track_ball.x) * out_scale
        out_y = pid_y(track_ball.y) * out_scale

        servo_x += out_x * servo_scale
        servo_y += out_y * servo_scale

        servo_x = max(min(servo_x, 2000), 1000)
        servo_y = max(min(servo_y, 2000), 1000)

        # motor turn
        motor_l = 1500
        motor_r = 1500

        if control_mode > 0:

            if found:
                turn = 0
                if servo_x < 1300:
                    turn = -1
                    motor_l -= turn_diff
                elif servo_x > 1700:
                    motor_l += turn_diff
                    turn = 1

                if turn != 0:
                    turn_time += 1
                    if turn_time > turn_max_t:
                        turn_time = -turn_cooldown
                        motor_l = 1500

                # motor move
                motor_go = False
                if servo_y > 1000:
                    motor_l += motor_speed
                    motor_r += motor_speed
                    motor_go = True

                if motor_go:
                    move_time += 1
                    if move_time > move_max_t:
                        move_time = -move_cooldown
                        motor_r = 1500

        else:
            if control_angle < 0:
                motor_l += turn_diff
            elif control_angle > 0:
                motor_l -= turn_diff

            if control_speed < 0:
                motor_l -= motor_speed
                motor_r -= motor_speed
            elif control_speed > 0:
                motor_l += motor_speed
                motor_r += motor_speed


        motor_l = max(min(motor_l, 2000), 1000)
        motor_r = max(min(motor_r, 2000), 1000)

                
        res_s = 'C' + str(int(servo_x)) + 'X' + str(int(servo_y)) + 'Y' + str(int(motor_l)) + 'N' + str(int(motor_r)) + 'M'
        # print(res_s)
        res = bytes(res_s, 'utf-8')

        #ser.write(res)


def main():


    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    mjpeg_server = MjpegServer(port=args.mjpeg_port)
    print('MJPEG server started...')
    try:
        loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis,
                        mjpeg_server=mjpeg_server)
    except Exception as e:
        print(e)
    finally:
        mjpeg_server.shutdown()
        cam.release()


if __name__ == '__main__':
    t1.start()
    main()

# python3 trt_yolo_mjpeg.py --usb 0 -m yolov3-tiny_last -p 8090
