from datetime import datetime
# file = open('recorder.csv', 'a')

#MAKE SURE CONNECTION IS STABLE WHEN START
import cv2
import numpy as np
import time

import asyncio
# from websockets.sync.client import connect

# now = datetime.now()

# current_time = now.strftime("%H:%M:%S:%MS")
# print("Current Time =", current_time)


def current_milli_time():
    return round(time.time() * 1000)

print(current_milli_time())


time_error = 0
count_timer = 20

# def hello():
#     global time_error
#     with connect("ws://192.168.0.99:8765") as websocket:
#         time_diffs = 0
#         for i in range(count_timer):
#             websocket.send(str(current_milli_time()))
#             message = websocket.recv()
#             print("in")
#             rem = current_milli_time()
#             jet = int(message)
#             print(f"rem:{rem}")
#             print(f"jet:{jet}")

#             diff_t = jet-rem
#             time_diffs+=diff_t

#         time_error = time_diffs/count_timer



# hello()
##
min_h = 10
max_h = 20
min_s = 190
max_s = 255
min_v = 90
max_v = 255


def on_min_h(val): 
    global min_h
    min_h = val
def on_min_s(val): 
    global min_s
    min_s = val
def on_min_v(val): 
    global min_v
    min_v = val
def on_max_h(val): 
    global max_h
    max_h = val
def on_max_s(val): 
    global max_s
    max_s = val
def on_max_v(val): 
    global max_v
    max_v = val


cv2.namedWindow("track")
cv2.createTrackbar("min_h", "track" , 1, 255, on_min_h)
cv2.createTrackbar("max_h", "track" , 1, 255, on_max_h)
cv2.createTrackbar("min_s", "track" , 1, 255, on_min_s)
cv2.createTrackbar("max_s", "track" , 1, 255, on_max_s)
cv2.createTrackbar("min_v", "track" , 1, 255, on_min_v)
cv2.createTrackbar("max_v", "track" , 1, 255, on_max_v)




##
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

res_w = 1920
res_h = 1080


vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

half_w = 1920/2
half_h = 1080/2

while(True):

    ret, frame = vid.read()
    if not ret: break

    blur = cv2.medianBlur(frame, 5)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    

    lower = np.array([min_h, min_s, min_v])
    upper = np.array([max_h, max_s, max_v]) 

    lower1 = np.array([1, 171, 184])
    upper1 = np.array([255, 198, 252])  

    lower2 = np.array([100, 150, 130])
    upper2 = np.array([120, 255, 255]) 

    mask = cv2.inRange(hsv, lower, upper)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    # M = cv2.moments(mask)
    # M1 = cv2.moments(mask1)
    # M2 = cv2.moments(mask2)

    # if M["m00"] == 0: continue
    # if M1["m00"] == 0: continue
    # if M2["m00"] == 0: continue

    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])

    # cX1 = int(M1["m10"] / M1["m00"])
    # cY1 = int(M1["m01"] / M1["m00"])

    # cX2 = int(M2["m10"] / M2["m00"])
    # cY2 = int(M2["m01"] / M2["m00"])
    
    # cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1) 
    # cv2.circle(frame, (cX1, cY1), 5, (255, 255, 25), -1) 
    # cv2.circle(frame, (cX2, cY2), 5, (255, 25, 255), -1) 

    # print(str(current_milli_time()))
    # print(',')
    # print(str((cX-half_h)-cY*0.1))
    # print(',')
    # print(str(cY))
    # print(',')
    # print(str(cX1))
    # print(',')
    # print(str(cY1))
    # print(',')
    # print(str(cX2))
    # print(',')
    # print(str(cY2))

    cv2.imshow('track1', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()