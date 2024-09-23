#        1        |
#-1      |       1|
#        |        |
#_______0,0_______|
#       /|\
#        |
#        0 deg

import math

pos_x = 0.0
pos_y = 0.0
yaw = 0.0

target = -1

span_to_rad = 0.1
grad_to_alpha = 0.1
theta_to_beta = 0.23

dist_ab = 1.2
frame_h = 1
margin_h = 0.01
pixel_to_unit = 1

def locallize_from_dist(a, b):
    global pos_x
    global pos_y

    print("disty:")
    print(a)
    print(b)

    t = abs(frame_h - b + margin_h)*pixel_to_unit
    b = abs(frame_h - a + margin_h)*pixel_to_unit
    a = t
    c = dist_ab

    0.549, 0.0675, 1.2

    print(str(a)+" | "+str(b)+" | "+str(c))

    pos_x = (((b*b) + (c*c) - (a*a)) / (2*b*c)) * b
    print("sqrt:")
    pos_y = math.sqrt(abs(b*b - pos_x*pos_x))


def locallize_from_goal(goal):
    global pos_x
    global pos_y
    global yaw

    rad = goal.span * span_to_rad
    alpha = goal.grad * grad_to_alpha
    beta = goal.theta * theta_to_beta

    pos_x = goal.span * 1/goal.grad
    
def walk(speed, turn_rate):
    
    
    print()

def turn(turn_rate):
    yaw += turn_rate