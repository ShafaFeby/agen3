# STRIKER DADAKAN


# find ball
# chase ball, while do localization (regular goal check) when 
# catch ball, move ball towards goal
# shoot

# convention:
# PRE - > ING -> mas juga

from walking import Vector2yaw
import time
import numpy as np
import math

# repeat

MEMULAI = 0

JALAN_DITEMPAT = 1
JALAN_DITEMPAT_2 = 2
JALAN_DITEMPAT_3 = 3
JALAN = 4

BERHENTI_JALAN = 5
AKTIFKAN_MOTION = 6
MOTION = 7
BERHENTI_MOTION = 8

PAUSE = 9

AKTIFKAN_MOTION_2 = 10

BERHENTI_JALAN_2 = 11

SELESAI_JALAN = 12

SELESAI_MOTION = 13

AKTIFKAN_MOTION_BACK = 14

MOTION_BACK = 15

SIAP2 = 16

ACTIONWALK_MODULE = 17

PAST = 18







states_dict = {
    0 : "MEMULAI",
    1 : "JALAN_DITEMPAT",
    2 : "JALAN_DITEMPAT_2",
    3 : "JALAN_DITEMPAT_3",
    4 : "JALAN",
    5 : "BERHENTI_JALAN",
    6 : "AKTIFKAN_MOTION",
    7 : "MOTION",
    8 : "BERHENTI_MOTION",
    9 : "PAUSE",
    10 : "AKTIFKAN_MOTION_2",
    11 : "BERHENTI_JALAN_2",
    12 : "SELESAI_JALAN",
    13 : "SELESAI_MOTION",
    14 : "AKTIFKAN_MOTION_BACK",
    15 : "MOTION_BACK",
    16 : "SIAP2",
    17 : "ACTIONWALK_MODULE",
    18 : "PAST"
   
}

state = PAUSE 

yaw_dead_area = 0.15

yaw_init = -0.75
last_yaw = 0.0
yaw = yaw_init

gt = None
infer = None
bt = None
walk = None

pubMotionIndex = None
isActionRunning = None
pubEnaMod = None
pubEnableOffset = None

enabled = True
goal_align_time = 0.0
start_align_turn = 0

gt_on_ball_search_last = 0
gt_on_ball_search_interval = 10
gt_on_ball_search_dead_yaw = 0.2
gt_on_ball_search_last_head_pos = [0,0]

setwalkparams = None
setwalkcmd = None

interval_checking = 10

goal_align_post_interval = 10
goal_align_post_start = 0
goal_align_angle_start = 0
goal_align_angle_time = 0

show_head_angle = False

timed_start = 0
timed_delay = 0

initialized = False

actionEnabled = False
ready_time = 15
play_delay = 4

turn_yaw_max_rate = 0.65

time_play_start = 0
max_time_from_play = 1000

pitch_ball_tar = -0.67
yaw_ball_tar = -0.14
ball_tar_deviation = 0.1
yaw_ball_dev_multipler = 1
pitch_ball_dev_multipler = 1

ball_search_loss = 0
max_ball_search_loss = 100

odo_10min_dev = 0.0
odo_deviation = 0.0
time_odo_start = 0

odo_min_max = 1.5
grad_to_yaw_gain = 1.0

show_ypr_counter = 0

z_amp_turning = 0.03
z_amp_normal = 0.026

yaw_compe_ball_align = 0.1
time_multi_goal_align_yaw = 4.5
time_multi_goal_align = 5
yaw_x_turning_max = 1.5

enable_goal_det = True
enable_ball_align = False

set_yaw_compe_start_time = 0.0
set_yaw_compe_start = 0.0

ypr = np.array([.0,.0,.0])
ypr_offset = np.array([.0,.0,.0])

standing_up = False

def clamp(val, _min, _max):
    return max(min(val, _max), _min)

def plus_or_min(val, out):
    if val > 0: return out
    return -out

def set_compe():
    set_state()

def set_ypr(newypr):
    global ypr
    global show_ypr_counter
    global yaw
    ypr[0] = newypr.yaw
    ypr[1] = newypr.pitch
    ypr[2] = newypr.roll
    ypr = ypr - ypr_offset

    if show_ypr_counter >= 30:
        # print("pitch: "+str(ypr[1]))
        # print("yaw: "+str(yaw))
        show_ypr_counter = 0
    show_ypr_counter+=1

    yaw = (ypr[0]/1800*math.pi) + update_odo_dev()

def zero_ypr():
    global ypr_offset
    global time_odo_start
    ypr_offset = ypr + ypr_offset
    time_odo_start = time.time()

def update_odo_dev():
    global odo_deviation
    odo_deviation = (time.time() - time_odo_start) / 600 * odo_10min_dev
    return odo_deviation

def set_state(new_state, _timed_delay = 0):
    global state
    global timed_start
    global timed_delay
    state = new_state
    timed_delay = _timed_delay
    timed_start = time.time()
    print("NEW STATE: "+states_dict[new_state])

def init_action(_pubMotionIndex, _isActionRunning, _pubEnaMod, _pubEnableOffset):
    global pubMotionIndex
    global isActionRunning
    global pubEnaMod
    global pubEnableOffset
    pubMotionIndex = _pubMotionIndex
    isActionRunning = _isActionRunning
    pubEnaMod = _pubEnaMod
    pubEnableOffset = _pubEnableOffset

def enableWalk():
    global actionEnabled
    global standing_up
    standing_up = False
    actionEnabled = False
    pubEnaMod.publish("walking_module")
    pubEnaMod.publish("head_control_module")

def enableAction():
    global actionEnabled
    actionEnabled = True
    pubEnaMod.publish("action_module")
    pubEnaMod.publish("head_control_module")

def playAction(index):
    pubMotionIndex.publish(index)

def stopAction():
    pubMotionIndex.publish(-2)

def init(goal_tracker, inference, ball_tracker, walking, _setwalkparams, _setwalkcmd):
    global gt
    global infer
    global bt
    global walk
    global setwalkparams
    global setwalkcmd
    global initialized

    gt = goal_tracker
    infer = inference
    bt = ball_tracker
    walk = walking
    setwalkcmd = _setwalkcmd
    setwalkparams = _setwalkparams
    initialized = True

def run(time, dets, track_ball, head_control):
    global gt_on_ball_search_last
    global gt_on_ball_search_last_head_pos
    global state
    global goal_align_time
    global start_align_turn
    global goal_align_post_start
    global goal_align_angle_start
    global goal_align_angle_time
    global yaw
    global time_play_start
    global ball_search_loss
    global set_yaw_compe_start_time
    global set_yaw_compe_start
    global odo_10min_dev
    global standing_up

    deltaT = time - timed_start
    timedEnd = deltaT > timed_delay

    head_pitch = head_control[0]
    head_yaw = head_control[1]
    goal = gt.goal
    goal_theta = goal.theta.item(0)
    pitch = ypr[1]

   
    if state == MEMULAI:
        if timedEnd:
            setwalkcmd("start")
            set_state(JALAN_DITEMPAT, 0.1)

    elif state == SIAP2:
        if timedEnd:
            enableWalk()
            set_state(MEMULAI, 0.1)

    elif state == JALAN_DITEMPAT:
        setwalkcmd("start")  
        setwalkparams(["z_move_amplitude", 0.01])
        walk.setTarget()
        # walk.setTarget(0, 0.4, 0.07)
        set_state(JALAN_DITEMPAT_2, 0.7)
        # set_state(JALAN_DITEMPAT_2, 15)
        
    elif state == JALAN_DITEMPAT_2:
        if timedEnd :
            setwalkcmd("start")
            setwalkparams(["z_move_amplitude", 0.014])
            walk.setTarget()
            set_state(JALAN_DITEMPAT_3, 0.7)
            # set_state(PAUSE)

    elif state == JALAN_DITEMPAT_3:
       if timedEnd :
            setwalkcmd("start")
            setwalkparams(["z_move_amplitude", 0.026])
            walk.setTarget(0, 0.5, 0.07)
            set_state(JALAN, 0.7)
    
    elif state == JALAN:
        if timedEnd:
            setwalkcmd("start")
            # walk.setTarget(0.1167, 0.45, 0.07)
            walk.setTarget(0.1167, 0.5, 0.07)
            set_state(BERHENTI_JALAN, 3)

    elif state == SELESAI_JALAN:
        if timedEnd:
            set_state(PAUSE)

    elif state == BERHENTI_JALAN:
        if timedEnd:
            setwalkparams(["z_move_amplitude", 0.022])
            walk.setTarget()
            # setwalkcmd("stop")
            set_state(BERHENTI_JALAN_2, 0.7)

    elif state == BERHENTI_JALAN_2:
        if timedEnd:
            walk.setTarget()
            setwalkcmd("stop")
            # set_state(PAUSE)
            set_state(SELESAI_JALAN, 0.7)


    elif state == AKTIFKAN_MOTION:
        if timedEnd:
            enableAction()
            set_state(AKTIFKAN_MOTION_2, 0.5)

    elif state == MOTION:
        if timedEnd:
            playAction(157)
            set_state(BERHENTI_MOTION, 3)

    elif state == BERHENTI_MOTION:
       if timedEnd:
            stopAction()
            set_state(PAUSE)
    
    elif state == PAUSE:
         if timedEnd:
            return
    
    elif state == AKTIFKAN_MOTION_2:
        if timedEnd:
            set_state(MOTION, 0.5)

    elif state == SELESAI_MOTION:
        if timedEnd:
            enableAction()
            set_state(AKTIFKAN_MOTION_BACK, 1)

    elif state == AKTIFKAN_MOTION_BACK:
        if timedEnd:
            set_state(MOTION_BACK, 1)

    elif state == MOTION_BACK:
        if timedEnd:
            playAction(159)
            set_state(BERHENTI_MOTION, 6)

    elif state == ACTIONWALK_MODULE:
            enableWalk()
            enableAction()
            set_state(PAST, 3)

    elif state == PAST:
        if timedEnd:
            set_state(PAUSE)

    

    
    
def update_odo():
    global yaw
    yaw += 0.05 * walk.vectorCurrent.yaw

    if yaw < -odo_min_max: yaw = (odo_min_max*2) + yaw
    elif yaw > odo_min_max: yaw = (odo_min_max*-2) + yaw