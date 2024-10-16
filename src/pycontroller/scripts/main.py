#!/usr/bin/env python3

import os
import signal
import sys

import rospy
import time
import json

# websocket
import threading
# import inference
import manuinf as inference
import ball_tracking
from configloader import read_walk_balance_conf

from walking import Vector2, Vector2yaw, CONTROL_MODE_HEADLESS, CONTROL_MODE_YAWMODE, Walking
from walk_utils import joints, getWalkParamsDict, setWalkParamsConvert


# import asyncio
from simple_websocket_server import WebSocketServer, WebSocket

from std_msgs.msg import String, Int32
from robotis_controller_msgs.msg import SyncWriteItem
from robotis_controller_msgs.msg import StatusMsg
# from robotis_controller_msgs.msg import SetJointModule, JointCtrlModule
from op3_walking_module_msgs.msg import WalkingParam
from op3_walking_module_msgs.msg import WalkingCorrection
from sensor_msgs.msg import Imu, JointState
from op3_walking_module_msgs.srv import GetWalkingParam
from op3_action_module_msgs.srv import IsRunning

    
###############################################################################

pubSWI = rospy.Publisher('/robotis/sync_write_item', SyncWriteItem, queue_size=10)
pubBT = rospy.Publisher('/robotis/open_cr/button', String, queue_size=10)
pubEnaMod = rospy.Publisher('/robotis/enable_ctrl_module', String, queue_size=10)
pubWalkCmd = rospy.Publisher('/robotis/walking/command', String, queue_size=10)
pubSetParams = rospy.Publisher('/robotis/walking/set_params', WalkingParam, queue_size=10)
pubWalkCorr = rospy.Publisher('walking_correction', WalkingCorrection, queue_size=10)
pubHeadControl = rospy.Publisher('/robotis/head_control/set_joint_states', JointState, queue_size=1)
pubMotionIndex = rospy.Publisher('/robotis/action/page_num', Int32, queue_size=10)


currentWalkParams = None # ? params
walkParams = None

s_ctrl_modules = None

server = None

robotIsOn = False
walking_module_enabled = False
action_module_enabled = False

action_status = False

walking = Walking()
track_ball = inference.Tracking()

lean = 0

clients = {}

SEND_PARAM_INTERVAL = 0.03
lastSendParamTic = time.time()

isManagerReady = False

module = 0
mode = 1

class WS(WebSocket):
    def handle(self):
        data = json.loads(self.data)
        cmd = data['cmd']

        if cmd == 'torque_on':
            startRobot()
        elif cmd == 'torque_off':
            setDxlTorque()
            send_message(-1, "torque_control", False)
        elif cmd == 'start_walk':
            setWalkCmd("start")
            send_message(-1, "walk_control", True)
        elif cmd == 'stop_walk':
            setWalkCmd("stop")
            send_message(-1, "walk_control", False)
        elif cmd == 'save_walk_params':
            setWalkCmd("save")
            send_message(-1, "walk_params_saved", True)
        elif cmd == 'get_walk_params':
            send_message(-1, "update_walk_params", getWalkParams())
        elif cmd == 'set_walk_params':
            setWalkParams(data['params'])
            send_message(-1, "controller_msg", 'Walk params changed')
        elif cmd == "set_walking":
            if(self.address[1] == walking.control):
                vectorDict = data['params']
                vector = Vector2yaw(vectorDict["x"], vectorDict["y"], vectorDict["yaw"])
                walking.setTarget(vector)
        elif cmd == 'set_walking_offset':
            walking.setWalkingOffset()
            send_message(-1, "controller_msg", 'Walking offset changed')
        elif cmd == 'set_walking_conf':
            walking.setWalkingConf(data['params'])
            send_message(-1, "controller_msg", 'Walking configuration changed')
        elif cmd == "set_control_walking":
            if data['params'] == 1:
                walking.control = self.address[1]
                send_message(-1, "control_override", self.address)
            else:
                walking.control = None
                send_message(-1, "control_override", -1)
        elif cmd == 'get_walking_conf':
            send_message(-1, "update_walking_conf", walking.getWalkingConf())
        elif cmd == 'get_walking':
            send_message(self.address[1], "update_walking", walking.getWalkingConf())
        elif cmd == 'gyro_init':
            init_gyro()
            send_message(-1, "controller_msg", "Init gyro success")
        elif cmd == 'head_direct':
            headControlDirect(data['params'])
        elif cmd == 'track_head_control':
            headControlHandle(data['params'])
        elif cmd == 'edit_head_pid':
            headPIDHandle(data['params'])
        elif cmd == "set_mode":
            setModeHandle(data['params'])
        elif cmd == 'resetk4':
            handleResetK4()

            # sendParameterized("edit_head_pid", JSON.stringify({"px" : 1.0,
            #                                        "ix" : 1.0,
            #                                        "dx" : 1.0,
            #                                        "sx" : 1.0,
            #                                        "py" : 1.0,
            #                                        "iy" : 1.0,
            #                                        "dy" : 1.0,
            #                                        "sy" : 1.0,}));

    def connected(self):
        print(self.address, 'connected')
        clientID = self.address[1]
        clients.update({clientID: self})
        send_message(clientID, "device_connected", self.address)
        send_message(clientID, "torque_control", robotIsOn)

    def handle_close(self):
        clients.pop(self.address)
        print(self.address, 'closed')
        send_message(-1, "device_disconnected", self.address)

def send_message(id, cmd, params):
    resp = {
        "cmd" : cmd,
        "params" : params
    }
    respJson = json.dumps(resp)
    if(id >= 0):
        clients[id].send_message(respJson)
    else:
        for client in clients.values():
            client.send_message(respJson)



def forever_ws(num):
    global server

    server = WebSocketServer('', 8077, WS)
    print("Websocket is running...")
    server.serve_forever()


t1 = threading.Thread(target=forever_ws, args=(10,))

def sendHeadControl(pitch, yaw):
    js = JointState()
    js.name.append("head_tilt")
    js.name.append("head_pan")
    js.position.append(pitch)
    js.position.append(-yaw)
    pubHeadControl.publish(js)

def sendWithWalkParams():
    if(walking.control == None): return
    
    global walkParams
    global pubSetParams
    walkParams.x_move_amplitude = walking.vectorCurrent.y
    if(walking.turn_mode == CONTROL_MODE_HEADLESS):
        walkParams.y_move_amplitude = walking.vectorCurrent.x
        walkParams.angle_move_amplitude = 0.0
    else:
        walkParams.angle_move_amplitude = walking.vectorCurrent.yaw
        walkParams.y_move_amplitude = 0.0
    
    pubSetParams.publish(walkParams)
    send_message(-1, "update_walking", walking.getWalkingCurrent())

def enableWalk():
    pubEnaMod.publish("walking_module")
    pubEnaMod.publish("head_control_module")

def enableAction():

    rospy.loginfo("enabbling action...")
    pubEnaMod.publish("action_module")
    pubEnaMod.publish("head_control_module")
    

def setDxlTorque(): # list comprehension
    global robotIsOn

    isTorqueOn = False

    if robotIsOn == False:
        return
    robotIsOn = False

    syncwrite_msg = SyncWriteItem()
    syncwrite_msg.item_name = "torque_enable"
    for joint_name in joints:
        if((not isTorqueOn) and (joint_name == "head_pan" or joint_name == "head_tilt")):
            continue
        syncwrite_msg.joint_name.append(joint_name)
        syncwrite_msg.value.append(isTorqueOn) 

    pubSWI.publish(syncwrite_msg)

def initGyro():
    syncwrite_msg = SyncWriteItem()
    syncwrite_msg.item_name = "imu_control"
    syncwrite_msg.joint_name.append("open-cr")
    syncwrite_msg.value.append(8)

    pubSWI.publish(syncwrite_msg)

def startRobot():
    global robotIsOn
    if robotIsOn:
        return
    robotIsOn = True
    pubBT.publish("user_long")

def headControlDirect(data):
    pitch = data["pitch"]
    yaw = data["yaw"]
    sendHeadControl(pitch, yaw)

def headControlHandle(data):
    if(data["enabled"] == True):
        ball_tracking.isEnabled = True
    elif(data["enabled"] == False):
        ball_tracking.isEnabled = False

def headPIDHandle(data):
    print(data)
    ball_tracking.pid_x.tunings = (data["px"], data["ix"], data["dx"])
    ball_tracking.out_scale_x = data["sx"]
    ball_tracking.pid_y.tunings = (data["py"], data["iy"], data["dy"])
    ball_tracking.out_scale_y = data["sy"]

def setModeHandle(data):
    global mode
    global module
    print(data)
    if(data["mdl"] == "walk"):
        enableWalk()
        module == 1
    elif(data["mdl"] == "act"):
        enableAction()
        module = 0 
    elif(data["mod"] == "k1"):
        enableAction()
        mode = 0
    elif(data["mod"] == "k2"):
        enableAction()
        mode = 1
    elif(data["mod"] == "k3"):
        enableAction()
        mode = 2
    elif(data["mod"] == "k4"):
        enableAction()
        mode = 3
        handleResetK4()
    
    send_message(-1, "set_mode", {"mdl":module,"mod":mode})

def handleResetK4():
    global lean
    lean = 0
    stopAction()
    playAction(79)
def handleResetK1():
    global lean
    lean = 0
    playAction(100)


def setWalkCmd(walkCmd):
    sendWalkCorrectionConf()
    if walkCmd == "start" or walkCmd == "stop" or walkCmd == "balance on" or walkCmd == "balance off" or walkCmd == "save":
        pubWalkCmd.publish(walkCmd)

def setWalkParams(param):
    global walkParams

    setWalkParamsConvert(walkParams, param)
    
    pubSetParams.publish(walkParams)

def getWalkParams():
    global currentWalkParams
    global walkParams
    rospy.wait_for_service('/robotis/walking/get_params')
    try:
        getParams = rospy.ServiceProxy('/robotis/walking/get_params', GetWalkingParam)
        resp = getParams()
        params = resp.parameters
        walkParams = params
        paramsDict = getWalkParamsDict(params)
        currentWalkParams = paramsDict
        return paramsDict
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def isActionRunning():
    rospy.wait_for_service('/robotis/action/is_running')
    try:
        getParams = rospy.ServiceProxy('/robotis/action/is_running', IsRunning)
        resp = getParams()
        global action_status
        action_status = resp.is_running
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def playAction(index):
    pubMotionIndex.publish(index)

def stopAction():
    pubMotionIndex.publish(-2)

def init_gyro():
    init_gyro_msg = SyncWriteItem()
    init_gyro_msg.item_name = "imu_control"
    init_gyro_msg.joint_name.append("open-cr")
    init_gyro_msg.value.append(8)
    pubSWI.publish(init_gyro_msg)

imu = Imu()

def handleImu(imu_msg_):
    global imu
    imu = imu_msg_

def handleBalanceMonitor(msg):
    send_message(-1, "balance_monitor", msg.data)

def onFinishInitPose():
    enableWalk()

def sendWalkCorrectionConf():
    wc = WalkingCorrection()

    wc.wb_p_gain = read_walk_balance_conf("Balance", "wb_p_gain")
    wc.wb_i_gain = read_walk_balance_conf("Balance", "wb_i_gain")
    wc.wb_d_gain = read_walk_balance_conf("Balance", "wb_d_gain")
    wc.zero_pitch_offset = read_walk_balance_conf("Balance", "zero_pitch_offset")
    wc.pitch_offset_multiplier = read_walk_balance_conf("Balance", "pitch_offset_multiplier")
    wc.x_offset_multiplier = read_walk_balance_conf("Balance", "x_offset_multiplier")

    pubWalkCorr.publish(wc)

def handleStatusMsg(statusMsg):
    print(statusMsg.status_msg)
    if(statusMsg.status_msg == "Walking Enabled"):
        init_gyro()
        print("init gyro...")

    if(statusMsg.status_msg == "Action Enabled"): 
        rospy.loginfo("Action enabled")
        global action_module_enabled
        action_module_enabled = True

    if(statusMsg.status_msg == "Finish Init Pose"):
        # enableWalk()
        enableAction()
        send_message(-1, "torque_control", True)
        global isManagerReady
        isManagerReady = True

    statusDict = {
        'type':statusMsg.type,
        'module_name':statusMsg.module_name,
        'status_msg':statusMsg.status_msg
    }

    send_message(-1, 'update_status', statusDict)

def handleButton2Msg(data):
    print("BUTON")
    print(data.data)




def main():


    global lastSendParamTic
    global track_ball
    global lean 
    global s_ctrl_modules

    t1.start()
    rospy.init_node('main', anonymous=True)

    #rospy.Subscriber("/robotis/open_cr/imu", Imu, handleImu)
    rospy.Subscriber("/robotis/status", StatusMsg, handleStatusMsg)
    rospy.Subscriber('/robotis/open_cr/button2', String, handleButton2Msg)
    # rospy.Subscriber("balance_monitor", String, handleBalanceMonitor)

    rospy.loginfo("Waiting manager...")
    global isManagerReady
    while(isManagerReady==False): 
        time.sleep(1)

    print("controller runnning")
    rospy.loginfo("Wait for manager complete")
    getWalkParams()
    startRobot()

    time.sleep(5)

    if(inference.startInference() < 0):
        rospy.loginfo("Inference Start ERROR")
        raise SystemExit('ERROR: failed to open camera!')
    else:
        rospy.loginfo("Inference Started")

    val = -0.9
    dir = 0.01

    enableAction()
    time.sleep(5)
    if(mode == 3):
        handleResetK4()
    elif(mode == 0):
        handleResetK1()

    while not rospy.is_shutdown():
        toc = time.time()
        delta_t = toc - lastSendParamTic
        if(delta_t > SEND_PARAM_INTERVAL):
            lastSendParamTic = toc

            walking.stepToTargetVel()
            sendWithWalkParams()

        inference.detect(track_ball)


        if(inference.ball_lock == False):
            ball_tracking.pitch = 0.1
            ball_tracking.search()
        else:
            ball_tracking.track(track_ball)

        # print(str(ball_tracking.pitch) + " : " + str(ball_tracking.yaw))
        if(module == 0):
            #mode 0
            if(mode == 0):
                if((lean == 0 and ball_tracking.pitch < -0.1)):
                    if(lean == 0):
                        isActionRunning()
                        if(action_status is not True):
                            if(ball_tracking.yaw < 0):
                                lean = 2
                                playAction(2)
                        
                            elif(ball_tracking.yaw > 0):
                                lean = 3
                                playAction(3)
                elif(lean != 0 and ball_tracking.pitch > 0.1):
                    isActionRunning()
                    if(action_status is not True):
                        if(lean == 2):
                            playAction(6)
                            lean = 0
                        elif(lean == 3):
                            playAction(5)
                            lean = 0
                elif(lean == 0):
                    isActionRunning()
                    if(action_status is not True):
                        playAction(100)

            #mode k2
            if(mode == 1):
                isActionRunning()

                if(ball_tracking.pitch < -0.2 and lean == 0):
                    stopAction()
                    if(ball_tracking.yaw < 0):
                        lean = 2
                        playAction(2)
                
                    elif(ball_tracking.yaw > 0):
                        lean = 3
                        playAction(3)

                elif(lean != 0 and ball_tracking.pitch > 0.1):
                    if(action_status is not True):
                        if(lean == 2):
                            playAction(6)
                            lean = 0
                        elif(lean == 3):
                            playAction(5)
                            lean = 0

                elif(action_status is not True and lean == 0):
                    playAction(100)

            #mode k3
            if(mode == 2):
                if((lean == 0 and ball_tracking.pitch < 0.01)):
                    if(lean == 0):
                        isActionRunning()
                        if(action_status is not True):
                            lean = 8
                            playAction(8)
                elif(lean != 0 and ball_tracking.pitch > 0.1):
                    isActionRunning()
                    if(action_status is not True): 
                        playAction(7)
                        lean = 0
            
            #mode k4
            if(mode == 3):
                isActionRunning()
                if((lean == 0 and ball_tracking.pitch < -0.2)):
                    stopAction()
                    if(lean == 0):
                        if(action_status is not True):
                            if(ball_tracking.yaw < 0):
                                lean = 68
                                playAction(68)
                                print("68")
                        
                            elif(ball_tracking.yaw > 0):
                                lean = 67
                                print("67")
                                playAction(67)

                elif(lean == 0):
                    if(action_status is not True):
                        playAction(79)

            #mode k5
            if(mode == 4):
                isActionRunning()

                if(ball_tracking.yaw < 0.1 and ball_tracking.yaw > -0.1 and ball_tracking.pitch > -0.7):
                    stopAction()
                    playAction(89)

                if(ball_tracking.pitch < -0.2 and lean == 0):
                    stopAction()
                    if(ball_tracking.yaw < 0):
                        lean = 2
                        playAction(2)
                
                    elif(ball_tracking.yaw > 0):
                        lean = 3
                        playAction(3)

                

                elif(lean != 0 and ball_tracking.pitch > 0.1):
                    if(action_status is not True):
                        if(lean == 2):
                            playAction(6)
                            lean = 0
                        elif(lean == 3):
                            playAction(5)
                            lean = 0

                elif(action_status is not True and lean == 0):
                    playAction(100)
                        
        


        sendHeadControl(ball_tracking.pitch, ball_tracking.yaw)
        
        # val+=dir
        # if(val >= 0.9):
        #     dir = -0.1
        # if(val <= -0.9):
        #     dir = 0.1
            


def shutdown():
    global server
    inference.shutdown()
    server.close()
    t1.join()
    sys.exit()

def close_sig_handler(signal, frame):
    shutdown()

signal.signal(signal.SIGINT, close_sig_handler)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        shutdown()
