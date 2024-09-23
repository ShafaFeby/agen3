#!/usr/bin/env python3

import socket
import rospy
from std_msgs.msg import String

ip = "255.255.255.255"
port = 3838

mode = 0

state = 0

game_states = [
    "initial", "ready", "set", "play", "finish"
]

client = None
current_state = 0

# def talker():
#     global current_state

#     client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) # UDP
#     client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
#     client.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

#     client.bind((ip, port))
#     print("socket created")

#     pub = rospy.Publisher('gamestate', String, queue_size=10)
#     rospy.init_node('gamestatelistener', anonymous=True)
#     rate = rospy.Rate(10) # 10hz
#     while not rospy.is_shutdown():

#         data, addr = client.recvfrom(1024)
#         # print(type(data[0]))

#         if(data[0] == 82):
#             if(data[1] == 71):
#                 if(data[2] == 109):
#                     if(data[3] == 101):
#                         for i in range(10):
#                             if(data[3+i] == 6):
#                                 if(current_state != data[4+i]):
#                                     current_state = data[4+i]
#                                     if current_state > 4: break
#                                     state_str = game_states[current_state]
#                                     rospy.loginfo(state_str)
#                                     pub.publish(state_str)
#                                     break
                        
#         rate.sleep()

#     client.close()

# if __name__ == '__main__':
#     try:
#         talker()
        
#     except rospy.ROSInterruptException:
#         client.close()
#         pass

# client.close()