#!/usr/bin/env python
import json
import time

def current_milli_time():
    return round(time.time() * 1000)

def append(data_str, uncluster):
    global file_object
    curent_time = str(current_milli_time())
    print(str(current_milli_time())+","+data_str+'\n')
    with open('/home/name/recorder.txt', 'a') as file_object:
        data = curent_time+","+data_str
        file_object.write(data+"\n")
        print("enddataset1")
        file_object.close()
    with open('/home/name/recorderdet.txt', 'a') as file_object:

        for item in uncluster:
            data = curent_time+","+json.dumps(item)
            file_object.write(data+"\n")
        print("enddataset2")
        file_object.close()
        
    
