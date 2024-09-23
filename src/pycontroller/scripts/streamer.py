import asyncio
import json
import os
import fractions
import cv2
import queue
import threading
from typing import Tuple
import time

from av import VideoFrame

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

frame_width=480
frame_height=320

pcs = set()

video = None

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    display_width=frame_width,
    display_height=frame_height,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d wbmode=0 awblock=true gainrange='8 8' ispdigitalgainrange='4 4' exposuretimerange='2000000 2000000' aelock=true !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 900
VIDEO_PTIME = 1 / 5  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)

# bufferless VideoCapture


class VideoCapture:

    def __init__(self):
        self.record = True
        
        self.frameOut = None
        self.frameIn = None
        self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

        # RECORDER
        if self.record:
            latest = 0

            dir_path = r'/home/name/records'
            for path in os.scandir(dir_path):
                if path.is_file():
                    current = int(path.name[0:-4])
                    if current > latest:
                        latest = current

            self.out = cv2.VideoWriter('/home/name/records/'+str(latest+1)+'.avi',cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame_width,frame_height))

        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read_in(self):
        frame = self.q.get()
        if self.record:
            self.out.write(frame)
        return frame
    def store_out(self, frame):
        self.frameIn = frame
        # if self.record:
        #     self.out.write(frame)
    def read_out(self):
        return self.frameIn

    def release(self):
        self.out.release()
        self.cap.release()


class VideoOpencvTrack(VideoStreamTrack):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap

    _start: float
    _timestamp: int

    def get_frame_size(self):
        height, width, _ = self.cap.read_in().shape
        print("frame size : ", height, ", ", width)
        return (width, height)

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            print("ERROR track unready!")
            return None

        if hasattr(self, "_timestamp"):
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, VIDEO_TIME_BASE

    async def recv(self):
        img = self.cap.read_out()

        if img is None:
            print("emptyframe")
            return None

        pts, time_base = await self.next_timestamp()

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base

        return new_frame


async def offer(request):
    print("processing answer...")
    global video
    params = request
    print(params["sdp"])
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    if video:
        pc.addTrack(video)

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    print("remote description has been set")

    return {"cmd" : "stream_answer", "sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
