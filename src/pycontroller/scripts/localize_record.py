#!/usr/bin/env python
import time
import asyncio
from websockets.server import serve

file_object = open('recorder.txt', 'a')
file_object.write('hello\n')
file_object.write('hello2\n')


def current_milli_time():
    return round(time.time() * 1000)


async def echo(websocket):
    async for message in websocket:
        print("in")
        print(f"jet: {str(current_milli_time())}")
        print(f"rem: {message}")
        await websocket.send(str(current_milli_time()))

async def main():
    async with serve(echo, "0.0.0.0", 8765):
        await asyncio.Future()  # run forever

asyncio.run(main())