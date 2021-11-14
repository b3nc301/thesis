import asyncio
import websockets
import subprocess


prep = ""

async def echo(websocket, path):
    async for message in websocket:
        await websocket.send(message)


async def setup(websocket, path):
    mess=await websocket.recv()
    message=mess.split(";")
    print (message)
    
    if message[0] == "start":
       global proc
       proc = subprocess.Popen(["python3", "backend.py"])
    elif message[0] == "detection":
        prep = message[1]
        proc.terminate()
        proc = subprocess.Popen(["python3", "backend.py"])
        await websocket.send(message[1])
    elif message[0] == "status":
        await websocket.send(prep)
    elif message[0] == "stop":
        proc.terminate()
    else:
        print ("shit")

async def main():
    async with websockets.serve(setup, "localhost", 8765):
        print("running")
        await asyncio.Future()  # run forever

asyncio.run(main())