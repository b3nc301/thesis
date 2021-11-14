import asyncio
import websockets
import subprocess

data = {"src":"rtsp://localhost/",
"wg":"best.pt",
"frames":"3",
"imgsz":"[640,640]",
"conf":"0.7",
"nms":"0.5",
"max":"10",
"min":"1.5"}

async def setup(websocket, path):
    mess=await websocket.recv()
    if mess.contains(";"):
        message=mess.split(";")        
        if message[0] == "start":
            global proc
            prep = data["src"] + data["wg"] + data["frames"] + data["imgsz"] + data["conf"] + data["nms"] + data["max"] + data["min"]
            proc = subprocess.Popen(["python3", "backend.py", prep])
        elif message[0] == "detection":
            prep = data["src"] + data["wg"] + data["frames"] + data["imgsz"] + data["conf"] + data["nms"] + data["max"] + data["min"]
            proc.terminate()
            proc = subprocess.Popen(["python3", "backend.py", prep])
            await websocket.send(message[1])
        elif message[0] == "status":
            await websocket.send(data)
        elif message[0] == "stop":
            proc.terminate()
    else:
        await websocket.send("ERROR")

async def main():
    async with websockets.serve(setup, "localhost", 8765):
        print("running")
        await asyncio.Future()  # run forever

asyncio.run(main())