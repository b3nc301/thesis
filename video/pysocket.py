import asyncio
import websockets
import subprocess
import json
import time

originalData = {"src":"rtsp://localhost/",
"conf":"0.7",
"max":"10",
"min":"1.5"}
proc = None

data=originalData
async def setup(websocket, path):
    global proc
    global data
    mess=await websocket.recv()
    if(isinstance(mess,str)):
        if ";" in mess:
            message=mess.split(";")
            receivedData={}
            if message[0] == "start":
                try:
                    receivedData=json.loads(message[1])
                except:
                    await websocket.send(json.dumps({"status":"fail","proc":"start"}))
                data = {**data, **receivedData}
                prep = data["src"]  + data["conf"] + data["max"] + data["min"]
                if proc is None:
                    proc = subprocess.Popen(["python3", "backend.py", prep])
                elif proc.poll() is not None:
                    proc = subprocess.Popen(["python3", "backend.py", prep])
                else:
                    await websocket.send(json.dumps({"status":"fail","proc":"start"}))
                time.sleep(3)
                if proc.poll() is not None:
                    await websocket.send(json.dumps({"status":"fail","proc":"start"}))
                else: 
                    await websocket.send(json.dumps({"status":"ok","proc":"start"}))


            elif message[0] == "change":
                try:
                    receivedData=json.loads(message[1])
                except:
                    await websocket.send("ERROR! JSON cannot be parsed!")
                data = {**data, **receivedData}
                if proc is not None:
                    if proc.poll() is None:
                        prep = data["src"] + data["conf"]  + data["max"] + data["min"]
                        proc.terminate()
                        roc = subprocess.Popen(["python3", "backend.py", prep])
                        if proc.poll() is not None:
                            await websocket.send(json.dumps({"status":"fail","proc":"change"}))
                        else: 
                            await websocket.send(json.dumps({"status":"ok","proc":"change"}))
                await websocket.send(json.dumps({"status":"ok","proc":"change"}))
            elif message[0] == "status":
                await websocket.send(json.dumps(data))
            elif message[0] == "stop":

                if proc is None:
                    await websocket.send(json.dumps({"status":"fail","proc":"stop"}))
                elif proc.poll() is not None:
                    await websocket.send(json.dumps({"status":"fail","proc":"stop"}))
                else:
                    proc.terminate()
                    time.sleep(3)

                    if proc.poll() is not None:
                        await websocket.send(json.dumps({"status":"ok","proc":"stop"}))
                    else: 
                        await websocket.send(json.dumps({"status":"fail","proc":"stop"}))
            else:
                await websocket.send("ERROR! Not a valid primary modifier key!")
        else:
            await websocket.send("ERROR! Not a properly formatted message!")
    else:
        await websocket.send("ERROR! Message is not formatted as a string!")

async def main():
    async with websockets.serve(setup, "localhost", 8765):
        print("running")
        await asyncio.Future()  # run forever

asyncio.run(main())