import asyncio
import websockets
import subprocess
import json
import time

#változók feltöltése
originalData = {"src":"rtsp://localhost/camera",
"conf":"0.25",
"min":"1.5"}
proc = None

data=originalData
#async függvény
async def setup(websocket, path):
    #global változók definiálása
    global proc
    global data
    #üzenetet várunk
    mess=await websocket.recv()
    #üzenetvizsgálat
    if(isinstance(mess,str)):
        #megnézzük van-e benne pontosvessző
        if ";" in mess:
            #ha van akkor elvágjuk a ;-nél és egy tömbbe helyezzük
            message=mess.split(";")
            receivedData={}
            #ha az üzenet első stringje start
            if message[0] == "start":
                #megpróbáljuk parseolni az üzenetet
                try:
                    receivedData=json.loads(message[1])
                except:
                    await websocket.send(json.dumps({"status":"fail","proc":"start"}))
                #összeállítjuk az új adatszerkezetet a meglévő és új paraméterekből
                data = {**data, **receivedData}
                #prep = "--source " + data["src"] +" --conf-thres " + data["conf"]  + " --min " + data["min"]
                prep=["python3",
                "detect.py",
                "--source", data["src"],
                "--conf-thres", data["conf"],
                "--min", data["min"]]
                #ha nincs futó process akkor elindul
                if proc is None:
                    proc = subprocess.Popen(prep)
                elif proc.poll() is not None:
                    proc = subprocess.Popen(prep)
                else:
                    await websocket.send(json.dumps({"status":"fail","proc":"start"}))
                time.sleep(3)
                #megnézi sikerült-e elindítani a processt
                if proc.poll() is not None:
                    await websocket.send(json.dumps({"status":"fail","proc":"start"}))
                else: 
                    await websocket.send(json.dumps({"status":"ok","proc":"start"}))

            #ha az üzenet első stringje change
            elif message[0] == "change":
                #megpróbáljuk parseolni az üzenetet
                try:
                    receivedData=json.loads(message[1])
                except:
                    await websocket.send("ERROR! JSON cannot be parsed!")
                #összeállítjuk az új adatszerkezetet a meglévő és új paraméterekből
                data = {**data, **receivedData}
                #ha van futó process akkor leállítjuk és elindítjuk újra
                if proc is not None:
                    if proc.poll() is None:
                        prep=["python3",
                            "detect.py",
                            "--source", data["src"],
                            "--conf-thres", data["conf"],
                            "--min", data["min"]]
                        proc.terminate()
                        roc = subprocess.Popen(prep)
                        if proc.poll() is not None:
                            await websocket.send(json.dumps({"status":"fail","proc":"change"}))
                        else: 
                            await websocket.send(json.dumps({"status":"ok","proc":"change"}))
                await websocket.send(json.dumps({"status":"ok","proc":"change"}))
            #ha az üzenet első stringje status akkor visszaküldjük az információkat
            elif message[0] == "status":
                await websocket.send(json.dumps(data))
            #ha az üzenet első stringje stop  
            elif message[0] == "stop":
                #megpróbáljuk leállítani a processt ha fut, ha nem hibát küldünk vissza
                if proc is None:
                    await websocket.send(json.dumps({"status":"fail","proc":"stop"}))
                elif proc.poll() is not None:
                    await websocket.send(json.dumps({"status":"fail","proc":"stop"}))
                else:
                    proc.terminate()
                    time.sleep(3)
                #ellenőrizzük leállt-e a process
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