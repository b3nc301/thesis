#!/usr/bin/env python

# WS client example

import asyncio
import websockets

async def hello():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        message=["stop", ";" , "1 0 1"]

        await websocket.send(message)

        greeting = await websocket.recv()
        print(f"<<< {greeting}")

asyncio.run(hello())