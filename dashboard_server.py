"""
WORKFLOW - HOW TO RUN:
(Internal server run automatically by run_with_dashboard.py)
"""
import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()
state_queue = asyncio.Queue(maxsize=5)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()
latest_state_cache = "{}"

def push_state(state_dict):
    try:
        state_queue.put_nowait(state_dict)
    except asyncio.QueueFull:
        pass

async def _broadcast_loop():
    global latest_state_cache
    while True:
        state_dict = await state_queue.get()
        latest_state_cache = json.dumps(state_dict)
        await manager.broadcast(latest_state_cache)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_broadcast_loop())

@app.get("/")
async def get_dashboard():
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/stats")
async def get_stats():
    if os.path.exists("training_stats.json"):
        with open("training_stats.json", "r") as f:
            return json.loads(f.read())
    return {}

@app.get("/state")
async def get_state():
    return json.loads(latest_state_cache)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)