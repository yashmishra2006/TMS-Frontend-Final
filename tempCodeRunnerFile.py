from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import asyncio
import websockets
import json

app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")

# WebSocket API URL
WS_API_URL = "ws://127.0.0.1:5001/ws/detect"  # Change to actual API address

async def fetch_vehicle_data():
    """Connects to WebSocket API and relays data to frontend."""
    async with websockets.connect(WS_API_URL) as ws:
        print("Connected to WebSocket API.")
        while True:
            try:
                data = await ws.recv()
                vehicle_data = json.loads(data)
                socketio.emit("vehicle_data", vehicle_data)  # Send to frontend
            except Exception as e:
                print(f"Error receiving data: {e}")
                break

@socketio.on("connect")
def handle_connect():
    print("Frontend connected to WebSocket.")
    socketio.start_background_task(lambda: asyncio.run(fetch_vehicle_data()))

@app.route("/")
def serve_index():
    return render_template("index.html")

@app.route("/upload", methods=["GET"])
def upload_file():
    return render_template("upload.html")

@app.route("/processed", methods=["GET"])
def processed_file():
    return render_template("processed.html")

@app.route("/analysis", methods=["GET"])
def analysis():
    return render_template("analysis.html")

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
