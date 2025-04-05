from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import random
import time
import threading
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def dashboard():
    # Example data structure
    data = {
        "vehicle_count": [
            {"time": "10:00", "count": 5},
            {"time": "10:05", "count": 10}
        ],
        "vehicle_category": {
            "Car": 15,
            "Bike": 10,
            "Bus": 5,
            "Truck": 3
        },
        "traffic_density": [
            [1, 2], [3, 4, 5], [1, 2, 3]
        ],
        "high_speed_violations": [3, 2, 1, 4],
        "speed_distribution": [10, 20, 30, 40, 50]
    }

    # Ensure `data` is always passed to the template
    return render_template("analysis.html", data=data)

@socketio.on("connect")
def handle_connect():
    print("✅ Client Connected")

@socketio.on("disconnect")
def handle_disconnect():
    print("❌ Client Disconnected")

def send_fake_data():
    """ Generate and send random traffic data every 5 seconds """
    while True:
        data = {
            "vehicle_count": [
                {"time": "10:00", "count": random.randint(5, 15)},
                {"time": "10:05", "count": random.randint(5, 15)}
            ],
            "vehicle_category": {
                "Car": random.randint(10, 30),
                "Bike": random.randint(5, 20),
                "Bus": random.randint(1, 10),
                "Truck": random.randint(1, 5)
            },
            "traffic_density": [
                [1, 2, 3], [2, 3, 4, 5], [3, 4], [1, 2, 3, 4, 5, 6]
            ],
            "high_speed_violations": [random.randint(0, 5) for _ in range(4)],
            "speed_distribution": [random.randint(10, 90) for _ in range(10)]
        }

        try:
            # Ensuring the data is serializable
            json.dumps(data)
            print("🚀 Sending WebSocket Data:", data)  # Debugging
            socketio.emit("dashboard_update", data)
        except TypeError as e:
            print("❌ JSON Serialization Error:", e)

        time.sleep(5)

# Run the data sender in a separate thread
threading.Thread(target=send_fake_data, daemon=True).start()

if __name__ == "__main__":
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
