from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="a"
)

werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
werkzeug_logger.addHandler(file_handler)
werkzeug_logger.propagate = False

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

data_store = {
    "vehicle_count": [],
    "vehicle_category": {},  # Now using timestamped keys
    "traffic_density": [],
    "high_speed_violations": [],
    "speed_distribution": []  # Now storing time-stamped entries
}

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

@app.route('/api/update_data', methods=['POST'])
def update_data():
    global data_store
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    logging.debug("Parsed JSON: %s", data)

    vehicle_count = data.get("vehicle_count", 0)
    vehicle_types = data.get("vehicle_types", {})
    average_speed = data.get("average_speed", 0)
    overspeeding = data.get("overspeeding", 0)
    timestamp = data.get("timestamp", "N/A")

    data_store["vehicle_count"].append({
        "time": timestamp,
        "count": vehicle_count
    })
    if len(data_store["vehicle_count"]) > 20:
        data_store["vehicle_count"].pop(0)

    data_store["vehicle_category"][timestamp] = vehicle_types
    if len(data_store["vehicle_category"]) > 20:
        data_store["vehicle_category"].pop(next(iter(data_store["vehicle_category"])))

    data_store["traffic_density"].append([])  # Placeholder for future logic
    if len(data_store["traffic_density"]) > 20:
        data_store["traffic_density"].pop(0)

    data_store["high_speed_violations"].append(overspeeding)
    if len(data_store["high_speed_violations"]) > 20:
        data_store["high_speed_violations"].pop(0)

    data_store["speed_distribution"].append({"time": timestamp, "speed": average_speed})
    if len(data_store["speed_distribution"]) > 20:
        data_store["speed_distribution"].pop(0)

    socketio.emit('dashboard_update', data_store)

    return jsonify({"status": "success"}), 200

@socketio.on("connect")
def handle_connect():
    logging.info("Frontend connected to WebSocket.")
    emit("dashboard_update", data_store)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
