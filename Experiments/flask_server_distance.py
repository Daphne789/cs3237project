from flask import Flask, request, jsonify
import time

app = Flask(__name__)

# 💾 Step 1: Create a simple, in-memory storage dictionary
# NOTE: This only stores the data temporarily while the app is running.
# For production, you'd use a database (e.g., SQLite, PostgreSQL).
distance_data = {}

# @app.route('/distance', methods=['POST'])
# def receive_distance():
#     global distance_data

#     # 📥 Step 2: Receive the JSON data
#     data = request.get_json()

#     if data and 'distance' in data:
#         print("Received data:", data)

#         # 💾 Step 3: Store the received data
#         # We'll overwrite the existing data with the latest received distance
#         distance_data = data

#         return jsonify({
#             "status": "ok",
#             "message": "Distance successfully stored.",
#             "data": data # Optionally echo back the received data
#         }), 200
#     else:
#         return jsonify({
#             "status": "error",
#             "message": "Invalid or missing JSON data."
#         }), 400

# ---
# This GET endpoint is modified to return the *stored* distance data.
# We'll simplify it to just retrieve the last recorded distance.
# ---


# return {
#     "ok": True,
#     "device_id": device_id,
#     "command": latest.get("command", "NONE"),
#     "label": latest.get("label", "NONE"),
#     "confidence": float(latest.get("confidence", 0.0)),
#     "speed": float(ctl.get("speed_lp", 0.0)),
#     "yaw_rate_dps": float(ctl.get("yaw_rate_lp_dps", 0.0)),
#     "turn_angle_deg": float(ctl.get("angle_deg", 0.0)),
#     "left_speed": float(ctl.get("left_speed", 0.0)),
#     "right_speed": float(ctl.get("right_speed", 0.0)),
#     "timestamp": time.time(),
# }
# returns stored distance data, just retreives last recorded data
@app.get("/getDistance")
def get_distance():
    global distance_data

    # if not distance_data:  # empty cuz post endpoint not hit yet
    #     return {"ok": False, "error": "404, Distance probably not recorded"}, 404

    latest = distance_data

    return {
        "ok": True,
        "data": {
            "device_id": latest.get("device_id", "unknown_device"),
            "distance": float(latest.get("distance", -1.0)),
            "timestamp": latest.get("timestamp", time.time()),
        },
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
