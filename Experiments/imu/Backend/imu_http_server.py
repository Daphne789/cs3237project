from flask import Flask, request, jsonify
import csv
from datetime import datetime
import sys

app = Flask(__name__)

if len(sys.argv) < 2:
    print("Usage: python imu_server.py ACTION_LABEL")
    sys.exit(1)

action_label = sys.argv[1]

CSV_FILENAME = f'gyro_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

# Write CSV header
with open(CSV_FILENAME, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'timestamp', 'accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z','action_label'
    ])

@app.route('/imu', methods=['POST'])
def receive_imu():
    data = request.get_json()

    if not data:
        return jsonify({"status": "error", "message": "No JSON received"}), 400

    print(f"Received IMU data: {data}")

    # Write data row with action_label
    with open(CSV_FILENAME, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            data.get('timestamp'),
            data.get('accel_x'),
            data.get('accel_y'),
            data.get('accel_z'),
            data.get('gyro_x'),
            data.get('gyro_y'),
            data.get('gyro_z'),
            action_label
        ])

    return jsonify({"status": "success"}), 200

if __name__ == "__main__":
    print(f"Flask IMU server started! Saving data to: {CSV_FILENAME}")
    print(f"Action Label: {action_label}")
    app.run(host='0.0.0.0', port=3237)
