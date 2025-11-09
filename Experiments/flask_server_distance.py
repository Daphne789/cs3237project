from flask import Flask, request, jsonify

app = Flask(__name__)

# 💾 Step 1: Create a simple, in-memory storage dictionary
# NOTE: This only stores the data temporarily while the app is running.
# For production, you'd use a database (e.g., SQLite, PostgreSQL).
distance_data = {}

@app.route('/distance', methods=['POST'])
def receive_distance():
    global distance_data
    
    # 📥 Step 2: Receive the JSON data
    data = request.get_json()
    
    if data and 'distance' in data:
        print("Received data:", data)
        
        # 💾 Step 3: Store the received data
        # We'll overwrite the existing data with the latest received distance
        distance_data = data
        
        return jsonify({
            "status": "ok", 
            "message": "Distance successfully stored.",
            "data": data # Optionally echo back the received data
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": "Invalid or missing JSON data."
        }), 400

# ---
# This GET endpoint is modified to return the *stored* distance data.
# We'll simplify it to just retrieve the last recorded distance.
# ---

@app.route('/getDistance', methods=['GET'])
def print_distance():
    global distance_data
    
    # 💡 Step 4: Check if any data has been stored
    if distance_data:
        # 📤 Step 5: Return the stored distance data in the response
        return jsonify({
            "status": "success",
            "message": "Latest distance reading retrieved.",
            "data": distance_data
        }), 200
    else:
        # Handle the case where the POST endpoint hasn't been hit yet
        return jsonify({
            "status": "error",
            "message": "No distance data has been recorded yet."
        }), 404 # Use 404 (Not Found) or 204 (No Content) for better practice

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
