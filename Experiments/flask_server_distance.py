from flask import Flask, request, jsonify

app = Flask(__name__)

distance_data = {} # in memory temp storage 

@app.route('/distance', methods=['POST'])
def receive_distance():
    global distance_data
    
    data = request.get_json()
    
    if data and 'distance' in data:
        print("Received data:", data)
        
        distance_data = data # overwrite existing data with latest received data
        
        return jsonify({
            "status": "ok", 
            "message": "Distance successfully stored.",
            "data": data # echoes back received data
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": "Invalid or missing JSON data."
        }), 400

# returns stored distance data, just retreives last recorded data
@app.route('/getDistance', methods=['GET'])
def print_distance():
    global distance_data
    
    if distance_data:
        return jsonify({
            "status": "success",
            "message": "Latest distance reading retrieved.",
            "data": distance_data
        }), 200
    else: #empty cuz post endpoint not hit yet 
        return jsonify({
            "status": "error",
            "message": "No distance data has been recorded yet."
        }), 404 # could use 204 to indicate no content 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
