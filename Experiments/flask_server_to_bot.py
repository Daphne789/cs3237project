from flask import Flask, jsonify
import psycopg2
import requests

app = Flask(__name__)
conn = psycopg2.connect(
    database="postgres",
    user="postgres",
    host="localhost",
    password="11223344",
    port=5431,
)


@app.route("/fetchData", methods=["GET"])
def fetch_distance_from_other_server():
    try:
        responseDist = requests.get("http://localhost:5001/getDistance")
        responseImu = requests.get("http://localhost:5003/control")

        print(responseDist.json())
        print(responseImu.json())
        distance = responseDist.json()["data"]["distance"]
        command = responseImu.json()["command"]

        return {
            "ok": True,
            "data": {
                "distance": distance,
                "command": command,
            },
        }
    except requests.exceptions.RequestException as e:
        return {"ok": False}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
