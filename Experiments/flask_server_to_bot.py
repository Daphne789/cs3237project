from flask import Flask, jsonify
import psycopg2
from psycopg2 import pool
import requests
from contextlib import contextmanager

app = Flask(__name__)

db_pool = psycopg2.pool.SimpleConnectionPool(
    1,
    10,
    database="postgres",
    user="postgres",
    host="localhost",
    password="11223344",
    port=5431,
)


@contextmanager
def get_db_connection():
    conn = db_pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        db_pool.putconn(conn)


@app.route("/fetchData", methods=["GET"])
def fetch_distance_from_other_server():
    try:
        responseDist = requests.get("http://localhost:5001/getDistance", timeout=5)
        responseImu = requests.get("http://10.235.243.177:5000/control", timeout=5)

        print(responseDist.json())
        print(responseImu.json())
        responseDist.raise_for_status()
        responseImu.raise_for_status()

        distance = responseDist.json()["data"]["distance"]
        command = responseImu.json()["command"]

        # with get_db_connection() as conn:
        #     with conn.cursor() as cur:
        #         cur.execute(
        #             "INSERT INTO test_table (distance, command) VALUES (%s, %s)",
        #             (distance, command),
        #         )

        if command == "STRAIGHT":
            commandNum = 1
        elif command == "BACKWARD":
            commandNum = 2
        elif command == "LEFT":
            commandNum = 3
        elif command == "RIGHT":
            commandNum = 4
        elif command == "SIDE_LEFT":
            commandNum = 5
        elif command == "SIDE_RIGHT":
            commandNum = 6
        elif command == "FULL_TURN":
            commandNum = 7
        elif command == "JUMP":
            commandNum = 8
        else:
            commandNum = 0

        return str(commandNum)

    except requests.exceptions.RequestException as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
