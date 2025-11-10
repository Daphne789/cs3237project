from flask import Flask, jsonify
import psycopg2
from psycopg2 import pool
import requests
from contextlib import contextmanager

app = Flask(__name__)

# Use connection pooling instead of a single connection
db_pool = psycopg2.pool.SimpleConnectionPool(
    1,
    10,  # min and max connections
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
        responseImu = requests.get("http://localhost:5003/control", timeout=5)

        responseDist.raise_for_status()
        responseImu.raise_for_status()

        distance = responseDist.json()["data"]["distance"]
        command = responseImu.json()["command"]

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO test_table (distance, command) VALUES (%s, %s)",
                    (distance, command),
                )

        return jsonify(
            {
                "ok": True,
                "data": {
                    "distance": distance,
                    "command": command,
                },
            }
        )
    except requests.exceptions.RequestException as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
