from flask import Flask
import random

app = Flask(__name__)

@app.route('/')
def command():
    value = random.randint(0, 4)
    print(f"Sent command: {value}")
    return str(value)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
