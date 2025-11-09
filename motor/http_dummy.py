from flask import Flask
import random

app = Flask(__name__)

@app.route('/')
def command():
    # value = 1
    # print(f"Sent command: {value}")
    # return str(value)
    # for i in range(0,8):
    #     print(f"Sent command: {i}")
    #     return str(i)
    value = random.randint(0, 7)
    print(f"Sent command: {value}")
    return str(value)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
