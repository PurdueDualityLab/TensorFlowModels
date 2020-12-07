from flask import Flask, request , jsonify
import numpy as np 
import base64
import cv2
import sys
import tensorflow as tf

# from yolo.demos.three_servers.model_server import ModelServer
# from yolo.utils.run_utils import prep_gpu
# from yolo.configs import yolo as exp_cfg
# from yolo.tasks.yolo import YoloTask
from queue import Queue
import urllib.request

app = Flask(__name__)

class dualQue(object):
    def __init__(self):
        self._return_buffer = Queue(maxsize = 10)
    
    def put(self, raw_frame):
        if self._return_buffer.full():
            return False
        with urllib.request.urlopen(raw_frame) as response:
            data = response.read()
        img = tf.io.decode_jpeg(data) 
        self._return_buffer.put(img.numpy())
        return True
    
    def get(self):
        if self._return_buffer.empty():
            return None
        f = self._return_buffer.get()
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.imencode('.jpg', f)[1].tostring()
        f = base64.b64encode(f).decode('utf-8')
        b64_src = 'data:image/jpg;base64,'
        f = b64_src + f
        return f

class ServerAttr(object):
    def __init__(self):
        self.model = None
        return 
    
    def init_model(self, model):
        self.model = dualQue()
        print(model)
        return

    def close_model(self):
        del self.model
        self.model = None
        return
    
    def get(self):
        return self.model.get()
    
    def put(self, image):
        return self.model.put(image)

    
    
MODEL = ServerAttr()

@app.route("/")
def init():
    """
    get attr for model version
    """
    return "<h1>hello world</h1>"


@app.route("/set/<version>", methods=['POST'])
def hello_model(version):
    """
    get attr for model version
    """
    MODEL.init_model(version)
    return f"<h1>{version}</h1>"


@app.route("/close", methods=['POST'])
def close_model():
    MODEL.close_model()
    return f"<h1>closed</h1>"

@app.route("/send_frame", methods=['POST'])
def send_frame():
    print("decoding image", request)
    data_url = request.values['frame']
    data = not MODEL.put(data_url)
    return jsonify({"dropped": data})

@app.route("/get_frame", methods=['GET'])
def get_frame():
    print("getting image")
    frame = MODEL.get()
    if frame == None:
        return jsonify({"frame":"null"})
    else:
        return jsonify({"frame": frame})

@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 5000)