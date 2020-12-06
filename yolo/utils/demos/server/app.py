from flask import Flask, request , jsonify
import numpy as np 
import base64
import cv2

from yolo.demos.three_servers.model_server import ModelServer
from yolo.utils.run_utils import prep_gpu
from yolo.configs import yolo as exp_cfg
from yolo.tasks.yolo import YoloTask

app = Flask(__name__)

class ServerAttr(object):
    def __init__(self):
        self.model = None
        return 
    
    def init_model(self):

        return

    def close_model(self):
        del self.model
        self.model = None
        return
    
MODEL = ServerAttr()

@app.route("/init", methods=['POST'])
def init_model():
    """
    get attr for model version
    """



    return

@app.route("/close", methods=['POST'])
def close_model():
    
    return

@app.route("/send_frame", methods=['POST'])
def send_frame():
    
    return

@app.route("/get_frame", methods=['GET'])
def get_frame():
    
    return 

@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == "__main__":
    