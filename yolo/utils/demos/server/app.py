from flask import Flask, request , jsonify
import numpy as np 
import base64
import cv2

app = Flask(__name__)


@app.route("/init", methods=['POST'])
def init_model():
    
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
    