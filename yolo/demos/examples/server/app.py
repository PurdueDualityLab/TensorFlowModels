from yolo.utils.run_utils import prep_gpu
try:
  prep_gpu()
except BaseException:
  print("GPU's already prepped")

from flask import Flask, request, jsonify
import numpy as np
import base64
import cv2
import sys
import tensorflow as tf

from yolo.demos.three_servers import model_server as ms

from yolo.configs import yolo as exp_cfg
from yolo.tasks.yolo import YoloTask
from yolo.utils.demos import utils
from yolo.utils.demos import coco
from queue import Queue
import urllib.request

app = Flask(__name__)


class dualQue(object):

  def __init__(self):
    self._return_buffer = Queue(maxsize=10)

  def put(self, raw_frame):
    if self._return_buffer.full():
      return False
    if hasattr(raw_frame, "numpy"):
      raw_frame = raw_frame.numpy()
    self._return_buffer.put(raw_frame)
    return True

  def get(self):
    if self._return_buffer.empty():
      return None
    f = self._return_buffer.get()
    return f


def build_model(version):
  if version == "v4":
    config = exp_cfg.YoloTask(
        model=exp_cfg.Yolo(
            base="v4",
            min_level=3,
            norm_activation=exp_cfg.common.NormActivation(activation="mish"),
            #_boxes = ['(10, 14)', '(23, 27)', '(37, 58)', '(81, 82)', '(135, 169)', '(344, 319)'],
            _boxes=[
                "(12, 16)", "(19, 36)", "(40, 28)", "(36, 75)", "(76, 55)",
                "(72, 146)", "(142, 110)", "(192, 243)", "(459, 401)"
            ],
        ))
  elif "tiny" in version:
    config = exp_cfg.YoloTask(
        model=exp_cfg.Yolo(
            base=version,
            min_level=4,
            norm_activation=exp_cfg.common.NormActivation(activation="leaky"),
            _boxes=[
                "(10, 14)", "(23, 27)", "(37, 58)", "(81, 82)", "(135, 169)",
                "(344, 319)"
            ],
            #_boxes = ['(12, 16)', '(19, 36)', '(40, 28)', '(36, 75)','(76, 55)', '(72, 146)', '(142, 110)', '(192, 243)','(459, 401)'],
        ))
  else:
    config = exp_cfg.YoloTask(
        model=exp_cfg.Yolo(
            base=version,
            min_level=3,
            norm_activation=exp_cfg.common.NormActivation(activation="leaky"),
            #_boxes = ['(10, 14)', '(23, 27)', '(37, 58)', '(81, 82)', '(135, 169)', '(344, 319)'],
            _boxes=[
                "(10, 13)", "(16, 30)", "(33, 23)", "(30, 61)", "(62, 45)",
                "(59, 119)", "(116, 90)", "(156, 198)", "(373, 326)"
            ],
        ))

  task = YoloTask(config)
  model = task.build_model()
  task.initialize(model)

  pfn = ms.preprocess_fn
  pofn = utils.DrawBoxes(
      classes=80, labels=coco.get_coco_names(), display_names=True, thickness=2)
  server_t = ms.ModelServer(
      model=model,
      preprocess_fn=pfn,
      postprocess_fn=pofn,
      wait_time=0.00001,
      max_batch=5)
  return server_t


class ServerAttr(object):

  def __init__(self):
    self.model = None
    return

  def init_model(self, model):
    self.model = build_model(model)
    self.model.start()
    print(model)
    return

  def close_model(self):
    self.model.close()
    del self.model
    self.model = None
    return

  def get(self):
    f = self.model.get()
    if np.max(f) <= 1:
      f = (f * 255).astype(np.uint8)
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    f = cv2.imencode(".jpg", f)[1].tostring()
    f = base64.b64encode(f).decode("utf-8")
    b64_src = "data:image/jpg;base64,"
    f = b64_src + f
    return f

  def getall(self):
    fs = self.model.getall()
    fl = []
    for f in fs:
      if np.max(f) <= 1:
        f = (f * 255).astype(np.uint8)
      f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
      f = cv2.imencode(".jpg", f)[1].tostring()
      f = base64.b64encode(f).decode("utf-8")
      b64_src = "data:image/jpg;base64,"
      f = b64_src + f
      fl.append(f)
    return fl

  def put(self, image):
    with urllib.request.urlopen(image) as response:
      data = response.read()
    img = tf.io.decode_jpeg(data)
    return self.model.put(img / 255)


MODEL = ServerAttr()


@app.route("/")
def init():
  """
    get attr for model version
    """
  return "<h1>hello world</h1>"


@app.route("/set/<version>", methods=["POST"])
def hello_model(version):
  """
    get attr for model version
    """
  MODEL.init_model(version)
  return f"<h1>{version}</h1>"


@app.route("/close", methods=["POST"])
def close_model():
  MODEL.close_model()
  return f"<h1>closed</h1>"


@app.route("/send_frame", methods=["POST"])
def send_frame():
  try:
    print("decoding image", request)
    data_url = request.values["frame"]
    data = not MODEL.put(data_url)
    return jsonify({"dropped": data})
  except BaseException:
    return jsonify({"dropped": True})


@app.route("/get_frame", methods=["GET"])
def get_frame():
  print("getting image")
  try:
    frame = MODEL.get()
    if type(frame) == type(None):
      return jsonify({"frame": "null"})
    else:
      return jsonify({"frame": frame})
  except BaseException:
    return jsonify({"frame": "null"})


@app.route("/getall_frames", methods=["GET"])
def getall_frames():
  print("getting image")
  try:
    frame = MODEL.getall()
    return jsonify({"frames": frame})
  except BaseException:
    return jsonify({"frames": "null"})


@app.after_request
def after_request(response):
  print("log: setting cors", file=sys.stderr)
  response.headers.add("Access-Control-Allow-Origin", "*")
  response.headers.add("Access-Control-Allow-Headers",
                       "Content-Type,Authorization")
  response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE")
  return response


def load_for_colab():
  from flask_ngrok import run_with_ngrok
  run_with_ngrok(app)  # Start ngrok when app is run
  app.run()
  return


def load(port):
  app.run(host="127.0.0.1", port=port)
  return


if __name__ == "__main__":
  load(5000)
