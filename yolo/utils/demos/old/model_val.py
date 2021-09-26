import tensorflow as tf

from yolo.utils.run_utils import prep_gpu
from yolo.configs import yolo as exp_cfg
from yolo.tasks.yolo import YoloTask

if __name__ == "__main__":
  # initialize YOLOv4 model
  prep_gpu()

  from tensorflow.keras.mixed_precision import experimental as mixed_precision
  mixed_precision.set_policy("float32")

  config = exp_cfg.YoloTask(
      model=exp_cfg.Yolo(
          _input_size=[608, 608, 3],
          base="v4",
          min_level=3,
          norm_activation=exp_cfg.common.NormActivation(
              activation="mish", use_sync_bn=False),
          _boxes=[
              "(12, 16)", "(19, 36)", "(40, 28)", "(36, 75)", "(76, 55)",
              "(72, 146)", "(142, 110)", "(192, 243)", "(459, 401)"
          ],
          dilate=False,
          filter=exp_cfg.YoloLossLayer(use_nms=False)))

  task = YoloTask(config)
  model = task.build_model()
  task.initialize(model)

  # pass in a all white image
  white_image = tf.fill([1, 608, 608, 3], 1.0)
  # layers = list(model.backbone.layers)
  output = model(white_image)

  # # raw output log
  raw_tensor = output["raw_output"]["5"]

  with open("yolov4_raw_output.txt", "w") as fh:
    print(raw_tensor.shape)
    for batch in range(raw_tensor.shape[0]):
      for channels in range(raw_tensor.shape[3]):
        for height in range(raw_tensor.shape[2]):
          for width in range(raw_tensor.shape[1]):
            element = raw_tensor[batch, height, width, channels]
            fh.write(f"{element:.6f}\n")
