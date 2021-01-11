import tensorflow as tf
from yolo.utils.run_utils import prep_gpu
from yolo.configs import yolo as exp_cfg
from yolo.tasks.yolo import YoloTask
# prep_gpu()

with tf.device("cpu:0"):
  model = None
  config = exp_cfg.YoloTask(
      model=exp_cfg.Yolo(base='v4tiny', 
                        min_level=4, 
                        norm_activation = exp_cfg.common.NormActivation(activation="leaky"), 
                        _boxes = ['(10, 14)', '(23, 27)', '(37, 58)', '(81, 82)', '(135, 169)', '(344, 319)'],
                        #_boxes = ['(20, 28)', '(46, 54)', '(74, 116)', '(81, 82)', '(135, 169)', '(344, 319)'],
                        #_boxes = ['(12, 16)', '(19, 36)', '(40, 28)', '(36, 75)','(76, 55)', '(72, 146)', '(142, 110)', '(192, 243)','(459, 401)'],
                        filter = exp_cfg.YoloLossLayer(use_nms=False)
                        ))  
  task = YoloTask(config)
  model = task.build_model()
  task.initialize(model)

  model(tf.ones((1, 416, 416, 3), dtype = tf.float32), training = False)
  model.save("saved_models/v4/tiny_no_nms")
  model.summary()

  # Convert the model

  converter = tf.lite.TFLiteConverter.from_saved_model("saved_models/v4/tiny_no_nms") # path to the SavedModel directory
  #converter.optimizations = [tf.lite.Optimize.DEFAULT]
  # converter.target_spec.supported_ops = [
  #   tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  #   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
  # ]
  #converter.target_spec.supported_types = [tf.float16]
  try:
    tflite_model = converter.convert()
  except:
    print("here")
    import sys
    sys.exit()

  # Save the model.
  with open('model-nopad.tflite', 'wb') as f:
    f.write(tflite_model)
