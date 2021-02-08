import tensorflow as tf

from yolo.utils.run_utils import prep_gpu
from yolo.configs import yolo as exp_cfg
from yolo.tasks.yolo import YoloTask


if __name__ == "__main__":
    # initialize YOLOv4 model
    prep_gpu()

    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    mixed_precision.set_policy('float32')

    config = exp_cfg.YoloTask(
        model=exp_cfg.Yolo(
            _input_size=[608, 608, 3],
            base='v4',
            min_level=3,
            norm_activation=exp_cfg.common.NormActivation(activation='mish', use_sync_bn=False),
            _boxes=[
                '(12, 16)', '(19, 36)', '(40, 28)', '(36, 75)', '(76, 55)',
                '(72, 146)', '(142, 110)', '(192, 243)', '(459, 401)'
            ],
            dilate=False,
            filter=exp_cfg.YoloLossLayer(use_nms=False)))

    task = YoloTask(config)
    model = task.build_model()
    task.initialize(model)

    



    # pass in a all white image
    white_image = tf.fill([1, 608, 608, 3], 1.0)
    output = model.predict(white_image)

    model.build([1, 608, 608, 3])
    model.backbone.summary()
    # # raw output log
    # raw_tensor = output["raw_output"]['3']

    # with open("yolov4_raw_output.txt", "w") as fh:
    #     for tensor in raw_tensor.numpy():
    #         for w in tensor:
    #             for h in w:
    #                 for element in h:
    #                     fh.write(f"{element:.6f}\n")

    # oshape = [1, 608, 608, 3]
    # for layer in model.backbone.layers:
    #   try:
    #     for s in layer.submodules:
    #       try:
    #         oshape = s.compute_output_shape(oshape)
    #         print(oshape)
    #       except Exception as e:
    #         print(e)
    #   except:
    #     print("nothing")
    #   # try:
      #   for j in layer.layers:
      #     print(j)
      # except:
      #   print("no layers")
    """
    tf.print(raw_tensor,
             output_stream="file://yolov4_raw_output.txt",
             summarize=-1,
             sep="\n", end="")
    """
