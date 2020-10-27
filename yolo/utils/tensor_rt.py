import tensorflow as tf
import tensorflow.keras as ks
# import tensorflow.experimental.tensorrt as trt
import tensorflow.lite as tflite

from tensorflow.python.compiler.tensorrt import trt_convert as trt

from yolo.utils.testing_utils import prep_gpu


def load_model_v4(name="testing_weights/yolov4/full_models/v4_32",
                  policy="float32"):
    from yolo.modeling.Yolov4 import Yolov4
    model = Yolov4(model="regular",
                   policy=policy,
                   use_nms=False,
                   using_rt=False)
    model.build((None, None, None, 3))
    model.load_weights_from_dn()
    model(tf.random.normal((1, 416, 416, 3)))
    model.save(name, include_optimizer=False)
    del model
    return name


def load_model_v3(type="regular",
                  name="testing_weights/yolov4/full_models/v4_32",
                  policy="float32"):
    from yolo.modeling.Yolov3 import Yolov3
    model = Yolov3(model=type, policy=policy, use_nms=False, using_rt=False)
    model.build((None, None, None, 3))
    model.load_weights_from_dn()
    model(tf.random.normal((1, 416, 416, 3)))
    model.save(name, include_optimizer=False)
    del model
    return name


def get_rt_model(modelpath, savepath, input_fn):
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode='FP16',
        is_dynamic_op=True,
        max_workspace_size_bytes=4000000000,
        max_batch_size=8)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=modelpath,
                                        conversion_params=params)
    converter.convert()
    converter.build(input_fn)
    converter.save(savepath)
    return


def get_func_from_saved_model(saved_model_dir):
    saved_model_loaded = tf.saved_model.load(
        saved_model_dir, tags=[tf.python.saved_model.tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[
        tf.python.saved_model.signature_constants.
        DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    return graph_func


class TensorRT(object):
    def __init__(self,
                 saved_model,
                 image_shape=(416, 416, 3),
                 over_write_model=False,
                 save_new_path=None,
                 precision_mode="FP16",
                 is_dynamic_op=True,
                 max_workspace_size_bytes=1,
                 max_batch_size=8,
                 use_calibration=False):
        self._model_path = saved_model
        self._model_save_path = saved_model if over_write_model else save_new_path
        self._batch_size = max_batch_size
        self._image_shape = image_shape
        self._use_calibration_fn = use_calibration
        if self._model_save_path == None:
            raise Exception("model save path not specified")
        self._params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=precision_mode,
            is_dynamic_op=is_dynamic_op,
            max_workspace_size_bytes=max_workspace_size_bytes,
            max_batch_size=8,
            use_calibration=use_calibration)
        #self._parent = None
        self._model = None
        self._processor = None
        self._converted = False
        return

    def callibration_fn(self):
        import numpy as np
        shaped_item = np.random.normal(size=(1, 1, *self._image_shape))
        shaped_item = shaped_item.astype(np.float32)
        print(shaped_item.shape, shaped_item.dtype)
        yield shaped_item

    def gen_fn(self):
        for i in range(1, self._batch_size + 1):
            item = tf.random.normal((i, *self._image_shape))
            print(item.shape)
            yield (item, )

    def convertModel(self):
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=self._model_path,
            conversion_params=self._params)
        if self._use_calibration_fn:
            converter.convert(calibration_input_fn=self.callibration_fn)
        else:
            converter.convert()
        converter.build(self.gen_fn)
        converter.save(self._model_save_path)
        self._converted = True
        return

    def compile(self):
        if self._converted:
            saved_model_dir = self._model_save_path
        else:
            saved_model_dir = self._model_path

        saved_model_loaded = tf.saved_model.load(
            saved_model_dir,
            tags=[tf.python.saved_model.tag_constants.SERVING])
        graph_func = saved_model_loaded.signatures[
            tf.python.saved_model.signature_constants.
            DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        #graph_func = saved_model_loaded.signatures["serving"]
        self._model = graph_func
        #self._parent = saved_model_loaded
        return

    def summary(self):
        print(dir(self._model))
        #print(dir(self._parent), self._parent._x_y_scales)
        print(self._model.output_shapes)
        print(self._model.outputs)
        pass

    def set_postprocessor_fn(self, postprocess_fn):
        self._processor = postprocess_fn
        return

    @tf.function
    def _compute(self, inputs):
        return self._model(inputs)

    @tf.function
    def _postprocess(self, inputs):
        return self._processor(inputs)

    @tf.function
    def predict(self, mod_inputs):
        if self._model == None:
            raise Exception("Compile the Model first")
        if self._processor != None:
            return self._postprocess(self._compute(mod_inputs))
        else:
            return self._compute(mod_inputs)

    @tf.function
    def stack(self, outputs, new):
        if isinstance(outputs, dict):
            keys = outputs.keys()
            out_dict = dict()
            for key in keys:
                out_dict[key] = tf.concat([outputs[key], new[key]], axis=0)
            return out_dict
        elif isinstance(outputs, list):
            return outputs
        else:
            return tf.concat([outputs, new], axis=0)

    def __call__(self, mod_inputs):
        if self._model == None:
            raise Exception("Compile the Model first")
        if self._processor != None:
            return self._postprocess(self._model(mod_inputs))
        else:
            return self._model(mod_inputs)

    @property
    def outputs():
        if self._model == None:
            raise Exception("Compile the Model first")
        a = self._model.outputs
        b = self._model.output_shapes
        return a, b


if __name__ == "__main__":
    prep_gpu()

    def func(inputs):
        boxes = inputs["bbox"]
        classif = inputs["classes"]
        nms = tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2), classifs, 200, 200, 0.5, 0.5)
        return {
            "bbox": nms.nmsed_boxes,
            "classes": nms.nmsed_classes,
            "confidence": nms.nmsed_scores,
        }

    name = "testing_weights/yolov4/full_models/v4_32"  #load_model()
    new_name = f"{name}_tensorrt"
    model = TensorRT(saved_model=new_name,
                     save_new_path=new_name,
                     max_workspace_size_bytes=4000000000)
    #model.convertModel()
    model.compile()
    model.summary()
    model.set_postprocessor_fn(func)
