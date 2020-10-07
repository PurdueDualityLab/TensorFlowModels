import tensorflow as tf 
import tensorflow.keras as ks
# import tensorflow.experimental.tensorrt as trt 
import tensorflow.lite as tflite

from tensorflow.python.compiler.tensorrt import trt_convert as trt

from yolo.utils.testing_utils import prep_gpu

def load_model():
    from yolo.modeling.Yolov4 import Yolov4
    prep_gpu()
    model = Yolov4(model = "regular", policy="float32", use_tie_breaker=True)
    model.build((None, 416, 416, 3))
    model.load_weights_from_dn()
    model(tf.random.normal((1, 416, 416, 3)))
    model.save("testing_weights/yolov4/full_models/v4_32_fixed_size", include_optimizer=False)

def load_model_16(name = "testing_weights/yolov4/full_models/v4_16"):
    from yolo.modeling.Yolov4 import Yolov4
    prep_gpu()
    model = Yolov4(model = "regular", policy="float16", use_tie_breaker=True)
    model.build((None, None, None, 3))
    model.load_weights_from_dn()
    model(tf.random.normal((1, 416, 416, 3)))
    model.save(name, include_optimizer=False)

def convert_trt():
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode='FP16',is_dynamic_op=True, max_workspace_size_bytes=4000000000,max_batch_size=8)
    
    converter = trt.TrtGraphConverterV2(input_saved_model_dir="testing_weights/yolov4/full_models/v4_32_fixed_size", conversion_params=params)
    converter.convert()
    converter.save("testing_weights/yolov4/full_models/v4_32_tensorrt_fixed")
    saved_model_loaded = tf.saved_model.load("testing_weights/yolov4/full_models/v4_32_tensorrt_fixed")
    graph_func = saved_model_loaded.signatures[tf.python.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    trt_graph = graph_func.graph.as_graph_def()
    for n in trt_graph.node:
        print(n.op)
        if n.op == "TRTEngineOp":
            print("Node: %s, %s" % (n.op, n.name.replace("/", "_")))
        else:
            print("Exclude Node: %s, %s" % (n.op, n.name.replace("/", "_")))


    trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
    print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
    all_nodes = len([1 for n in trt_graph.node])
    print("numb. of all_nodes in TensorRT graph:", all_nodes)

def get_rt_model(modelpath, savepath, input_fn):
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode='FP16',is_dynamic_op=True, max_workspace_size_bytes=4000000000,max_batch_size=8)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=modelpath, conversion_params=params)
    converter.convert()
    converter.build(input_fn)
    converter.save(savepath)
    return

def get_func_from_saved_model(saved_model_dir):
  saved_model_loaded = tf.saved_model.load(saved_model_dir, tags=[tf.python.saved_model.tag_constants.SERVING])
  graph_func = saved_model_loaded.signatures[tf.python.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  return graph_func

if __name__ == "__main__": 
    #load_model_16()
    convert_trt()
    # load_model()
    # load_model_16()