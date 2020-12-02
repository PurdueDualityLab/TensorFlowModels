import socket
import tensorflow as tf

class SocketServer(object):
    def __init__(self, 
                 address = "localhost", 
                 PORT = 80, 
                 model = None, 
                 pwidth = 416,
                 pheight = 416,
                 que_size = 10, 
                 max_batch = 10, 
                 policy = "float16",
                 run_strategy = "mirrored",
                 preprocess_on_cpu = False,
                 resolution = None,
                 wait_time = None):
        
        self.address = address
        self.PORT = PORT

        self._device = self.get_device(run_strategy)
        self._dtype = self.set_policy(policy)
        self._que_size = que_size
        self._max_batch = max_batch
        self._que_size = que_size
        self._pwidth = pwidth
        self._pheight = pheight

        self._wait_init = wait_time
        self._wait_time = self._get_wait_time(wait_time, max_batch)

        self._pred_fn = model
        self._preprocess_fn = None
        self._post_fn = None

        if hasattr(model, "predict"):
            self._inference_fn = model.predict
        else:
            self._inference_fn = model 

        self._load_que = Queue(self._batch_size * scale_que)
        self._return_que = Queue(scale_que)
        return 

    def get_wait_time(self, wait_time, batch_size):
        if wait_time == None:
            return 0.001 * batch_size
        return wait_time

    def get_device(self, policy):
        if not isinstance(policy, str):
            return policy
        elif policy == "mirrored":
            return tf.distribute.MirroredStrategy()
        elif policy == "oneDevice":
            return tf.distribute.OneDeviceStrategy()
        elif policy == "tpu":
            return tf.distribute.TPUStrategy()
        elif "GPU" in policy:
            return tf.device(policy)
        return tf.device("/CPU:0")

    def set_policy(self, policy_name):
        if policy_name == None:
            return tf.float32
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy(policy_name)
        mixed_precision.set_policy(policy)
        dtype = policy.compute_dtype
        return dtype
        
    def set_model(self):
        return 

    def put_que(self):
        return