from yolo.modeling.yolo_v3 import DarkNet53, Yolov3

class Model(object):
    def __init__(self, model_generation = 3, model_type = "regular", anchors = None, masks = None):
        return

    def __call__(self, x):
        return

    def predict(self, x):
        return
    
    def train_tpu(self, key, dataset):
        return
    
    def train_gpu(self, dataset):
        return
    
    def train(self, dataset):
        return
    
    def save_weights(self):
        return

if __name__ == "__main__":
    model = Yolov3(type = 'regular', classes=80)
    model.build(input_shape = (None, None, None, 3))
    model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True, config_file=None, weights_file="yolov3_416.weights")
    model.summary()
    model.save(filepath = "/Users/vishnubanna/Projects")