import tensorflow as tf


def get_graph():
    return 

def model_fn(model):
    @tf.function
    def run_model(inputs):
        return model(inputs)
    return run_model

def main():
    path = "/home/vbanna/Desktop/Research/TensorFlowModelGardeners/saved_models/v4/regular"

    model = tf.saved_model.load(path)
    graph_func = model.signatures['serving_default']

    # hold the graph
    hold = graph_func.graph.as_graph_def(add_shapes = True)
    print(dir(hold))
    return 


def load_cv2():
    import cv2
    cv2.dnn.l

if __name__ == "__main__":
    main()