import tensorflow as tf
from yolo.modeling.functions.yolo_v1_loss import Yolo_Loss_v1

def testBoxFormat():
    # sample prediction, batch size 1, s = 1
    y_pred = [[[0, 0, 1, 50, 1,                 # box 1  
                0, 0, 1, 1, 0,                  # box 2
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9]]] # class probs
    
    # sample ground truth, batch size 1, s = 1
    y_true = [[[0, 0, 1, 1, 1,                   # box 1  
                0, 0, 1, 1, 1,                   # box 2
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]] # class probs
    
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)

    loss = Yolo_Loss_v1(num_classes=10)
    loss(y_true, y_pred)

def testShape1():
    S = 7
    B = 2
    C = 20
    # Random prediction
    y_pred = tf.random.uniform([1, S, S, B*5 + C], minval=0, maxval=1, dtype=tf.float32, name="pred")

    # Random ground truth
    y_true_xywh = tf.random.uniform([1, S, S, 4], minval=0, maxval=1, dtype=tf.float32, name="true")
    y_true_c = tf.cast(tf.random.uniform([1, S, S, 1], minval=0, maxval=2, dtype=tf.int32, name="true"), tf.float32)
    y_true_xywhc = tf.concat([y_true_xywh, y_true_c], axis=-1)

    y_true_class = tf.random.uniform([1, S, S, C], minval=0, maxval=2, dtype=tf.float32, name="true")
    y_true = tf.concat([y_true_xywhc, y_true_xywhc, y_true_class], axis=-1)

    tf.print("prediction: ", y_pred)
    tf.print("ground truth: ", y_true)

    loss = Yolo_Loss_v1()
    tf.print(loss(y_true, y_pred))

def testNoObject():
    S = 2
    B = 2
    C = 20

    y_pred = tf.random.uniform([1, S, S, B*5 + C], minval=0, maxval=1, dtype=tf.float32, name="pred")

    y_true = tf.convert_to_tensor(
                [[[
                [0, 0, 1, 1, 1,                   
                0, 0, 1, 1, 1,                   
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 1, 1, 1,                   
                0, 0, 1, 1, 1,                   
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ],

                [[0, 0, 0, 0, 0,                   
                0, 0, 0, 0, 0,                   
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0,                   
                0, 0, 0, 0, 0,                   
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                ]], dtype=tf.float32)

    loss = Yolo_Loss_v1()
    loss(y_true, y_pred)
if __name__ == "__main__":
    testShape1()
    
