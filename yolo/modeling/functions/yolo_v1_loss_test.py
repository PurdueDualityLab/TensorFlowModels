import tensorflow as tf
from yolo.modeling.functions.yolo_v1_loss import Yolo_Loss_v1

def makeRandomPrediction(batchSize, numCells, numBoxes, numClasses):
    # Creates a random tensor as shape [batchSize, numCells, numBoxes, numClasses]
    # to simulate the model output.

    y_pred = tf.random.uniform([batchSize, numCells, numCells, numBoxes*5 + numClasses], 
                                minval=0, maxval=1, dtype=tf.float32, name="pred")

    return y_pred

def makeRandomGroundTruth(batchSize, numCells, numBoxes, numClasses):
    # Creates a random tensor as shape [batchSize, numCells, numBoxes, numClasses]
    # to simulate the ground truth. Note that the confidence for each cell takes on
    # values {0, 1} randomly and the classes are one-hot encoded.

    y_true_xywh = tf.random.uniform([batchSize, numCells, numCells, 4], 
                                    minval=0, maxval=1, dtype=tf.float32, name="true")
    y_true_c = tf.cast(tf.random.uniform([batchSize, numCells, numCells, 1], 
                       minval=0, maxval=2, dtype=tf.int32, name="true"), tf.float32)
    y_true_xywhc = tf.concat([y_true_xywh, y_true_c], axis=-1)

    y_true_class = tf.random.uniform([batchSize, numCells, numCells, numClasses], 
                                     minval=0, maxval=1, dtype=tf.float32, name="true")
    
    y_true_class = tf.math.divide_no_nan(y_true_class, tf.reduce_max(y_true_class, axis=-1, keepdims=True))
    y_true = tf.concat([y_true_xywhc, y_true_xywhc, y_true_class], axis=-1)

    return y_true

def testLossOnShape(batchSize, numCells, numBoxes, numClasses):
    tf.print("=-" * 50)
    tf.print("Testing loss function on parameters:\nBatch Size: {}\nNum Grids: {}x{}\nNum Boxes: {}\nNum Classes: {}\n" \
    .format(batchSize, numCells, numCells, numBoxes, numClasses))
    y_pred = makeRandomPrediction(batchSize, numCells, numBoxes, numClasses)
    y_true = makeRandomGroundTruth(batchSize, numCells, numBoxes, numClasses)

    tf.print("Prediction Shape: ", tf.shape(y_pred))
    tf.print("Ground Truth Shape: ", tf.shape(y_true))
    loss = Yolo_Loss_v1()

    tf.print("Calculated Loss: ", loss(y_true, y_pred))
    tf.print("=-" * 50)


if __name__ == "__main__":
    testLossOnShape(1, 7, 2, 20)
    testLossOnShape(10, 7, 2, 20)
    testLossOnShape(64, 7, 2, 20)
    testLossOnShape(100, 7, 2, 20)

    
