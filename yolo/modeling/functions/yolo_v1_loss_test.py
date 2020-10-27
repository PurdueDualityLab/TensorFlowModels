import tensorflow as tf
import tensorflow.keras as ks
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

from yolo.modeling.functions.yolo_v1_loss import Yolo_Loss_v1
from yolo.modeling.functions.build_gridded_gt import build_gridded_gt_v1

def makeRandomPrediction(batchSize, numCells, numBoxes, numClasses, seed=36835):
    # Creates a random tensor as shape [batchSize, numCells, numBoxes, numClasses]
    # to simulate the model output.

    y_pred = tf.random.uniform([batchSize, numCells, numCells, numBoxes*5 + numClasses], 
                                minval=0, maxval=1, dtype=tf.float32, name="pred", seed=seed)
    return y_pred

def makeRandomGroundTruth(batchSize, numCells, numBoxes, numClasses, seed=94820):
    # Creates a random tensor as shape [batchSize, numCells, numBoxes, numClasses]
    # to simulate the ground truth. Note that the confidence for each cell takes on
    # values {0, 1} randomly and the classes are one-hot encoded.

    y_true_xywh = tf.random.uniform([batchSize, numCells, numCells, 4], 
                                    minval=0, maxval=1, dtype=tf.float32, name="true", seed=seed)
    y_true_c = tf.cast(tf.random.uniform([batchSize, numCells, numCells, 1], 
                       minval=0, maxval=2, dtype=tf.int32, name="true", seed=seed), tf.float32)
    y_true_xywhc = tf.concat([y_true_xywh, y_true_c], axis=-1)

    y_true_class = tf.random.uniform([batchSize, numCells, numCells, numClasses], 
                                     minval=0, maxval=1, dtype=tf.float32, name="true", seed=seed)
    y_true_class = tf.math.divide_no_nan(y_true_class, tf.reduce_max(y_true_class, axis=-1, keepdims=True))

    y_true_components = []
    for _ in range(numBoxes):
        y_true_components.append(y_true_xywhc)
    
    y_true_components.append(y_true_class)
    y_true = tf.concat(y_true_components, axis=-1)

    return y_true

def testLossOnShape(batchSize, numCells, numBoxes, numClasses):
    # Tests loss function on different shaped inputs as specified by
    # batchSize, numCells, numBoxes, and numClasses.
    tf.print("=-" * 50)
    tf.print("Testing loss function on parameters:\nBatch Size: {}\nNum Grids: {}x{}\nNum Boxes: {}\nNum Classes: {}\n" \
    .format(batchSize, numCells, numCells, numBoxes, numClasses))
    y_pred = makeRandomPrediction(batchSize, numCells, numBoxes, numClasses)
    y_true = makeRandomGroundTruth(batchSize, numCells, numBoxes, numClasses)

    tf.print("Prediction Shape: ", tf.shape(y_pred))
    tf.print("Ground Truth Shape: ", tf.shape(y_true))
    loss = Yolo_Loss_v1(num_boxes=numBoxes, num_classes=numClasses)

    tf.print("Calculated Loss: ", loss(y_true, y_pred))
    tf.print("=-" * 50)

def testCompile():
    # Tests loss function with model.compile() with trivial model
    tf.print("=-" * 50)
    tf.print("Testing loss function on model.compile()")
    loss = Yolo_Loss_v1(num_boxes=2, num_classes=20)

    inputs = tf.keras.layers.Input(shape=(3,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    while True:
        try:
            model.compile(optimizer="Adam", loss=loss, metrics=["mae"])
            break
        except ValueError:
            tf.print("Loss function does not compile!")
            return

    tf.print("Loss function compiles!") 

def gt_test():
    DATASET_DIRECTORY = "D:\Datasets" # modify to select download location
    train, info = tfds.load('voc', 
                      split='train', 
                      shuffle_files=False, 
                      with_info=True, 
                      data_dir=DATASET_DIRECTORY)

    train.batch(batch_size=1)
    train = train.enumerate()

    i = 0
    for element in train.as_numpy_iterator():
        print(element)
        data = element[1] # gives dict with keys: image, image/filename, image/id, objects
        y_true = data['objects'] # gives dict with keys: area, bbox, id, is_crowd, label

        tf.print("YTRUE BEFORE GT: ", y_true)
        y_true_after_gt = build_gridded_gt_v1(y_true, num_classes=20, size=7,
                        num_boxes=2, dtype=tf.float64)
        tf.print(y_true_after_gt)
        i += 1
        if i == 1:
            break;

    # prediction = makeRandomPrediction(batchSize=1, numCells=7, numBoxes=2, numClasses=20)

    # tf.print("Output of ground truth builder shape: ", y_true_after_gt.get_shape())
    # tf.print("Prediction shape: ", prediction.get_shape())

if __name__ == "__main__":
    # tf.print("\n\nTesting shapes:")
    # testLossOnShape(1, 7, 2, 20)
    # testLossOnShape(10, 7, 2, 20)
    # testLossOnShape(64, 7, 2, 20)

    # testLossOnShape(64, 5, 2, 20)

    # testLossOnShape(64, 7, 2, 20)

    # testLossOnShape(64, 7, 3, 20)

    # testCompile()
    gt_test()
