import tensorflow as tf


def detection_lr(epoch, learning_rate):
    if epoch == 60 or epoch == 90:
        learning_rate /= 10
    return learning_rate


def classification_lr(epoch, learning_rate):
    return learning_rate


def get_callbacks():
    return
