import tensorflow as tf


class BatchLRScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule, **kwargs):
        super().__init__(**kwargs)
        self._schedule = schedule

        return

    def on_batch_begin(self, batch, logs=None):

        return
