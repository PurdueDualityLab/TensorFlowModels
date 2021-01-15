import tensorflow.keras as ks


class HistoryFull(ks.callbacks.Callback):

  def __init__(self):
    super().__init__()
    self._lr = []
    self._loss_list = []
    self._acc_list = []

  def on_train_batch_end(self, batch, logs):
    self._loss_list.append(logs["loss"])
    # self._acc_list.append(logs["categorical_accuracy"])
    self._lr.append(logs["lr"])

  def get_logs(self, *args):
    return self._loss_list, self._acc_list, self._lr
