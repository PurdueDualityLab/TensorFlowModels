import tensorflow.keras as ks
import tensorflow.keras.backend as K


class LearningRateScheduler(ks.callbacks.Callback):

  def __init__(self, schedule, verbose=0):
    super(LearningRateScheduler, self).__init__()
    self.schedule = schedule
    self.verbose = verbose

  def on_train_batch_begin(self, batch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    lr = float(K.get_value(self.model.optimizer.lr))
    lr, global_step = self.schedule.run()
    self._batches = global_step
    K.set_value(self.model.optimizer.lr, K.get_value(lr))
    if self.verbose > 0:
      print(
          '\nEpoch %05d: LearningRateScheduler reducing learning rate to %s.' %
          (self._batches + 1, lr))

  def on_train_batch_end(self, batch, logs=None):
    logs = logs or {}
    logs['lr'] = K.get_value(self.model.optimizer.lr)
    logs['batch_step'] = self._batches
    string = ['%s: %0.5f\t' % (key, logs[key]) for key in logs.keys()]
    print(''.join(string), end='\r', flush=True)

  def on_epoch_end(self, batch, logs=None):
    logs = logs or {}
    logs['lr'] = K.get_value(self.model.optimizer.lr)
    logs['batch_step'] = self._batches
    try:
      string = ['%s: %0.5f\t' % (key, logs[key]) for key in logs.keys()]
      print('\n', ''.join(string), end='\n\n', flush=True)
    except Exception as e:
      print(e)
