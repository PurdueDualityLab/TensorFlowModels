import tensorflow as tf
import math

class Darknet_Classification_LR(tf.keras.callbacks.Callback):
  def __init__(self, schedule):
    super(Darknet_Classification_LR, self).__init__()
    self.steps_schedule = schedule
    self.steps = 0
    self.lr = 0

  def on_train_batch_begin(self,batch,logs=None):
    if not hasattr(self.model.optimizer, "lr"):
      raise ValueError('Optimizer must have a "lr" attribute.')
    # Get the current learning rate from model's optimizer.
    if batch == 0:
      self.lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

    # Call schedule function to get the scheduled learning rate.
    scheduled_lr = self.steps_schedule(self.steps, self.lr)
    # Set the value back to the optimizer before this epoch starts
    tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
    self.steps += 1

config = {
"burn_rate":1000.0,
"max_batches":800000,  
"power":4  
}

def poly_schedule(batch_num,lr, burn):
  if batch_num == 0:
    return lr
  batch_num = float(batch_num)
  if batch_num < config["burn_rate"]:
    return lr * float(math.pow((batch_num/config["burn_rate"]),4))
  return lr * math.pow((1-batch_num / config["max_batches"]),config["power"])