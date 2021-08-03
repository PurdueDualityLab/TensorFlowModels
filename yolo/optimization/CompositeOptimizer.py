from tensorflow.python.keras.optimizer_v2 import optimizer_v2
#try:
from tensorflow_recommenders.experimental.optimizers import CompositeOptimizer as co
# except:
#   from tensorflow_recommenders.optimizers import CompositeOptimizer as co

import tensorflow as tf

__all__ = ['SGD']


# problem is that sub division cannot change between saves
class CompositeOptimizer(co):

  # def __init__(self, 
  #              optimizers_and_vars,
  #              name = None) -> None:
  #   super().__init__(optimizers_and_vars, name=name)

  @property
  def learning_rate(self):
    optimizers = self.optimizers
    return {optimizer.name: optimizer.learning_rate(
                            self.iterations) for optimizer in optimizers}
