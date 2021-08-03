# #try:
# from tensorflow_recommenders.experimental.optimizers import CompositeOptimizer as co
# except:
#   from tensorflow_recommenders.optimizers import CompositeOptimizer as co

import tensorflow as tf

import collections
from typing import Callable, List, Optional, Sequence, Tuple, Union

import tensorflow as tf
from tensorflow.keras import optimizers

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]


class CompositeOptimizer(tf.keras.optimizers.Optimizer):
  """An optimizer that composes multiple individual optimizers.
  It allows different optimizers to be applied to different subsets of the
  model's variables. For example, it makes it possible to apply one
  optimizer to the model's embeddings (sparse variables) and another
  optimizer to the rest of its variables.
  To specify which optimizer should apply to each variable, pass a list of
  pairs of (optimizer instance, function returning a list of variables the
  optimizer should apply to).
  For example:
  ```python
    optimizer = CompositeOptimizer([
        (tf.keras.optimizers.SGD(), lambda: model.sparse_trainable_variables),
        (tf.keras.optimizers.Adam(), lambda: model.dense_trainable_variables),
    ])
  ```
  """

  def __init__(self,
               optimizers_and_vars: Sequence[
                   Tuple[tf.keras.optimizers.Optimizer,
                         Callable[[], Sequence[tf.Variable]]]],
               name: str = "CompositeOptimizer") -> None:
    """Initializes an CompositeOptimizer instance.
    Args:
      optimizers_and_vars:  List of tuples of (optimizer instance, function
        returning variables that the optimizer should apply to).
      name: The optimizer name.
    """
    super().__init__(name=name)
    if not optimizers_and_vars:
      raise ValueError("`optimizers_and_vars` can't be empty")
    self._optimizers_and_vars = optimizers_and_vars
    for i, optimizer_and_var in enumerate(optimizers_and_vars):
      optimizer = optimizer_and_var[0]
      self._track_trackable(optimizer, name=str(i))

  def apply_gradients(self, grads_and_vars: Sequence[Tuple[Tensor, Tensor]],
                      name: Optional[str] = None) -> None:
    """See base class."""
    var_optimizer_dict = {}

    for optimizer, var_callable in self._optimizers_and_vars:
      for v in var_callable():
        if v.ref() in var_optimizer_dict:
          raise ValueError(
              "The set of variables handled by each optimizer should be \
                disjoint, but variable is handled both 2 optimizers")
        var_optimizer_dict[v.ref()] = optimizer

    optimizer_grads_and_vars = collections.defaultdict(list)
    for g, v in grads_and_vars:
      if v.ref() in var_optimizer_dict:
        optimizer = var_optimizer_dict[v.ref()]
        optimizer_grads_and_vars[optimizer].append((g, v))
      else:
        raise ValueError("Variable is not handled by any optimizer. \
                          This would cause it to be not trained.")

    for optimizer, opt_grads_and_vars in optimizer_grads_and_vars.items():
      optimizer.apply_gradients(opt_grads_and_vars, name=name)

  def get_config(self):
    # raise NotImplementedError("CompositeOptimizer cannot be serialized because"
    #                           " it uses callable to get variables.")
    return {}

  @property
  def iterations(self):
    """See base class."""
    # Returning iterations from the first optimizer.
    return self._optimizers_and_vars[0][0].iterations

  @iterations.setter
  def iterations(self, variable):
    """See base class."""
    for optimizer, _ in self._optimizers_and_vars:
      optimizer.iterations = variable

  def variables(self):
    """Returns the optimizer's variables."""
    # OptimizerV2.variables() returns self._weights, so override that method.
    return self.weights

  @property
  def weights(self) -> List[tf.Variable]:
    """Returns the optimizer's variables."""
    weights = []
    for optimizer, _ in self._optimizers_and_vars:
      weights += optimizer.weights
    return weights

  @property
  def optimizers(self) -> List[tf.keras.optimizers.Optimizer]:
    """Returns the optimizers in composite optimizer (in the original order)."""
    return [optimizer for optimizer, _ in self._optimizers_and_vars]

  @property
  def learning_rate(self):
    optimizers = self.optimizers
    return {optimizer.name: optimizer.learning_rate(
                            self.iterations) for optimizer in optimizers}