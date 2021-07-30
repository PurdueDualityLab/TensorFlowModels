from logging import raiseExceptions
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.training import gen_training_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils

import tensorflow as tf

__all__ = ['SGD']

class _UnwrapPreventer(object):
  """Wrapper that DistributionStrategy will not unwrap.
  Typically, DistributionStrategy will unwrap values when going from a cross-
  replica context to a replica context via `call_for_each_replica`. This class
  is a wrapper that DistributionStrategy will not unwrap, so it can be used to
  prevent it from unwrapping a value.
  TODO(reedwm): Find/implement a better way of preventing values from being
  unwrapped by DistributionStrategy
  """

  __slots__ = ['value']

  def __init__(self, value):
    self.value = value

@keras_export("keras.optimizers.SGD")
class ScaledYoloSGD(optimizer_v2.OptimizerV2):
  r"""Gradient descent (with momentum) optimizer.
  Update rule for parameter `w` with gradient `g` when `momentum` is 0:
  ```python
  w = w - learning_rate * g
  ```
  Update rule when `momentum` is larger than 0:
  ```python
  velocity = momentum * velocity - learning_rate * g
  w = w + velocity
  ```
  When `nesterov=True`, this rule becomes:
  ```python
  velocity = momentum * velocity - learning_rate * g
  w = w + momentum * velocity - learning_rate * g
  ```
  Args:
    learning_rate: A `Tensor`, floating point value, or a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
      that takes no arguments and returns the actual value to use. The
      learning rate. Defaults to 0.01.
    momentum: float hyperparameter >= 0 that accelerates gradient descent
      in the relevant
      direction and dampens oscillations. Defaults to 0, i.e., vanilla gradient
      descent.
    nesterov: boolean. Whether to apply Nesterov momentum.
      Defaults to `False`.
    name: Optional name prefix for the operations created when applying
      gradients.  Defaults to `"SGD"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.
  Usage:
  >>> opt = tf.keras.optimizers.SGD(learning_rate=0.1)
  >>> var = tf.Variable(1.0)
  >>> loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
  >>> step_count = opt.minimize(loss, [var]).numpy()
  >>> # Step is `- learning_rate * grad`
  >>> var.numpy()
  0.9
  >>> opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
  >>> var = tf.Variable(1.0)
  >>> val0 = var.value()
  >>> loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
  >>> # First step is `- learning_rate * grad`
  >>> step_count = opt.minimize(loss, [var]).numpy()
  >>> val1 = var.value()
  >>> (val0 - val1).numpy()
  0.1
  >>> # On later steps, step-size increases because of momentum
  >>> step_count = opt.minimize(loss, [var]).numpy()
  >>> val2 = var.value()
  >>> (val1 - val2).numpy()
  0.18
  Reference:
      - For `nesterov=True`, See [Sutskever et al., 2013](
        http://jmlr.org/proceedings/papers/v28/sutskever13.pdf).
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.01,
               momentum=0.0,
               nesterov=False,
               name="SGD",
               **kwargs):
    super(ScaledYoloSGD, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("decay", self._initial_decay)
    self._names = None

    self._momentum = False
    if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
      self._momentum = True
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError("`momentum` must be between [0, 1].")
    self._set_hyper("momentum", momentum)

    self.nesterov = nesterov

    self._weight_up = False 
    self._biases_up = False 
    self._other_up = False

  def _set_model_names(self, names):
    self._names = names
    return

  def _create_slots(self, var_list):
    if self._momentum:
      for var in var_list:
        self.add_slot(var, "momentum")

  def _momentum_t(self, var_dtype):
    """Get decayed learning rate as a Tensor with dtype=var_dtype."""
    momentum_t = self._get_hyper("momentum", var_dtype)
    if isinstance(momentum_t, learning_rate_schedule.LearningRateSchedule):
      local_step = tf.cast(self.iterations, var_dtype)
      momentum_t = tf.cast(momentum_t(local_step), var_dtype)
    return momentum_t

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(ScaledYoloSGD, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(
        self._momentum_t(var_dtype))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    if self._momentum:
      momentum_var = self.get_slot(var, "momentum")
      return gen_training_ops.ResourceApplyKerasMomentum(
          var=var.handle,
          accum=momentum_var.handle,
          lr=coefficients["lr_t"],
          grad=grad,
          momentum=coefficients["momentum"],
          use_locking=self._use_locking,
          use_nesterov=self.nesterov)
    else:
      return gen_training_ops.ResourceApplyGradientDescent(
          var=var.handle,
          alpha=coefficients["lr_t"],
          delta=grad,
          use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # This method is only needed for momentum optimization.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    momentum_var = self.get_slot(var, "momentum")
    return gen_training_ops.ResourceSparseApplyKerasMomentum(
        var=var.handle,
        accum=momentum_var.handle,
        lr=coefficients["lr_t"],
        grad=grad,
        indices=indices,
        momentum=coefficients["momentum"],
        use_locking=self._use_locking,
        use_nesterov=self.nesterov)

  def get_config(self):
    config = super(ScaledYoloSGD, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "decay": self._initial_decay,
        "momentum": self._serialize_hyperparameter("momentum"),
        "nesterov": self.nesterov,
    })
    return config

  def apply_gradients(self, grads_and_vars, name = None, experimental_aggregate_gradients=True):
    
    grads_and_vars = tuple(grads_and_vars)
    grads = [g for g, _ in grads_and_vars]
    wrapped_vars = _UnwrapPreventer([v for _, v in grads_and_vars])

    bias = []
    weights = []
    other = []
    for var, grad, name in zip(wrapped_vars.value, grads, self._names):
      if "bias" in name:
        bias.append((grad, var))
      elif "beta" in name:
        bias.append((grad, var))
      elif "kernel" in name or "weight" in name:
        weights.append((grad, var))
      else:
        other.append((grad, var))


    super().apply_gradients(weights, experimental_aggregate_gradients = experimental_aggregate_gradients, name = name)
    super().apply_gradients(bias, experimental_aggregate_gradients = experimental_aggregate_gradients, name = name)
    super().apply_gradients(other, experimental_aggregate_gradients = experimental_aggregate_gradients, name = name)
    return #super().apply_gradients(zipped, experimental_aggregate_gradients = experimental_aggregate_gradients, name = name)


# if __name__ == "__main__":
#   from yolo import run
#   import os
#   optimizer = SGDAccumulated(accumulation_steps = 8)

#   config = [os.path.abspath('yolo/configs/experiments/yolov4-eval.yaml')]
#   model_dir = "" #os.path.abspath("../checkpoints/yolo_dt8_norm_iou")

#   task, model, params = run.load_model(experiment='yolo_custom', config_path=config, model_dir=model_dir)

#   train_data = task.build_inputs(task.task_config.train_data)
#   validation_data = task.build_inputs(task.task_config.train_data)

#   model.compile(optimizer = optimizer)
