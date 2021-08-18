from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.training import gen_training_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow_addons.optimizers import DecoupledWeightDecayExtension

import tensorflow as tf

__all__ = ['SGD']

# problem is that sub division cannot change between saves
class SGDMomentumWarmupW(optimizer_v2.OptimizerV2):
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
               weight_decay = 0.0,
               learning_rate=0.01,
               momentum=0.0,
               momentum_start=0.0,
               warmup_steps=1000,
               nesterov=False,
               sim_torch=False, 
               name="SGD",
               **kwargs):
    super(SGDMomentumWarmupW, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("bias_learning_rate", kwargs.get("lr", learning_rate))

    self._set_hyper("decay", self._initial_decay)
    self._set_hyper("weight_decay", weight_decay)
    print(weight_decay)
    self._weight_decay = weight_decay != 0.0

    self._momentum = False
    if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
      self._momentum = True
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError("`momentum` must be between [0, 1].")

    self._set_hyper("momentum", momentum)
    self._set_hyper("momentum_start", momentum_start)
    self._set_hyper("warmup_steps", tf.cast(warmup_steps, tf.int32))
    self.nesterov = nesterov
    self.sim_torch = sim_torch

  def _set_bias_lr(self, lr, bias_key):
    self._LR_bias_depth = bias_key
    self._set_hyper("bias_learning_rate", lr)

  def _get_momentum(self, iteration):
    momentum = self._get_hyper("momentum")
    momentum_start = self._get_hyper("momentum_start")
    momentum_warm_up_steps = tf.cast(
        self._get_hyper("warmup_steps"), iteration.dtype)
    value = tf.cond(
        (iteration - momentum_warm_up_steps) < 0,
        true_fn=lambda: (momentum_start +
                         (tf.cast(iteration, momentum.dtype) *
                          (momentum - momentum_start) / tf.cast(
                              momentum_warm_up_steps, momentum.dtype))),
        false_fn=lambda: momentum)
    return value

  def _create_slots(self, var_list):
    if self._momentum:
      for var in var_list:
        self.add_slot(var, "momentum")

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(SGDMomentumWarmupW, self)._prepare_local(var_device, var_dtype,
                                                  apply_state)
    weight_decay = self._get_hyper("weight_decay")
    apply_state[(var_device,
                   var_dtype)]["weight_decay"] = tf.cast(weight_decay, var_dtype)

    if self._momentum:
      momentum = self._get_momentum(self.iterations)
      momentum = tf.cast(momentum, var_dtype)
      apply_state[(var_device,
                   var_dtype)]["momentum"] = array_ops.identity(momentum)

    bias_lr = self._get_hyper("bias_learning_rate")
    if isinstance(bias_lr, learning_rate_schedule.LearningRateSchedule):
      bias_lr = bias_lr(self.iterations)

    bias_lr = tf.cast(bias_lr, var_dtype)
    apply_state[(var_device,
                 var_dtype)]["bias_lr"] = array_ops.identity(bias_lr)
    return apply_state[(var_device, var_dtype)]

  def _apply(self, grad, var, coefficients):
    dparams = grad
    groups = []
    
    if self._weight_decay:
      weight_decay = coefficients["weight_decay"]
      dparams += (weight_decay * var)
    
    if self._momentum:
      momentum_var = self.get_slot(var, "momentum")
      momentum = coefficients["momentum"]
      momentum_update = momentum_var.assign(
        momentum * momentum_var + dparams, use_locking=self._use_locking)
      groups.append(momentum_update)

      if self.nesterov:
        dparams += (momentum * momentum_update)
      else:
        dparams = momentum_update

    lr = coefficients["lr_t"]
    weight_update = var.assign_add(-lr * dparams, use_locking=self._use_locking)
    groups.append(weight_update)
    return tf.group(*groups)

  def _apply_tf(self, grad, var, coefficients):
    lr = coefficients["lr_t"]

    def decay_op(var, learning_rate, apply_state):
      if self._weight_decay:
        return var.assign_sub(
            learning_rate * var *
            apply_state['weight_decay'],
            use_locking=self._use_locking)
      return tf.no_op()
    
    decay = decay_op(var, lr, coefficients)
    with tf.control_dependencies([decay]):
      if self._momentum:
        momentum_var = self.get_slot(var, "momentum")
        return gen_training_ops.ResourceApplyKerasMomentum(
            var=var.handle,
            accum=momentum_var.handle,
            lr=lr,
            grad=grad,
            momentum=coefficients["momentum"],
            use_locking=self._use_locking,
            use_nesterov=self.nesterov)
      else:
        return gen_training_ops.ResourceApplyGradientDescent(
            var=var.handle,
            alpha=lr,
            delta=grad,
            use_locking=self._use_locking)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))
    if self.sim_torch:
      return self._apply(grad, var, coefficients)
    else:
      return self._apply_tf(grad, var, coefficients)


  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices,
                                               **kwargs):
    if self._momentum:
      return super(SGDMomentumWarmupW,
                   self)._resource_apply_sparse_duplicate_indices(
                       grad, var, indices, **kwargs)
    else:
      var_device, var_dtype = var.device, var.dtype.base_dtype
      coefficients = (
          kwargs.get("apply_state", {}).get((var_device, var_dtype)) or
          self._fallback_apply_state(var_device, var_dtype))

      if self.sim_torch:
        return self._apply(grad, var, coefficients)
      else:
        return self._apply_tf(grad, var, coefficients)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # This method is only needed for momentum optimization.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))
    if self.sim_torch:
      return self._apply(grad, var, coefficients)
    else:
      return self._apply_tf(grad, var, coefficients)


  def get_config(self):
    config = super(SGDMomentumWarmupW, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "decay": self._initial_decay,
        "momentum": self._serialize_hyperparameter("momentum"),
        "momentum_start": self._serialize_hyperparameter("momentum_start"),
        "warmup_steps": self._serialize_hyperparameter("warmup_steps"),
        "nesterov": self.nesterov,
    })
    return config