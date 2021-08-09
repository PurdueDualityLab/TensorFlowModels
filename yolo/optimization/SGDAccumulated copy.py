import tensorflow as tf
from tensorflow.python.keras.backend import zeros_like
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.ops import math_ops, state_ops, control_flow_ops, array_ops
from tensorflow.python import ops
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
# from tensorflow.python.keras.utils import control_flow_util
# from tensorflow.python.keras import backend_config

__all__ = ['SGDAccumulated']


# problem is that sub division cannot change between saves
class SGDAccumulated(OptimizerV2):
  """Optimizer that implements the Adam algorithm with gradient accumulation."""

  def __init__(self,
               accumulation_steps=1,
               accumulation_type='mean',
               learning_rate=0.01,
               momentum=0.0,
               nesterov=False,
               one_offset=False, 
               adjusted_for_accum = True,

               momentum_start=0.0, 
               warmup_steps=1000,
               name="SGD",
               **kwargs):
    r"""Construct a new SGD optimizer.
    Args:
        accumulation_steps: An integer. Update gradient in every accumulation steps.
        learning_rate: A Tensor or a floating point value.    The learning rate.
        beta_1: A float value or a constant float tensor. The exponential decay
            rate for the 1st moment estimates.
        beta_2: A float value or a constant float tensor. The exponential decay
            rate for the 2nd moment estimates.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
        amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from
            the paper "On the Convergence of Adam and beyond".
        name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".    @compatibility(eager) When eager execution is
            enabled, `learning_rate`, `beta_1`, `beta_2`, and `epsilon` can each be
            a callable that takes no arguments and returns the actual value to use.
            This can be useful for changing these values across different
            invocations of optimizer functions. @end_compatibility
        **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for backward
            compatibility, recommended to use `learning_rate` instead.
    """

    super(SGDAccumulated, self).__init__(name, **kwargs)
    self._set_hyper('accumulation_steps', tf.cast(accumulation_steps, tf.int32))
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper("decay", self._initial_decay)
    self._set_hyper("update", tf.cast(0.0, tf.float32))
    self._momentum = False
    if isinstance(momentum, tf.Tensor) or callable(momentum) or momentum > 0:
      self._momentum = True
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError("`momentum` must be between [0, 1].")
    
    self._set_hyper("momentum", momentum)
    self._set_hyper("momentum_start", momentum_start)

    if adjusted_for_accum: 
      self._set_hyper("warmup_steps", tf.cast(warmup_steps, tf.int32))
    else:
      self._set_hyper("warmup_steps", tf.cast(warmup_steps * accumulation_steps, tf.int32))

    self.nesterov = nesterov

    self._offset = tf.cast(0, tf.int64)
    if one_offset:
      self._offset = tf.cast(1, tf.int64)

    self._accumulation_type = accumulation_type


  def _get_momentum(self, iteration):
    momentum = self._get_hyper("momentum")
    momentum_start = self._get_hyper("momentum_start")
    momentum_warm_up_steps = tf.cast(
        self._get_hyper("warmup_steps"), iteration.dtype)

    wm_momentum = (momentum_start +
                         (tf.cast(iteration, momentum.dtype) *
                          (momentum - momentum_start) / tf.cast(
                              momentum_warm_up_steps, momentum.dtype)))
    value = tf.cond(
        (iteration - momentum_warm_up_steps) < 0,
        true_fn=lambda: wm_momentum,
        false_fn=lambda: momentum)
    return value

  def _get_accumulation_steps(self, iteration):
    ac = tf.cast(self._get_hyper("accumulation_steps"), tf.float64)
    acs = tf.cast(0, ac.dtype)
    momentum_warm_up_steps = tf.cast(
        self._get_hyper("warmup_steps"), iteration.dtype)

    wm_ac = (acs +
                tf.cast((tf.cast(iteration, ac.dtype) *
                          (ac - acs) / tf.cast(
                              momentum_warm_up_steps, ac.dtype)), ac.dtype))
    base = tf.cast(1, tf.int64)
    ac = tf.maximum(base, tf.cast(ac, tf.int64))
    wm_ac = tf.maximum(base, tf.cast(wm_ac, tf.int64))
    value = tf.cond(
        (iteration - momentum_warm_up_steps) < 0,
        true_fn=lambda: wm_ac,
        false_fn=lambda: ac)
    return value

  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var, 'g')
    if self._momentum:
      for var in var_list:
        self.add_slot(var, "momentum")

  def momentum_update(self, coefficients, var, grad):
    momentum_var = self.get_slot(var, "momentum")
    return tf.raw_ops.ResourceApplyKerasMomentum(
        var=var.handle,
        accum=momentum_var.handle,
        lr=coefficients["lr_t"],
        grad=grad,
        momentum=coefficients["momentum"],
        use_locking=self._use_locking,
        use_nesterov=self.nesterov)

  def no_momentum_update(self, coefficients, var, grad):
    return tf.raw_ops.ResourceApplyGradientDescent(
        var=var.handle,
        alpha=coefficients["lr_t"],
        delta=grad,
        use_locking=self._use_locking)


  def raw_update(self, coefficients, var, grad):
    def func():
      # tf.print("up")
      if self._momentum:
        return self.momentum_update(coefficients, var, grad)
      else:
        return self.no_momentum_update(coefficients, var, grad)
    return func

  def no_update(self, coefficients, var, grad):
    def func():
      # tf.print("no up")
      var_update = state_ops.assign(var, var, use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update])
    return func

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(SGDAccumulated, self)._prepare_local(var_device, var_dtype,
                                               apply_state)
    apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(
        self._get_hyper("momentum", var_dtype))
    
    if self._momentum:
      momentum = self._get_momentum(self.iterations)
      momentum = tf.cast(momentum, var_dtype)
      apply_state[(var_device,
                   var_dtype)]["momentum"] = array_ops.identity(momentum)
    
    accumulation_steps = self._get_accumulation_steps(self.iterations)
    accumulation_steps = array_ops.identity(accumulation_steps)
    if self._accumulation_type == "sum":
      accumulation_steps = tf.cast(1, var_dtype)
    
    apply_state[(var_device, var_dtype)]["accumulation_steps"] = accumulation_steps
    apply_state[(var_device, var_dtype)]["update"] = tf.cast((self.iterations + self._offset) % tf.cast(accumulation_steps, self.iterations.dtype) == 0, var_dtype)


  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    grad = grad/coefficients["accumulation_steps"]
    return tf.cond(coefficients["update"] == 1, 
                   self.raw_update(coefficients, var, grad), 
                   self.no_update(coefficients, var, grad))

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # This method is only needed for momentum optimization.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    grad = grad/coefficients["accumulation_steps"]
    momentum_var = self.get_slot(var, "momentum")
    return tf.raw_ops.ResourceSparseApplyKerasMomentum(
        var=var.handle,
        accum=momentum_var.handle,
        lr=coefficients["lr_t"],
        grad=grad,
        indices=indices,
        momentum=coefficients["momentum"],
        use_locking=self._use_locking,
        use_nesterov=self.nesterov)

  def get_config(self):
    config = super(SGDAccumulated, self).get_config()
    config.update({
        'accumulation_steps':
            self._serialize_hyperparameter('accumulation_steps'),
        'learning_rate':
            self._serialize_hyperparameter('learning_rate'),
        'decay':
            self._serialize_hyperparameter('decay'),
        'momentum':
            self._serialize_hyperparameter('momentum'),
    })
    return config


  def apply_gradients(self, grads_and_vars, 
                      name=None,
                      experimental_aggregate_gradients=True):
    grads_and_vars = optimizer_utils.filter_empty_gradients(grads_and_vars)
    var_list = [v for (_, v) in grads_and_vars]

    with tf.name_scope(self._name):
      # Create iteration if necessary.
      with tf.init_scope():
        self._create_all_weights(var_list)

    accumulation_steps = self._get_accumulation_steps(self.iterations)
    momentum = self._get_momentum(self.iterations)
    #self._get_hyper('accumulation_steps', 'int64')

    tf.print(accumulation_steps, momentum, self.iterations)

    ng = []
    ag = []
    for grad, var in grads_and_vars:
      tf.print(var.ref())
      g = self.get_slot(var, 'g') # accumulated gradient
      g_a = grad + g
      g_a = state_ops.assign(g, g_a)    
      ng.append(tf.zeros_like(grad))
      ag.append(g_a)

    grad_list = ng
    if (self.iterations + self._offset) % accumulation_steps == 0:
      grad_list = ag

    super().apply_gradients(zip(grad_list,var_list), 
                            name=name, 
                            experimental_aggregate_gradients=experimental_aggregate_gradients)

    if (self.iterations + self._offset) % accumulation_steps == 0:
      for grad, var in grads_and_vars:
        g = self.get_slot(var, 'g') # accumulated gradient
        g_a = state_ops.assign(g, tf.zeros_like(grad))    


    return
