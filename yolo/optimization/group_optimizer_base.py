from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.training import gen_training_ops

import tensorflow as tf
import re
import logging

try:
  from keras.optimizer_v2.optimizer_v2 import _var_key
except:
  def _var_key(var):
    """Key for representing a primary variable, for looking up slots.
    In graph mode the name is derived from the var shared name.
    In eager mode the name is derived from the var unique id.
    If distribution strategy exists, get the primary variable first.
    Args:
      var: the variable.
    Returns:
      the unique name of the variable.
    """

    # pylint: disable=protected-access
    # Get the distributed variable if it exists.
    if hasattr(var, "_distributed_container"):
      var = var._distributed_container()
    if var._in_graph_mode:
      return var._shared_name
    return var._unique_id


class GroupOpt(tf.keras.optimizers.Optimizer):
  """Optimizer that simulates the SGD module used in pytorch. 
  
  
  For details on the differences between the original SGD implemention and the 
  one in pytorch: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html.
  This optimizer also allow for the usage of a momentum warmup along side a 
  learning rate warm up, though using this is not required. 
  Example of usage for training:
  ```python
  opt = SGDTorch(learning_rate, weight_decay = 0.0001)
  l2_regularization = None
  # iterate all model.trainable_variables and split the variables by key 
  # into the weights, biases, and others.
  optimizer.search_and_set_variable_groups(model.trainable_variables)
  # if the learning rate schedule on the biases are different. if lr is not set 
  # the default schedule used for weights will be used on the biases. 
  opt.set_bias_lr(<lr schedule>)
  # if the learning rate schedule on the others are different. if lr is not set 
  # the default schedule used for weights will be used on the biases. 
  opt.set_other_lr(<lr schedule>)
  ```
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               lr = 0.001, # default params if not specified in a group
               momentum = 0.9, # default params if not specified in a group
               weight_decay = 0.01, # default params if not specified in a group
               groups = None, 
               name="SGD",
               **kwargs):
    super(GroupOpt, self).__init__(name, **kwargs)

    # all groups must have the SAME keys.
    groups = groups or [  #]
      {
        "keys": ["kernel", "weights"], # weights
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.005,
        "name": "weights"
      },
      {
        "keys": ["bias", "beta"], # biases
        "lr": 0.003,
        # "momentum": 0.1,
        # "weight_decay": 0.001,
        "name": "biases"
      }
    ]

    self.others = {
        "keys": ["(.)*"], # required covers anything not in a group.
        "lr": lr,
        "momentum":momentum,
        "weight_decay": weight_decay,
        "name": "others"
      }
    groups.append(self.others)
    self.groups = self.set_groups(groups)
    self._variables_set = False

  def set_groups(self, groups):
    primary = groups[-1] # the others config i.e the default config

    # what are all the viable keys on all the groups
    key_set = set()
    for group in groups:
      key_set.update(group.keys())

    # construct and fill in the blanks.
    groups_dict = dict()
    skip_keys = ["keys", "name", "opt_keys", "varset"]
    for group in groups: 
      name = group["name"]
      group["opt_keys"] = {}
      for param in key_set: # keys across all configs not just this one
        if param not in skip_keys: # if you are to add this key as a param
          if param in group: 
            # if the param is in the config add its value
            self._set_hyper(f"{name}_{param}", group[param]) # set the lr, momentum etc for each group with a known key
          else:
            # if the param is not the config add the default value
            group[param] = primary[param]
            self._set_hyper(f"{name}_{param}", primary[param])
          group["opt_keys"][f"{name}_{param}"] = param
      
      groups_dict[group["name"]] = group
      group["varset"] = set()
    print(groups_dict)
    return groups_dict

  def _search(self, var, keys):
    """Search all all keys for matches. Return True on match."""
    if keys is not None:
      # variable group is not ignored so search for the keys. 
      for r in keys:
        if re.search(r, var.name) is not None:
          return True
    return False

  def search_and_set_variable_groups(self, variables):
    """Search all variable for matches at each group. 
    Args:
      variables: List[tf.Variable] from model.trainable_variables
    """
    kwargs = {}

    for key in self.groups:
      kwargs[key] = []

    for var in variables:
      for key in self.groups:
        if self._search(var, self.groups[key]["keys"]):
          kwargs[key].append(var)
          break

    self._set_variable_groups(**kwargs)
    return kwargs

  def _set_variable_groups(self, **kwargs):
    """Sets the variables to be used in each group."""
    log_msg = ""
    for key in kwargs:
      self.groups[key]["varset"].update(set([_var_key(v) for v in kwargs[key]]))
      log_msg += f"{key.capitalize()} : {len(self.groups[key]['varset'])} "
      
    print(log_msg)
    logging.info(log_msg)
    self._variables_set = True

  def _get_value_from_key(self, key, var_dtype):
    unit = self._get_hyper(key)
    if isinstance(unit, LearningRateSchedule):
      unit = unit(self.iterations)
    unit = tf.cast(unit, var_dtype)
    return unit

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(GroupOpt, self)._prepare_local(var_device, var_dtype,apply_state)

    for group in self.groups.values():
      for key in group["opt_keys"]:
        value = self._get_value_from_key(key, var_dtype)
        apply_state[(var_device, var_dtype)][key] = array_ops.identity(value)

    return apply_state[(var_device, var_dtype)]

  def _get_group_coeffecients(self, var, coefficients):
    if self._variables_set:
      for group in self.groups.values():
        if _var_key(var) in group["varset"]:
          return {value : coefficients[key] for key, value in group["opt_keys"].items()}
    else:
      for group in self.groups.values():
        if self._search(var, group["keys"]):
          return {value : coefficients[key] for key, value in group["opt_keys"].items()}
    return

  def _create_slots(self, var_list):
    """Create a momentum variable for each variable."""
    """SGD only"""
    for var in var_list:
      # check if trainable to support GPU EMA. 
      if var.trainable: 
        self.add_slot(var, "momentum")

  def _apply(self, grad, var, weight_decay, momentum, lr):
    """Uses Pytorch Optimizer with Weight decay SGDW."""
    dparams = grad
    groups = []

    # # do not update non-trainable weights
    # if not var.trainable:
    #   return tf.group(*groups)

    # if self._weight_decay:
    #   dparams += (weight_decay * var)

    # if self._momentum:
    #   momentum_var = self.get_slot(var, "momentum")
    #   momentum_update = momentum_var.assign(
    #       momentum * momentum_var + dparams, use_locking=self._use_locking)
    #   groups.append(momentum_update)

    #   if self.nesterov:
    #     dparams += (momentum * momentum_update)
    #   else:
    #     dparams = momentum_update

    # weight_update = var.assign_add(-lr * dparams, use_locking=self._use_locking)
    # groups.append(weight_update)
    return tf.group(*groups)

  def _run_sgd(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))

    group_coeffs = self._get_group_coeffecients(var, coefficients)
    weight_decay = group_coeffs["weight_decay"]
    momentum = group_coeffs["momentum"]
    lr = group_coeffs["lr"]

    tf.print(var.name, lr, weight_decay, momentum)
    return self._apply(grad, var, weight_decay, momentum, lr)
  
  def _resource_apply_dense(self, grad, var, apply_state=None):
    return self._run_sgd(grad, var, apply_state=apply_state)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # This method is only needed for momentum optimization.
    holder = tf.tensor_scatter_nd_add(
        tf.zeros_like(var), tf.expand_dims(indices, axis = -1), grad) 
    return self._run_sgd(holder, var, apply_state=apply_state)


if __name__ == "__main__":
  from yolo.run import load_model
  task, model, params = load_model(
      experiment="yolo_darknet",
      config_path=["yolo/configs/experiments/yolov4-tiny/inference/416.yaml"],
      model_dir='')
  k = GroupOpt()
  # k.search_and_set_variable_groups(model.trainable_variables)

  gradients = model.trainable_variables
  k.apply_gradients(zip(model.trainable_variables, gradients))