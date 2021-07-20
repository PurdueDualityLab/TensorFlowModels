import numpy as np


class Schedule(object):

  def __init__(self,
               warmup_steps,
               learning_rate,
               total_steps,
               type='COSINE',
               hold_base_rate_steps=0,
               offset=0.0,
               **kwargs):
    self._global_step = 0
    self._warmup_steps = warmup_steps
    self.__warm_up_sumstep = learning_rate / warmup_steps
    self._learning_rate = learning_rate
    self._total_steps = total_steps
    self._curr_lr = 0.0
    self._offset = offset
    self._hold_lr_befor_decay = hold_base_rate_steps
    print(self._total_steps)

    lr_function = Schedule._TYPES[type]
    self._lr_function = lambda: lr_function(self, **kwargs)

  def run(self):
    self._global_step += 1
    if self._global_step <= self._warmup_steps:  # Does this equals sign belong here?
      self._curr_lr += self.__warm_up_sumstep
    elif self._curr_lr >= self._offset:
      self._curr_lr = self._lr_function()
    else:
      self._curr_lr = self._offset
    return self._curr_lr, self._global_step

  def cosine_lr(self):
    if self._global_step - self._warmup_steps <= self._hold_lr_befor_decay:  # Does this belong here?
      return self._learning_rate
    learning_rate = 0.5 * self._learning_rate * (1 + np.cos(
        np.pi *
        (self._global_step - self._warmup_steps - self._hold_lr_befor_decay) /
        float(self._total_steps - self._warmup_steps -
              self._hold_lr_befor_decay)))
    return learning_rate

  def stepwise_lr(self, step_changes, decay_factor):
    curr_lr = self._curr_lr
    if (
        self._global_step - self._warmup_steps
    ) in step_changes:  # Does subtracting the self._warmup_steps belong here?
      curr_lr *= decay_factor
    return curr_lr


Schedule._TYPES = {
    'COSINE': Schedule.cosine_lr,
    'STEPWISE': Schedule.stepwise_lr,
}
