import tensorflow as tf
import official.core.base_task as bt
from official.core import input_reader
from yolo.dataloaders import yolo_input
from yolo.dataloaders.decoders import tfds_coco_decoder
from tensorflow.keras.mixed_precision import experimental as mixed_precision


class Trainer(tf.keras.Model):

  def __init__(self, task: bt.Task):
    super().__init__()

    self._task = task
    self._train_step = None
    self._test_step = None
    self._metrics = None

    self._model = task.build_model()
    task.initialize(self._model)

  def compile(self, optimizer, run_eagerly=False):
    self._metrics = self._task.build_metrics(training=False)
    super(Trainer, self).compile(optimizer=optimizer, run_eagerly=run_eagerly)
    return

  def train_step(self, inputs):
    # get the data point
    image, label = inputs
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      # compute a prediction
      # cast to float32
      y_pred = self._model(image, training=True)
      loss, metrics = self._task.build_losses(y_pred["raw_output"], label)
      scaled_loss = loss / num_replicas

      # scale the loss for numerical stability
      if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
        scaled_loss = self.optimizer.get_scaled_loss(scaled_loss)
    # compute the gradient
    train_vars = self._model.trainable_variables
    gradients = tape.gradient(scaled_loss, train_vars)
    # get unscaled loss if the scaled_loss was used
    if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
      gradients = self.optimizer.get_unscaled_gradients(gradients)
    if self._task.task_config.gradient_clip_norm > 0.0:
      gradients, _ = tf.clip_by_global_norm(gradients,
                                            self.task_config.gradient_clip_norm)
    self.optimizer.apply_gradients(zip(gradients, train_vars))

    # custom metrics
    logs = {"loss": loss}
    logs.update(metrics)
    return logs

  # def train_step(self, inputs):
  #     return self._task.train_step(inputs, self._model, self._optimizer, metrics=self._metrics)

  def test_step(self, inputs):
    return self._task.validation_step(
        inputs, self._model, metrics=self._metrics)

  def build_inputs(self, params, is_training=True, input_context=None):
    params.is_training = False
    decoder = tfds_coco_decoder.MSCOCODecoder()
    parser = yolo_input.Parser(
        image_w=params.parser.image_w,
        image_h=params.parser.image_h,
        num_classes=self._task.task_config.model.num_classes,
        fixed_size=params.parser.fixed_size,
        jitter_im=params.parser.jitter_im,
        jitter_boxes=params.parser.jitter_boxes,
        net_down_scale=params.parser.net_down_scale,
        min_process_size=params.parser.min_process_size,
        max_process_size=params.parser.max_process_size,
        max_num_instances=params.parser.max_num_instances,
        random_flip=params.parser.random_flip,
        pct_rand=params.parser.pct_rand,
        seed=params.parser.seed,
        anchors=self._task.task_config.model.boxes)

    if is_training:
      post_process_fn = parser.postprocess_fn()
    else:
      post_process_fn = None

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(is_training),
        postprocess_fn=post_process_fn)

    dataset = reader.read(input_context=input_context)
    return dataset

  def build_datasets(self, take=1000):
    train_data = self.build_inputs(
        self._task.task_config.train_data, is_training=True).take(1000)
    validation_data = self.build_inputs(
        self._task.task_config.validation_data, is_training=False).take(1000)
    return train_data, validation_data

  def train(self, train_data, test_data, optimizer, epochs=1, callbacks=None):
    self.compile(optimizer)
    self.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0)
    return

  @staticmethod
  def fit_optimizer(optimizer, scaling="dynamic"):
    policy = tf.keras.mixed_precision.experimental.global_policy()
    if "float32" not in policy.name:
      return tf.keras.mixed_precision.experimental.LossScaleOptimizer(
          optimizer, scaling)
    return optimizer


if __name__ == "__main__":
  import datetime
  from yolo.utils.run_utils import prep_gpu
  prep_gpu()
  from yolo.configs import yolo as exp_cfg
  from yolo.utils.training import batch_lr_schedule as bls
  from yolo.utils.training import schedule
  from yolo.utils.training import history_full_callback as hfc
  from yolo.tasks.yolo import YoloTask

  config = exp_cfg.YoloTask(
      model=exp_cfg.Yolo(
          base="v4",
          min_level=3,
          norm_activation=exp_cfg.common.NormActivation(activation="mish"),
          #norm_activation = exp_cfg.common.NormActivation(activation="leaky"),
          #_boxes = ['(10, 14)', '(23, 27)', '(37, 58)', '(81, 82)', '(135, 169)', '(344, 319)'],
          _boxes=[
              "(12, 16)", "(19, 36)", "(40, 28)", "(36, 75)", "(76, 55)",
              "(72, 146)", "(142, 110)", "(192, 243)", "(459, 401)"
          ],
          filter=exp_cfg.YoloLossLayer(use_nms=True)),
      darknet_load_decoder=False)

  task = YoloTask(config)
  trainer = Trainer(task)
  train, test = trainer.build_datasets()
  size = tf.data.experimental.cardinality(train).numpy()
  tsize = tf.data.experimental.cardinality(test).numpy()

  EPOCHS = 1
  step_changes = [int(size * EPOCHS * 0.8), int(size * EPOCHS * 0.9)]
  schedule = schedule.Schedule(
      1000,
      0.001,
      total_steps=size * EPOCHS,
      type="STEPWISE",
      step_changes=step_changes,
      decay_factor=0.1)
  lr_callback = bls.LearningRateScheduler(schedule)
  history = hfc.HistoryFull()
  callbacks = [lr_callback, history]

  # TODO: negative box loss error in ciou
  optimizer = Trainer.fit_optimizer(tf.keras.optimizers.SGD())

  start = datetime.datetime.now()  # time.time()
  try:
    trainer.train(
        train, test, optimizer=optimizer, epochs=EPOCHS, callbacks=callbacks)
  except Exception as e:
    print(e)
  stop = datetime.datetime.now()
  print(
      f"\n\n\n\n  train dataset size: {size * EPOCHS}, test dataset size: {tsize}, total time: {stop - start}"
  )

  # trainer._model.save("saved_models/v4/wacko_2_epochs")

  # observation 1:
  # the keras trainer needs far more memory to operate
  # train dataset size: 1000, test dataset size: 1000, total time: 0:26:41.862174
