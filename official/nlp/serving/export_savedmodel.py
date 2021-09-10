# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A binary/library to export TF-NLP serving `SavedModel`."""
import os
from typing import Any, Dict, Text
from absl import app
from absl import flags
import dataclasses
import yaml
from official.core import base_task
from official.core import task_factory
from official.modeling import hyperparams
from official.modeling.hyperparams import base_config
from official.nlp.serving import export_savedmodel_util
from official.nlp.serving import serving_modules
from official.nlp.tasks import masked_lm
from official.nlp.tasks import question_answering
from official.nlp.tasks import sentence_prediction
from official.nlp.tasks import tagging

FLAGS = flags.FLAGS

SERVING_MODULES = {
    sentence_prediction.SentencePredictionTask:
        serving_modules.SentencePrediction,
    masked_lm.MaskedLMTask:
        serving_modules.MaskedLM,
    question_answering.QuestionAnsweringTask:
        serving_modules.QuestionAnswering,
    tagging.TaggingTask:
        serving_modules.Tagging
}


def define_flags():
  """Defines flags."""
  flags.DEFINE_string("task_name", "SentencePrediction", "The task to export.")
  flags.DEFINE_string("config_file", None,
                      "The path to task/experiment yaml config file.")
  flags.DEFINE_string(
      "checkpoint_path", None,
      "Object-based checkpoint path, from the training model directory.")
  flags.DEFINE_string("export_savedmodel_dir", None,
                      "Output saved model directory.")
  flags.DEFINE_string(
      "serving_params", None,
      "a YAML/JSON string or csv string for the serving parameters.")
  flags.DEFINE_string(
      "function_keys", None,
      "A string key to retrieve pre-defined serving signatures.")
  flags.DEFINE_bool("convert_tpu", False, "")
  flags.DEFINE_multi_integer("allowed_batch_size", None,
                             "Allowed batch sizes for batching ops.")


def lookup_export_module(task: base_task.Task):
  export_module_cls = SERVING_MODULES.get(task.__class__, None)
  if export_module_cls is None:
    ValueError("No registered export module for the task: %s", task.__class__)
  return export_module_cls


def create_export_module(*, task_name: Text, config_file: Text,
                         serving_params: Dict[Text, Any]):
  """Creates a ExportModule."""
  task_config_cls = None
  task_cls = None
  # pylint: disable=protected-access
  for key, value in task_factory._REGISTERED_TASK_CLS.items():
    print(key.__name__)
    if task_name in key.__name__:
      task_config_cls, task_cls = key, value
      break
  if task_cls is None:
    raise ValueError("Failed to identify the task class. The provided task "
                     f"name is {task_name}")
  # pylint: enable=protected-access
  # TODO(hongkuny): Figure out how to separate the task config from experiments.

  @dataclasses.dataclass
  class Dummy(base_config.Config):
    task: task_config_cls = task_config_cls()

  dummy_exp = Dummy()
  dummy_exp = hyperparams.override_params_dict(
      dummy_exp, config_file, is_strict=False)
  dummy_exp.task.validation_data = None
  task = task_cls(dummy_exp.task)
  model = task.build_model()
  export_module_cls = lookup_export_module(task)
  params = export_module_cls.Params(**serving_params)
  return export_module_cls(params=params, model=model)


def main(_):
  serving_params = yaml.load(
      hyperparams.nested_csv_str_to_json_str(FLAGS.serving_params),
      Loader=yaml.FullLoader)
  export_module = create_export_module(
      task_name=FLAGS.task_name,
      config_file=FLAGS.config_file,
      serving_params=serving_params)
  export_dir = export_savedmodel_util.export(
      export_module,
      function_keys=[FLAGS.function_keys],
      checkpoint_path=FLAGS.checkpoint_path,
      export_savedmodel_dir=FLAGS.export_savedmodel_dir)

  if FLAGS.convert_tpu:
    # pylint: disable=g-import-not-at-top
    from cloud_tpu.inference_converter import converter_cli
    from cloud_tpu.inference_converter import converter_options_pb2
    tpu_dir = os.path.join(export_dir, "tpu")
    options = converter_options_pb2.ConverterOptions()
    if FLAGS.allowed_batch_size is not None:
      allowed_batch_sizes = sorted(FLAGS.allowed_batch_size)
      options.batch_options.num_batch_threads = 4
      options.batch_options.max_batch_size = allowed_batch_sizes[-1]
      options.batch_options.batch_timeout_micros = 100000
      options.batch_options.allowed_batch_sizes[:] = allowed_batch_sizes
      options.batch_options.max_enqueued_batches = 1000
    converter_cli.ConvertSavedModel(
        export_dir, tpu_dir, function_alias="tpu_candidate", options=options,
        graph_rewrite_only=True)


if __name__ == "__main__":
  define_flags()
  app.run(main)
