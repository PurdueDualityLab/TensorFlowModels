# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Run ALBERT on SQuAD 1.1 and SQuAD 2.0 in TF 2.x."""

import json
import os
import time

# Import libraries
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from official.common import distribute_utils
from official.nlp.albert import configs as albert_configs
from official.nlp.bert import run_squad_helper
from official.nlp.bert import tokenization
from official.nlp.data import squad_lib_sp

flags.DEFINE_string(
    'sp_model_file', None,
    'The path to the sentence piece model. Used by sentence piece tokenizer '
    'employed by ALBERT.')

# More flags can be found in run_squad_helper.
run_squad_helper.define_common_squad_flags()

FLAGS = flags.FLAGS


def train_squad(strategy,
                input_meta_data,
                custom_callbacks=None,
                run_eagerly=False):
  """Runs bert squad training."""
  bert_config = albert_configs.AlbertConfig.from_json_file(
      FLAGS.bert_config_file)
  run_squad_helper.train_squad(strategy, input_meta_data, bert_config,
                               custom_callbacks, run_eagerly)


def predict_squad(strategy, input_meta_data):
  """Makes predictions for the squad dataset."""
  bert_config = albert_configs.AlbertConfig.from_json_file(
      FLAGS.bert_config_file)
  tokenizer = tokenization.FullSentencePieceTokenizer(
      sp_model_file=FLAGS.sp_model_file)

  run_squad_helper.predict_squad(strategy, input_meta_data, tokenizer,
                                 bert_config, squad_lib_sp)


def eval_squad(strategy, input_meta_data):
  """Evaluate on the squad dataset."""
  bert_config = albert_configs.AlbertConfig.from_json_file(
      FLAGS.bert_config_file)
  tokenizer = tokenization.FullSentencePieceTokenizer(
      sp_model_file=FLAGS.sp_model_file)

  eval_metrics = run_squad_helper.eval_squad(
      strategy, input_meta_data, tokenizer, bert_config, squad_lib_sp)
  return eval_metrics


def export_squad(model_export_path, input_meta_data):
  """Exports a trained model as a `SavedModel` for inference.

  Args:
    model_export_path: a string specifying the path to the SavedModel directory.
    input_meta_data: dictionary containing meta data about input and model.

  Raises:
    Export path is not specified, got an empty string or None.
  """
  bert_config = albert_configs.AlbertConfig.from_json_file(
      FLAGS.bert_config_file)
  run_squad_helper.export_squad(model_export_path, input_meta_data, bert_config)


def main(_):
  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  if FLAGS.mode == 'export_only':
    export_squad(FLAGS.model_export_path, input_meta_data)
    return

  # Configures cluster spec for multi-worker distribution strategy.
  if FLAGS.num_gpus > 0:
    _ = distribute_utils.configure_cluster(FLAGS.worker_hosts, FLAGS.task_index)
  strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      all_reduce_alg=FLAGS.all_reduce_alg,
      tpu_address=FLAGS.tpu)

  if 'train' in FLAGS.mode:
    train_squad(strategy, input_meta_data, run_eagerly=FLAGS.run_eagerly)
  if 'predict' in FLAGS.mode:
    predict_squad(strategy, input_meta_data)
  if 'eval' in FLAGS.mode:
    eval_metrics = eval_squad(strategy, input_meta_data)
    f1_score = eval_metrics['final_f1']
    logging.info('SQuAD eval F1-score: %f', f1_score)
    summary_dir = os.path.join(FLAGS.model_dir, 'summaries', 'eval')
    summary_writer = tf.summary.create_file_writer(summary_dir)
    with summary_writer.as_default():
      # TODO(lehou): write to the correct step number.
      tf.summary.scalar('F1-score', f1_score, step=0)
      summary_writer.flush()
    # Also write eval_metrics to json file.
    squad_lib_sp.write_to_json_files(
        eval_metrics, os.path.join(summary_dir, 'eval_metrics.json'))
    time.sleep(60)


if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
