# Lint as: python3
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
"""Tests for official.nlp.tasks.electra_task."""

import tensorflow as tf

from official.nlp.configs import bert
from official.nlp.configs import electra
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader
from official.nlp.tasks import electra_task


class ElectraPretrainTaskTest(tf.test.TestCase):

  def test_task(self):
    config = electra_task.ElectraPretrainConfig(
        model=electra.ElectraPretrainerConfig(
            generator_encoder=encoders.EncoderConfig(
                bert=encoders.BertEncoderConfig(vocab_size=30522,
                                                num_layers=1)),
            discriminator_encoder=encoders.EncoderConfig(
                bert=encoders.BertEncoderConfig(vocab_size=30522,
                                                num_layers=1)),
            num_masked_tokens=20,
            sequence_length=128,
            cls_heads=[
                bert.ClsHeadConfig(
                    inner_dim=10, num_classes=2, name="next_sentence")
            ]),
        train_data=pretrain_dataloader.BertPretrainDataConfig(
            input_path="dummy",
            max_predictions_per_seq=20,
            seq_length=128,
            global_batch_size=1))
    task = electra_task.ElectraPretrainTask(config)
    model = task.build_model()
    metrics = task.build_metrics()
    dataset = task.build_inputs(config.train_data)

    iterator = iter(dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    task.validation_step(next(iterator), model, metrics=metrics)


if __name__ == "__main__":
  tf.test.main()
