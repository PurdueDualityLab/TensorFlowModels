# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests official.nlp.albert.export_albert_tfhub."""
import os

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from official.nlp.albert import configs
from official.nlp.albert import export_albert_tfhub


class ExportAlbertTfhubTest(tf.test.TestCase):

  def test_export_albert_tfhub(self):
    # Exports a savedmodel for TF-Hub
    albert_config = configs.AlbertConfig(
        vocab_size=100,
        embedding_size=8,
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_hidden_layers=1)
    bert_model, encoder = export_albert_tfhub.create_albert_model(albert_config)
    model_checkpoint_dir = os.path.join(self.get_temp_dir(), "checkpoint")
    checkpoint = tf.train.Checkpoint(model=encoder)
    checkpoint.save(os.path.join(model_checkpoint_dir, "test"))
    model_checkpoint_path = tf.train.latest_checkpoint(model_checkpoint_dir)

    sp_model_file = os.path.join(self.get_temp_dir(), "sp_tokenizer.model")
    with tf.io.gfile.GFile(sp_model_file, "w") as f:
      f.write("dummy content")

    hub_destination = os.path.join(self.get_temp_dir(), "hub")
    export_albert_tfhub.export_albert_tfhub(
        albert_config,
        model_checkpoint_path,
        hub_destination,
        sp_model_file=sp_model_file)

    # Restores a hub KerasLayer.
    hub_layer = hub.KerasLayer(hub_destination, trainable=True)

    if hasattr(hub_layer, "resolved_object"):
      with tf.io.gfile.GFile(
          hub_layer.resolved_object.sp_model_file.asset_path.numpy()) as f:
        self.assertEqual("dummy content", f.read())
    # Checks the hub KerasLayer.
    for source_weight, hub_weight in zip(bert_model.trainable_weights,
                                         hub_layer.trainable_weights):
      self.assertAllClose(source_weight.numpy(), hub_weight.numpy())

    dummy_ids = np.zeros((2, 10), dtype=np.int32)
    hub_outputs = hub_layer([dummy_ids, dummy_ids, dummy_ids])
    source_outputs = bert_model([dummy_ids, dummy_ids, dummy_ids])

    # The outputs of hub module are "pooled_output" and "sequence_output",
    # while the outputs of encoder is in reversed order, i.e.,
    # "sequence_output" and "pooled_output".
    encoder_outputs = reversed(encoder([dummy_ids, dummy_ids, dummy_ids]))
    self.assertEqual(hub_outputs[0].shape, (2, 16))
    self.assertEqual(hub_outputs[1].shape, (2, 10, 16))
    for source_output, hub_output, encoder_output in zip(
        source_outputs, hub_outputs, encoder_outputs):
      self.assertAllClose(source_output.numpy(), hub_output.numpy())
      self.assertAllClose(source_output.numpy(), encoder_output.numpy())


if __name__ == "__main__":
  tf.test.main()
