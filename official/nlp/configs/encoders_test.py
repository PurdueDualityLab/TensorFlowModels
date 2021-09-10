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

"""Tests for official.nlp.configs.encoders."""
import os

import tensorflow as tf

from official.modeling import hyperparams
from official.nlp.configs import encoders


class EncodersTest(tf.test.TestCase):

  def test_encoder_from_yaml(self):
    config = encoders.EncoderConfig(
        type="bert", bert=encoders.BertEncoderConfig(num_layers=1))
    encoder = encoders.build_encoder(config)
    ckpt = tf.train.Checkpoint(encoder=encoder)
    ckpt_path = ckpt.save(self.get_temp_dir() + "/ckpt")
    params_save_path = os.path.join(self.get_temp_dir(), "params.yaml")
    hyperparams.save_params_dict_to_yaml(config, params_save_path)

    retored_cfg = encoders.EncoderConfig.from_yaml(params_save_path)
    retored_encoder = encoders.build_encoder(retored_cfg)
    status = tf.train.Checkpoint(encoder=retored_encoder).restore(ckpt_path)
    status.assert_consumed()


if __name__ == "__main__":
  tf.test.main()
