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

"""TensorFlow Model Garden Vision trainer.

All custom registry are imported from registry_imports. Here we use default
trainer so we directly call train.main. If you need to customize the trainer,
branch from `official/vision/beta/train.py` and make changes.
"""
from absl import app

from official.common import flags as tfm_flags
from official.vision.beta import train
from official.vision.beta.projects.example import registry_imports  # pylint: disable=unused-import


if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(train.main)
