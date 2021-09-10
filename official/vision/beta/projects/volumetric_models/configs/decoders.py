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

# Lint as: python3
"""Decoders configurations."""
from typing import Optional, Sequence

import dataclasses

from official.modeling import hyperparams


@dataclasses.dataclass
class Identity(hyperparams.Config):
  """Identity config."""
  pass


@dataclasses.dataclass
class UNet3DDecoder(hyperparams.Config):
  """UNet3D decoder config."""
  model_id: int = 4
  pool_size: Sequence[int] = (2, 2, 2)
  kernel_size: Sequence[int] = (3, 3, 3)
  use_batch_normalization: bool = True
  use_deconvolution: bool = True


@dataclasses.dataclass
class Decoder(hyperparams.OneOfConfig):
  """Configuration for decoders.

  Attributes:
    type: 'str', type of decoder be used, on the of fields below.
    identity: identity decoder config.
    unet_3d_decoder: UNet3D decoder config.
  """
  type: Optional[str] = None
  identity: Identity = Identity()
  unet_3d_decoder: UNet3DDecoder = UNet3DDecoder()
