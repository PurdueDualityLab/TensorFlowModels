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
"""Loads dataset for the sentence prediction (classification) task."""
from typing import List, Mapping, Optional

import dataclasses
import tensorflow as tf
import tensorflow_hub as hub

from official.common import dataset_fn
from official.core import config_definitions as cfg
from official.core import input_reader
from official.nlp import modeling
from official.nlp.data import data_loader
from official.nlp.data import data_loader_factory

LABEL_TYPES_MAP = {'int': tf.int64, 'float': tf.float32}


@dataclasses.dataclass
class SentencePredictionDataConfig(cfg.DataConfig):
  """Data config for sentence prediction task (tasks/sentence_prediction)."""
  input_path: str = ''
  global_batch_size: int = 32
  is_training: bool = True
  seq_length: int = 128
  label_type: str = 'int'
  # Whether to include the example id number.
  include_example_id: bool = False


@data_loader_factory.register_data_loader_cls(SentencePredictionDataConfig)
class SentencePredictionDataLoader(data_loader.DataLoader):
  """A class to load dataset for sentence prediction (classification) task."""

  def __init__(self, params):
    self._params = params
    self._seq_length = params.seq_length
    self._include_example_id = params.include_example_id

  def _decode(self, record: tf.Tensor):
    """Decodes a serialized tf.Example."""
    label_type = LABEL_TYPES_MAP[self._params.label_type]
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'input_mask': tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'label_ids': tf.io.FixedLenFeature([], label_type),
    }
    if self._include_example_id:
      name_to_features['example_id'] = tf.io.FixedLenFeature([], tf.int64)

    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in example:
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def _parse(self, record: Mapping[str, tf.Tensor]):
    """Parses raw tensors into a dict of tensors to be consumed by the model."""
    x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids']
    }
    if self._include_example_id:
      x['example_id'] = record['example_id']

    y = record['label_ids']
    return (x, y)

  def load(self, input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a tf.dataset.Dataset."""
    reader = input_reader.InputReader(
        params=self._params, decoder_fn=self._decode, parser_fn=self._parse)
    return reader.read(input_context)


@dataclasses.dataclass
class SentencePredictionTextDataConfig(cfg.DataConfig):
  """Data config for sentence prediction task with raw text."""
  # Either set `input_path`...
  input_path: str = ''
  # Either `int` or `float`.
  label_type: str = 'int'
  # ...or `tfds_name` and `tfds_split` to specify input.
  tfds_name: str = ''
  tfds_split: str = ''
  # The name of the text feature fields. The text features will be
  # concatenated in order.
  text_fields: Optional[List[str]] = None
  label_field: str = 'label'
  global_batch_size: int = 32
  seq_length: int = 128
  is_training: bool = True
  # Either build preprocessing with Python code by specifying these values
  # for modeling.layers.BertTokenizer()/SentencepieceTokenizer()....
  tokenization: str = 'WordPiece'  # WordPiece or SentencePiece
  # Text vocab file if tokenization is WordPiece, or sentencepiece.ModelProto
  # file if tokenization is SentencePiece.
  vocab_file: str = ''
  lower_case: bool = True
  # ...or load preprocessing from a SavedModel at this location.
  preprocessing_hub_module_url: str = ''
  # Either tfrecord or sstsable or recordio.
  file_type: str = 'tfrecord'


class TextProcessor(tf.Module):
  """Text features processing for sentence prediction task."""

  def __init__(self,
               seq_length: int,
               vocab_file: Optional[str] = None,
               tokenization: Optional[str] = None,
               lower_case: Optional[bool] = True,
               preprocessing_hub_module_url: Optional[str] = None):
    if preprocessing_hub_module_url:
      self._preprocessing_hub_module = hub.load(preprocessing_hub_module_url)
      self._tokenizer = self._preprocessing_hub_module.tokenize
      def set_shape(t):
        # Before TF2.4, the sequence length dimension loaded from the
        # preprocessing hub module is None, so we recover the shape here.
        # TODO(b/157636658): Remove once TF2.4 is released and being used.
        t.set_shape([None, seq_length])
        return t

      def pack_inputs_fn(inputs):
        result = self._preprocessing_hub_module.bert_pack_inputs(
            inputs, seq_length=seq_length)
        result = tf.nest.map_structure(set_shape, result)
        return result

      self._pack_inputs = pack_inputs_fn
      return

    if tokenization == 'WordPiece':
      self._tokenizer = modeling.layers.BertTokenizer(
          vocab_file=vocab_file, lower_case=lower_case)
    elif tokenization == 'SentencePiece':
      self._tokenizer = modeling.layers.SentencepieceTokenizer(
          model_file_path=vocab_file, lower_case=lower_case,
          strip_diacritics=True)  # Strip diacritics to follow ALBERT model
    else:
      raise ValueError('Unsupported tokenization: %s' % tokenization)

    self._pack_inputs = modeling.layers.BertPackInputs(
        seq_length=seq_length,
        special_tokens_dict=self._tokenizer.get_special_tokens_dict())

  def __call__(self, segments):
    segments = [self._tokenizer(s) for s in segments]
    # BertTokenizer returns a RaggedTensor with shape [batch, word, subword],
    # and SentencepieceTokenizer returns a RaggedTensor with shape
    # [batch, sentencepiece],
    segments = [
        tf.cast(x.merge_dims(1, -1) if x.shape.rank > 2 else x, tf.int32)
        for x in segments
    ]
    return self._pack_inputs(segments)


@data_loader_factory.register_data_loader_cls(SentencePredictionTextDataConfig)
class SentencePredictionTextDataLoader(data_loader.DataLoader):
  """Loads dataset with raw text for sentence prediction task."""

  def __init__(self, params):
    if bool(params.tfds_name) != bool(params.tfds_split):
      raise ValueError('`tfds_name` and `tfds_split` should be specified or '
                       'unspecified at the same time.')
    if bool(params.tfds_name) == bool(params.input_path):
      raise ValueError('Must specify either `tfds_name` and `tfds_split` '
                       'or `input_path`.')
    if not params.text_fields:
      raise ValueError('Unexpected empty text fields.')
    if bool(params.vocab_file) == bool(params.preprocessing_hub_module_url):
      raise ValueError('Must specify exactly one of vocab_file (with matching '
                       'lower_case flag) or preprocessing_hub_module_url.')

    self._params = params
    self._text_fields = params.text_fields
    self._label_field = params.label_field
    self._label_type = params.label_type
    self._text_processor = TextProcessor(
        seq_length=params.seq_length,
        vocab_file=params.vocab_file,
        tokenization=params.tokenization,
        lower_case=params.lower_case,
        preprocessing_hub_module_url=params.preprocessing_hub_module_url)

  def _bert_preprocess(self, record: Mapping[str, tf.Tensor]):
    """Berts preprocess."""
    segments = [record[x] for x in self._text_fields]
    model_inputs = self._text_processor(segments)
    y = record[self._label_field]
    return model_inputs, y

  def _decode(self, record: tf.Tensor):
    """Decodes a serialized tf.Example."""
    name_to_features = {}
    for text_field in self._text_fields:
      name_to_features[text_field] = tf.io.FixedLenFeature([], tf.string)

    label_type = LABEL_TYPES_MAP[self._label_type]
    name_to_features[self._label_field] = tf.io.FixedLenFeature([], label_type)
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in example:
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def load(self, input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a tf.dataset.Dataset."""
    reader = input_reader.InputReader(
        dataset_fn=dataset_fn.pick_dataset_fn(self._params.file_type),
        decoder_fn=self._decode if self._params.input_path else None,
        params=self._params,
        postprocess_fn=self._bert_preprocess)
    return reader.read(input_context)
