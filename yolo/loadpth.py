from official.vision.beta.configs import backbones
from typing import Final

from numpy.core.numeric import full
from yolo.modeling.backbones.swin import SwinTransformer
from official.vision.beta.modeling import classification_model
import tensorflow as tf
import numpy as np
 


shape = [224, 224, 3]
input_specs = tf.keras.layers.InputSpec(shape=[None]+shape)
model = SwinTransformer(
  input_specs = input_specs, 
  min_level = 3, 
  max_level = None, 
  patch_size = 4, 
  embed_dims = 96, 
  window_size = [7, 7, 7, 7],  
  depths = [2, 2, 6, 2], 
  num_heads = [3, 6, 12, 24], 
  down_sample_all=False, 
  dense_embeddings=False, 
  normalize_endpoints = True,
)
model.summary()

full_model = classification_model.ClassificationModel(
  backbone = model, 
  num_classes = 1000, 
  input_specs = tf.keras.layers.InputSpec(
          shape=[None, 224, 224, 3])
)
full_model.summary()

import torch
PATH = "/home/vbanna/Research/TensorFlowModels/swin_tiny_patch4_window7_224.pth"
checkpoint = torch.load(PATH, map_location=torch.device("cpu"))

points = list(checkpoint["model"].keys())

print(points)

linear = []
norm = []
conv = []
tables = []
indexes = []
other = []

for key in points:
  if "proj" in key and "patch_embed" in key:
    conv.append(key)
    print(checkpoint["model"][key].size())
  elif "norm" in key:
    norm.append(key)
  elif "index" in key:
    indexes.append(key)
  elif "table" in key:
    tables.append(key)
  elif "mask" in key:
    other.append(key)
  else:
    linear.append(key)

# t = 3
# j = -3
# final = None
# for block in model.submodules:
#   # print()
#   # for block in module.submodules:

#   if isinstance(block, tf.keras.layers.Conv2D):
#     num_vars = len(block.trainable_variables)
#     load_vars = []
#     for i, varible in enumerate(block.trainable_variables):
#       load_vars.append(checkpoint["model"][conv.pop(0)])
#     load_vars[0] = np.transpose(load_vars[0], axes = [2, 3, 1, 0])
#     block.set_weights(load_vars)
#     print("conv: ", len(load_vars), len(conv))
#   elif isinstance(block, tf.keras.layers.Dense):
#     num_vars = len(block.trainable_variables)
#     load_vars = []
#     for varible in block.trainable_variables:
#       if num_vars == 1:
#         key = linear.pop(j)
#         j += 1
#       else:
#         key = linear.pop(0)
#       var = checkpoint["model"][key]
#       load_vars.append(var)
#       # print(varible.name, varible.shape, var.size(), key)
#     load_vars[0] = np.transpose(load_vars[0], axes=[1, 0])
#     block.set_weights(load_vars)
#     print("linear: ", len(load_vars), len(linear))
#   elif isinstance(block, tf.keras.layers.LayerNormalization):
#     if t > 0:
#       print(block)
#       if t == 1:
#         final = block
#       t -= 1
#       continue
#     num_vars = len(block.trainable_variables)
#     load_vars = []
#     for varible in block.trainable_variables:
#       load_vars.append(checkpoint["model"][norm.pop(0)])
#     block.set_weights(load_vars)
#     print("norm: ", len(load_vars), len(norm))

# if final is not None:
#   block = final
#   print(block)
#   num_vars = len(block.trainable_variables)
#   print(num_vars)
#   load_vars = []
#   for varible in block.trainable_variables:
#     load_vars.append(checkpoint["model"][norm.pop(0)])
#   block.set_weights(load_vars)
#   print("norm: ", len(load_vars), len(norm))

t = 3
j = -3
final = None
dense = None
for block in full_model.submodules:
  if isinstance(block, tf.keras.layers.Conv2D):
    num_vars = len(block.trainable_variables)
    load_vars = []
    for i, varible in enumerate(block.trainable_variables):
      load_vars.append(checkpoint["model"][conv.pop(0)])
    load_vars[0] = np.transpose(load_vars[0], axes = [2, 3, 1, 0])
    block.set_weights(load_vars)
    print("conv: ", len(load_vars), len(conv))
  elif isinstance(block, tf.keras.layers.Dense):
    if dense is None:
      dense = block 
      continue
    num_vars = len(block.trainable_variables)
    load_vars = []
    for varible in block.trainable_variables:
      if num_vars == 1:
        key = linear.pop(j)
        j += 1
      else:
        key = linear.pop(0)
      var = checkpoint["model"][key]
      load_vars.append(var)
      # print(varible.name, varible.shape, var.size(), key)
    load_vars[0] = np.transpose(load_vars[0], axes=[1, 0])
    block.set_weights(load_vars)
    print("linear: ", len(load_vars), len(linear))
  elif isinstance(block, tf.keras.layers.LayerNormalization):
    if t > 0:
      print(block)
      if t == 1:
        final = block
      t -= 1
      continue
    num_vars = len(block.trainable_variables)
    load_vars = []
    for varible in block.trainable_variables:
      load_vars.append(checkpoint["model"][norm.pop(0)])
    block.set_weights(load_vars)
    print("norm: ", len(load_vars), len(norm))

if final is not None:
  block = final
  num_vars = len(block.trainable_variables)
  print(num_vars)
  load_vars = []
  for varible in block.trainable_variables:
    load_vars.append(checkpoint["model"][norm.pop(0)])
  block.set_weights(load_vars)
  print("norm: ", len(load_vars), len(norm))

if isinstance(dense, tf.keras.layers.Dense):
  block = dense
  num_vars = len(block.trainable_variables)
  load_vars = []
  for varible in block.trainable_variables:
    if num_vars == 1:
      key = linear.pop(j)
      j += 1
    else:
      key = linear.pop(0)
    var = checkpoint["model"][key]
    load_vars.append(var)
    # print(varible.name, varible.shape, var.size(), key)
  load_vars[0] = np.transpose(load_vars[0], axes=[1, 0])
  block.set_weights(load_vars)
  print("linear: ", len(load_vars), len(linear))

check_ind = False 
if check_ind:
  variables = full_model.trainable_variables
else:
  variables = full_model.variables
for i, variable in enumerate(full_model.variables):
  if "table" in variable.name:
    key = tables.pop(0)
    var = tf.convert_to_tensor(checkpoint["model"][key].numpy())
    variable.assign(var)
    # print(variable.name, variable.shape, var.shape)
  elif "index" in variable.name:
    key = indexes.pop(0)
    var = tf.convert_to_tensor(checkpoint["model"][key].numpy())
    tf.print(tf.reduce_all(var == variable))
    # variable.assign(var)
    print(variable.name, variable.shape, var.shape)


ckpt = tf.train.Checkpoint(full_model)
ckpt.write("../checkpoints/loaded_swin/loaded_swin_ckpt")