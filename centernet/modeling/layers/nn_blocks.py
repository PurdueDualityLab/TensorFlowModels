import tensorflow as tf

from official.vision.beta.modeling.layers.nn_blocks import ResidualBlock

class kp_module(tf.keras.Model):
  def __init__(
    self, n, dims, modules, layer=ResidualBlock, **kwargs
  ):
    super().__init__()

    self.n   = n

    curr_mod = modules[0]
    next_mod = modules[1]

    curr_dim = dims[0]
    next_dim = dims[1]

    self.up1  = self.make_up_layer(
      3, curr_dim, curr_dim, curr_mod,
      layer=layer, **kwargs
    )
    self.max1 = self.make_pool_layer(curr_dim)
    self.low1 = self.make_hg_layer(
      3, curr_dim, next_dim, curr_mod,
      layer=layer, **kwargs
    )
    self.low2 = type(self)(
      n - 1, dims[1:], modules[1:], layer=layer, **kwargs
    ) if self.n > 1 else \
    self.make_low_layer(
      3, next_dim, next_dim, next_mod,
      layer=layer, **kwargs
    )
    self.low3 = self.make_hg_layer_revr(
      3, next_dim, curr_dim, curr_mod,
      layer=layer, **kwargs
    )
    self.up2  = self.make_unpool_layer(curr_dim)

    self.merge = self.make_merge_layer(curr_dim)

  def call(self, x):
    up1  = self.up1(x)
    max1 = self.max1(x)
    low1 = self.low1(max1)
    low2 = self.low2(low1)
    low3 = self.low3(low2)
    up2  = self.up2(low3)
    return self.merge([up1, up2])

  def make_layer(self, k, inp_dim, out_dim, modules, layer=tf.keras.layers.Conv2D, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
      layers.append(layer(k, out_dim, out_dim, **kwargs))
    return tf.keras.Sequential(layers)

  def make_layer_revr(self, k, inp_dim, out_dim, modules, layer=tf.keras.layers.Conv2D, **kwargs):
      layers = []
      for _ in range(modules - 1):
          layers.append(layer(k, inp_dim, inp_dim, **kwargs))
      layers.append(layer(k, inp_dim, out_dim, **kwargs))
      return tf.keras.Sequential(layers)

  def make_up_layer(self, k, inp_dim, out_dim, modules, layer=tf.keras.layers.Conv2D, **kwargs):
    return self.make_layer(k, inp_dim, out_dim, modules, layer=tf.keras.layers.Conv2D, **kwargs)

  def make_low_layer(self, k, inp_dim, out_dim, modules, layer=tf.keras.layers.Conv2D, **kwargs):
    return self.make_layer(k, inp_dim, out_dim, modules, layer=tf.keras.layers.Conv2D, **kwargs)

  def make_hg_layer(self, k, inp_dim, out_dim, modules, layer=tf.keras.layers.Conv2D, **kwargs):
    return self.make_layer(k, inp_dim, out_dim, modules, layer=tf.keras.layers.Conv2D, **kwargs)

  def make_hg_layer_revr(self, k, inp_dim, out_dim, modules, layer=tf.keras.layers.Conv2D, **kwargs):
    return self.make_layer_revr(k, inp_dim, out_dim, modules, layer=tf.keras.layers.Conv2D, **kwargs)

  def make_pool_layer(self, dim):
    return tf.identity #tf.keras.Sequential([]) # tf.keras.layers.MaxPool2D(strides=2)

  def make_unpool_layer(self, dim):
    return tf.keras.layers.UpSampling2D(2)

  def make_merge_layer(self, dim):
    return tf.keras.layers.Add()

def test():
  n       = 5
  dims    = [256, 256, 384, 384, 384, 512]
  modules = [2, 2, 2, 2, 2, 4]
  return kp_module(n, dims, modules), tf.keras.Input((1, 408, 408, 3))
