import tensorflow as tf

from official.vision.beta.modeling.layers.nn_blocks import ResidualBlock
# from yolo.modeling.layers.nn_blocks import DarkResidual as ResidualBlock

def ConvBNRelu2D(*args, **kwargs):
  return tf.keras.layers.Conv2D(
    *args,
    activation='relu',
    use_bias=True,
    **kwargs
  )

class HourglassBlock(tf.keras.layers.Layer):
  def __init__(
    self, n, dims, modules, k=0, **kwargs
  ):
    super().__init__()

    self.n   = n
    self.k   = k
    self.modules = modules
    self.dims = dims

    self._kwargs = kwargs

  def build(self, input_shape):
    modules = self.modules
    dims = self.dims
    k = self.k
    kwargs = self._kwargs

    curr_mod = modules[k]
    next_mod = modules[k + 1]

    curr_dim = dims[k + 0]
    next_dim = dims[k + 1]

    self.up1  = self.make_up_layer(
      3, curr_dim, curr_dim, curr_mod, **kwargs
    )
    self.max1 = self.make_pool_layer(curr_dim)
    self.low1 = self.make_hg_layer(
      3, curr_dim, next_dim, curr_mod, **kwargs
    )
    self.low2 = type(self)(
      self.n, dims, modules, k=k+1, **kwargs
    ) if self.n - k > 1 else \
    self.make_low_layer(
      3, next_dim, next_dim, next_mod, **kwargs
    )
    self.low3 = self.make_hg_layer_revr(
      3, next_dim, curr_dim, curr_mod, **kwargs
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

  def make_layer(self, k, inp_dim, out_dim, modules, **kwargs):
    layers = [ResidualBlock(out_dim, 1, **kwargs)]
    for _ in range(1, modules):
      layers.append(ResidualBlock(out_dim, 1, **kwargs))
    return tf.keras.Sequential(layers)

  def make_layer_revr(self, k, inp_dim, out_dim, modules, **kwargs):
      layers = []
      for _ in range(modules - 1):
          layers.append(ResidualBlock(inp_dim, 1, **kwargs)) # inp_dim is not a bug
      layers.append(ResidualBlock(out_dim, 1, **kwargs))
      return tf.keras.Sequential(layers)

  def make_up_layer(self, k, inp_dim, out_dim, modules, **kwargs):
    return self.make_layer(k, inp_dim, out_dim, modules, **kwargs)

  def make_low_layer(self, k, inp_dim, out_dim, modules, **kwargs):
    return self.make_layer(k, inp_dim, out_dim, modules, **kwargs)

  def make_hg_layer(self, k, inp_dim, out_dim, modules, **kwargs):
    return self.make_layer(k, inp_dim, out_dim, modules, **kwargs)

  def make_hg_layer_revr(self, k, inp_dim, out_dim, modules, **kwargs):
    return self.make_layer_revr(k, inp_dim, out_dim, modules, **kwargs)

  def make_pool_layer(self, dim):
    return tf.identity #tf.keras.Sequential([]) # tf.keras.layers.MaxPool2D(strides=2)

  def make_unpool_layer(self, dim):
    return tf.keras.layers.UpSampling2D(2)

  def make_merge_layer(self, dim):
    return tf.keras.layers.Add()

kp_module = HourglassBlock

def test():
  n       = 5
  dims    = [256, 256, 384, 384, 384, 512]
  modules = [2, 2, 2, 2, 2, 4]
  # n = 1
  # dims = [384, 512]
  # modules = [2, 4]
  return kp_module(n, dims, modules), tf.keras.Input((512, 512, 256))
