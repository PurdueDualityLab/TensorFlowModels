import tensorflow as tf
# from centernet.modeling.layers.nn_blocks import kp_module
# from official.vision.beta.modeling.layers.nn_blocks import ResidualBlock
from centernet.modeling.backbones.residual import ResidualBlock

class CenterNetBackbone(tf.keras.Model):
    """
    CenterNet Hourglass backbone
    """
    def __init__(self, 
                 order, 
                 filter_sizes, 
                 rep_sizes,
                 n_stacks=2,
                 pre_layers=None,
                 **kwargs):
        """
        Args:
            order: integer, number of downsampling (and subsequent upsampling) 
                   steps per hourglass module
            filter_sizes: list of filter sizes for Residual blocks
            rep_sizes: list of residual block repetitions per down/upsample
            n_stacks: integer, number of hourglass modules in backbone
            pre_layers: tf.keras layer to process input before stacked hourglasses 
        """
        self._n_stacks = n_stacks
        self._pre_layers = pre_layers
        self._order = order
        self._filter_sizes = filter_sizes
        self._rep_sizes = rep_sizes

        super().__init__(**kwargs)
    
    def build(self, input_shape):
        if self._pre_layers is None:
            self._pre_layers = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=128, kernel_size=7, strides=2, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                ResidualBlock(filters=256, use_projection=True, strides=2) # shape errors happening
            ])

        # Create hourglass stacks
        self.hgs = tf.keras.Sequential([Hourglass(order=self._order, 
                                            filter_sizes=self._filter_sizes,
                                            rep_sizes=self._rep_sizes) for _ in range(self._n_stacks)])

        super(CenterNetBackbone, self).build(input_shape)

    def call(self, x):
        x = self._pre_layers(x)

        # TODO: add intermediate layers
        return self.hgs(x)

class Hourglass(tf.keras.Model):
    """
    Hourglass module
    """
    def __init__(self, 
                 order, 
                 filter_sizes, 
                 rep_sizes,
                 strides=1,
                 **kwargs):
        """
        Args:
            order: integer, number of downsampling (and subsequent upsampling) steps 
            filter_sizes: list of filter sizes for Residual blocks
            rep_sizes: list of residual block repetitions per down/upsample
            strides: integer, stride parameter to the Residual block
        """
        self._order = order
        self._filter_sizes = filter_sizes
        self._rep_sizes = rep_sizes
        self._strides = strides

        self._filters = filter_sizes[0]
        self._reps = rep_sizes[0]
        
        super(Hourglass, self).__init__()
    
    def build(self, input_shape):
        if self._order == 1:
            # base case, residual block repetitions in most inner part of hourglass 
            blocks = [ResidualBlock(filters=self._filters, 
                                    strides=self._strides,
                                    use_projection=True) for _ in range(self._reps)]
            self.blocks = tf.keras.Sequential(blocks)

        else:
            # outer hourglass structures
            main_block = [ResidualBlock(filters=self._filters, 
                                        strides=self._strides,
                                        use_projection=True) for _ in range(self._reps)]
            side_block = [ResidualBlock(filters=self._filters, 
                                        strides=self._strides,
                                        use_projection=True) for _ in range(self._reps)]
            self.pool = tf.keras.layers.MaxPool2D(pool_size=2)

            # recursively define inner hourglasses
            self.inner_hg = Hourglass(order=self._order-1, 
                                      filter_sizes=self._filter_sizes[1:], 
                                      rep_sizes=self._rep_sizes[1:], 
                                      stride=self._strides)
            end_block = [ResidualBlock(filters=self._filters, 
                                       strides=self._strides,
                                       use_projection=True) for _ in range(self._reps)]
            self.upsample_layer = tf.keras.layers.UpSampling2D(size=2, 
                                                               interpolation='nearest')
            
            self.main_block = tf.keras.Sequential(main_block, name="Main_Block")
            self.side_block = tf.keras.Sequential(side_block, name="Side_Block")
            self.end_block = tf.keras.Sequential(end_block, name="End_Block")
        super(Hourglass, self).build(input_shape)
    
    def call(self, x):
        if self._order == 1:
            return self.blocks(x)
        else:
            x_pre_pooled = self.main_block(x)
            x_side = self.side_block(x_pre_pooled)
            x_pooled = self.pool(x_pre_pooled)
            inner_output = self.inner_hg(x_pooled)
            hg_output = self.end_block(inner_output)
            
            return self.upsample_layer(hg_output) + x_side