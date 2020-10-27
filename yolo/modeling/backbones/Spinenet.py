import tensorflow as tf 
import tensorflow.keras as ks 
import tensorflow.keras.backend as K

class ResidualBlock(ks.layers.Layer):
    def __init__(self, 
                filters, 
                dilation_rate = (1,1), 
                kernel_initializer = 'glorot_uniform', 
                kernel_regularizer = None, 
                bias_regularizer = None, 
                momentum = 0.99, 
                epsilon = 0.001, 
                downsample = False,
                use_bias = False, 
                use_sync = False):
        # parameters Conv2D
        self._filters = filters
        self._dilation_rate = dilation_rate
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._use_bias = use_bias
        self._use_sync = use_sync

        # parameters Batch Norm
        if K.image_data_format() == "channels_last":
            # channels_last: (batch_size, height, width, channels)
            self._axis = -1
        else:
            # not channels_last: (batch_size, channels, height, width)
            self._axis = 1

        self._momentum = momentum 
        self._epsilon = epsilon 

        # downsample
        self._downsample = downsample

        #layers
        if downsample:
            # syncs the normalization so that it to the normalization stats across all devices in use
            # done so the normalization is the same across all devices used in distributed strategy
            # self.bn = tf.keras.layers.experimental.SyncBatchNormalization
            self._strides = (2,2)
        else:
            self._strides = (1,1)

        if use_sync: 
            self.bn = tf.keras.layers.experimental.SyncBatchNormalization
        else: 
            self.bn = ks.layers.BatchNormalization
        self.conv = ks.layers.Conv2D
        self._activation = ks.layers.ReLU()

        super(ResidualBlock, self).__init__()
        pass

    def build(self, ishape):
        if self._downsample:
            self._dsampleconv = self.conv(self._filters, 
                                          kernel_size = (1,1), 
                                          strides = (2,2), 
                                          padding = "same", 
                                          dilation_rate = self._dilation_rate,
                                          use_bias=self._use_bias,
                                          kernel_initializer = self._kernel_initializer, 
                                          kernel_regularizer = self._kernel_regularizer, 
                                          bias_regularizer = self._bias_regularizer)
            self._dsamplebn = self.bn(axis = self._axis, 
                                      momentum = self._momentum, 
                                      epsilon = self._epsilon)

        self._conv1 = self.conv(self._filters, 
                                        kernel_size = (3,3), 
                                        strides = self._strides, # -> 2 if downsample, 1 if not
                                        padding = "same", 
                                        dilation_rate = self._dilation_rate,
                                        use_bias=self._use_bias,
                                        kernel_initializer = self._kernel_initializer, 
                                        kernel_regularizer = self._kernel_regularizer, 
                                        bias_regularizer = self._bias_regularizer)
        self._bn1 = self.bn(axis = self._axis, 
                            momentum = self._momentum, 
                            epsilon = self._epsilon)
        
        self._conv2 = self.conv(self._filters, 
                                        kernel_size = (3,3), 
                                        strides = (1,1), 
                                        padding = "same", 
                                        dilation_rate = self._dilation_rate,
                                        use_bias=self._use_bias,
                                        kernel_initializer = self._kernel_initializer, 
                                        kernel_regularizer = self._kernel_regularizer, 
                                        bias_regularizer = self._bias_regularizer)
        self._bn2 = self.bn(axis = self._axis, 
                            momentum = self._momentum, 
                            epsilon = self._epsilon)

        self._add = ks.layers.Add()
        pass

    def call(self, inputs):
        sample = inputs
        if self._downsample:
            sample = self._dsampleconv(sample)
            sample = self._dsamplebn(sample)

        x = self._conv1(inputs)
        x = self._bn1(x)
        x = self._activation(x)

        x = self._conv2(x)
        x = self._bn2(x)

        x = self._add([x, sample])
        return self._activation(x)


class BottleNeckBlock(ks.layers.Layer):
    def __init__(self, 
                filters, 
                dilation_rate = (1,1), 
                kernel_initializer = 'glorot_uniform', 
                kernel_regularizer = None, 
                bias_regularizer = None, 
                momentum = 0.99, 
                epsilon = 0.001,
                projection = False,
                downsample = False, 
                use_bias = False, 
                use_sync = False):
        # parameters Conv2D
        self._filters = filters
        self._dilation_rate = dilation_rate
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._use_bias = use_bias
        self._use_sync = use_sync

        # parameters Batch Norm
        if K.image_data_format() == "channels_last":
            # channels_last: (batch_size, height, width, channels)
            self._axis = -1
        else:
            # not channels_last: (batch_size, channels, height, width)
            self._axis = 1

        self._momentum = momentum 
        self._epsilon = epsilon 

        self._projection = projection

        if downsample:
            self._strides = (2,2)
        else:
            self._strides = (1,1)

        #layers
        if use_sync: 
            # ensure all batch norm layers normalize using all the data, and the same noramalization stats/parameters, global
            self.bn = tf.keras.layers.experimental.SyncBatchNormalization
        else: 
            # allow batch norm to normalize according to the data that flows through it only, not realted in any way to the surrounding batch norm layers  
            self.bn = ks.layers.BatchNormalization
        self.conv = ks.layers.Conv2D
        self._activation = ks.layers.ReLU()

        super(BottleNeckBlock, self).__init__()
        pass

    def build(self, input_shape):
        if self._projection:
            # project lower dimention input to high dimentions, used after each down sampling resnet block
            self._projconv = self.conv(self._filters * 4, 
                                       kernel_size = (1,1), 
                                       strides = self._strides, # -> 2 if downsample, 1 if not
                                       padding = "same", 
                                       dilation_rate = self._dilation_rate,
                                       use_bias=self._use_bias,
                                       kernel_initializer = self._kernel_initializer, 
                                       kernel_regularizer = self._kernel_regularizer, 
                                       bias_regularizer = self._bias_regularizer)
            self._projbn = self.bn(axis = self._axis, 
                                   momentum = self._momentum, 
                                   epsilon = self._epsilon)
        self._conv1 = self.conv(self._filters, 
                                kernel_size = (1,1), 
                                strides = self._strides, # -> 2 if downsample, 1 if not
                                padding = "same", 
                                dilation_rate = self._dilation_rate,
                                use_bias=self._use_bias,
                                kernel_initializer = self._kernel_initializer, 
                                kernel_regularizer = self._kernel_regularizer, 
                                bias_regularizer = self._bias_regularizer)
        self._bn1 = self.bn(axis = self._axis, 
                            momentum = self._momentum, 
                            epsilon = self._epsilon)

        self._conv2 = self.conv(self._filters, 
                                kernel_size = (3,3), 
                                strides = (1,1), 
                                padding = "same", 
                                dilation_rate = self._dilation_rate,
                                use_bias=self._use_bias,
                                kernel_initializer = self._kernel_initializer, 
                                kernel_regularizer = self._kernel_regularizer, 
                                bias_regularizer = self._bias_regularizer)
        self._bn2 = self.bn(axis = self._axis, 
                            momentum = self._momentum, 
                            epsilon = self._epsilon)
        
        self._conv3 = self.conv(self._filters * 4, 
                                kernel_size = (1,1), 
                                strides = (1,1), 
                                padding = "same", 
                                dilation_rate = self._dilation_rate,
                                use_bias=self._use_bias,
                                kernel_initializer = self._kernel_initializer, 
                                kernel_regularizer = self._kernel_regularizer, 
                                bias_regularizer = self._bias_regularizer)
        self._bn3 = self.bn(axis = self._axis, 
                            momentum = self._momentum, 
                            epsilon = self._epsilon)

        self._add = ks.layers.Add()

        super(BottleNeckBlock, self).build(input_shape)
        pass

    def call(self, inputs):
        sample = inputs
        if self._projection:
            sample = self._projconv(sample)
            sample = self._projbn(sample)
            
        x = self._conv1(inputs)
        x = self._bn1(x)
        x = self._activation(x)

        x = self._conv2(x)
        x = self._bn2(x)
        x = self._activation(x)

        x = self._conv3(x)
        x = self._bn3(x)

        x = self._add([x, sample])
        return self._activation(x)
    

class normalized_conv(ks.layers.Layer):
    def __init__(self,
                 filters, 
                 kernel_size= (1,1),
                 strides = (1,1),
                 padding = 'same',       
                 kernel_initializer = 'glorot_uniform', 
                 kernel_regularizer = None, 
                 bias_regularizer = None, 
                 use_bias = False,
                 bn_axis = -1, 
                 momentum = 0.99, 
                 epsilon = 0.001, 
                 activation = 'relu',
                 use_sync = False):
        
        #conv params 
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._use_bias = use_bias
        self._activation = activation
    
        #normalization
        self._bn_axis = bn_axis
        self._momentum = momentum
        self._epsilon = epsilon

        if use_sync: 
            self._bn = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            self._bn = ks.layers.BatchNormalization
        
        super(normalized_conv, self).__init__()
        return
    

    def build(self, input_shape):
        self.conv = ks.layers.Conv2D(self._filters,
                                     kernel_size=self._kernel_size,
                                     strides = self._strides,
                                     use_bias = self._use_bias, 
                                     padding = self._padding,
                                     kernel_initializer = self._kernel_initializer,
                                     kernel_regularizer = self._kernel_regularizer, 
                                     bias_regularizer = self._bias_regularizer)
        self.bn = self._bn(momentum = self._momentum, epsilon = self._epsilon, axis = self._bn_axis)
        if self._activation == None:
            self._activation_fn = ks.layers.Activation(activation='linear')
        else:
            self._activation_fn = ks.layers.Activation(activation=self._activation)

        super(normalized_conv, self).build(input_shape)
        return 
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self._activation_fn(x)
        return x


class resample(ks.layers.Layer):
    def __init__ (self, 
                  curr_shape, 
                  target_shape, 
                  resampling_alpha = 0.5,
                  input_fn = "bottleneck",  
                  target_fn = "bottleneck",   
                  kernel_initializer = 'glorot_uniform', 
                  kernel_regularizer = None, 
                  bias_regularizer = None, 
                  use_bias = False,
                  bn_axis = -1, 
                  momentum = 0.99, 
                  epsilon = 0.001,
                  activation = 'relu'):
    
        # conv
        self._curr_shape = curr_shape 
        self._target_shape = target_shape
        self._target_fn = target_fn
        self._resampling_alpha = resampling_alpha
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._use_bias = use_bias
        self._bias_regularizer = bias_regularizer

        if input_fn == "bottleneck":
            self._bottleneck = int(curr_shape[2]/4 * self._resampling_alpha)
        else:
            self._bottleneck = int(curr_shape[2] * self._resampling_alpha)

        # norm
        self._bn_axis = bn_axis
        self._momentum = momentum
        self._epsilon = epsilon

        # activation
        self._activation = activation

        # layers 
        layer = resample

        super(resample, self).__init__()
        return

    def build(self, input_shape):
        self.conv1 = normalized_conv(self._bottleneck, 
                                     kernel_size = (1,1),
                                     strides = (1,1),         
                                     kernel_initializer = self._kernel_initializer, 
                                     kernel_regularizer = self._kernel_regularizer, 
                                     bias_regularizer = self._bias_regularizer, 
                                     use_bias = self._use_bias,
                                     bn_axis = self._bn_axis, 
                                     momentum = self._momentum, 
                                     epsilon = self._epsilon,
                                     activation = self._activation)
        
        self.spacial_sample = ks.Sequential()
        tf.print(self._curr_shape, self._target_shape)
        if self._curr_shape[0] > self._target_shape[0]:
            self.spacial_sample.add(normalized_conv(self._bottleneck, 
                                         kernel_size = (3,3),
                                         strides = (2,2),
                                         padding = 'same',          
                                         kernel_initializer = self._kernel_initializer, 
                                         kernel_regularizer = self._kernel_regularizer, 
                                         bias_regularizer = self._bias_regularizer, 
                                         use_bias = self._use_bias,
                                         bn_axis = self._bn_axis, 
                                         momentum = self._momentum, 
                                         epsilon = self._epsilon,
                                         activation = self._activation))
            print(self._curr_shape[0])
            input_wid = self._curr_shape[0] / 2
            print(input_wid)
            while input_wid > self._target_shape[0]:
                self.spacial_sample.add(ks.layers.MaxPool2D(pool_size = 3, strides = 2, padding = "same"))
                input_wid /= 2
                print(input_wid,  self._target_shape[0])
        elif self._curr_shape[0] < self._target_shape[0]:
            shape = self._target_shape[0]//self._curr_shape[0]
            self.spacial_sample.add(ks.layers.UpSampling2D(size = (shape,shape), interpolation="nearest"))

            upsampled_shape = self._curr_shape.as_list()
            upsampled_shape[0] = upsampled_shape[0] * shape
            upsampled_shape[1] = upsampled_shape[1] * shape
            if upsampled_shape[0] != self._target_shape[0]:
                padding = self._target_shape[0] - upsampled_shape[0] 
                self.spacial_sample.add(ks.layers.ZeroPadding2D(padding = ((padding//2, tf.cast(tf.math.ceil(padding/2), dtype = tf.int32)), (padding//2, tf.cast(tf.math.ceil(padding/2), dtype = tf.int32)))))
                tf.print("padding_shape", padding)
                tf.print("padding_shape", shape)


        # input_shape1 = list(input_shape)
        # input_shape1[-1] = self._bottleneck
        # self.spacial_sample.build(tuple(input_shape1))
        # self.spacial_sample.summary()

        if self._target_fn == "bottleneck":
            scale = 4
        else:
            scale = 1

        self.conv2 = normalized_conv(self._target_shape[2] * scale, 
                                     kernel_size = (1,1),
                                     strides = (1,1),
                                     padding = 'same',          
                                     kernel_initializer = self._kernel_initializer, 
                                     kernel_regularizer = self._kernel_regularizer, 
                                     bias_regularizer = self._bias_regularizer, 
                                     use_bias = self._use_bias,
                                     bn_axis = self._bn_axis, 
                                     momentum = self._momentum, 
                                     epsilon = self._epsilon,
                                     activation = None) 
        
        super(resample, self).build(input_shape)
        return
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.spacial_sample(x)
        x = self.conv2(x)
        return x

class activated_sum(ks.layers.Layer):
    def __init__(self, activation = 'relu'):
        self._activation = activation
        super(activated_sum, self).__init__()
        return
    
    def build(self, input_shape):
        self._add = ks.layers.Add()
        self._activation_fn = ks.layers.Activation(activation=self._activation)
        super(activated_sum, self).build(input_shape)
        return
    
    def call(self, in1, in2):
        x = self._add([in1, in2])
        x = self._activation_fn(x)
        return x


input_shape = (608, 608, 3)

blocks = {
    'basic': ks.layers.Conv2D, 
    'residual': ResidualBlock, 
    'bottleneck': BottleNeckBlock
}

SCALING_MAPS = {
    '49': {
        'endpoints_num_filters': 256, 
        'filter_size_scale': 1.0,
        'resample_alpha': 0.5,
        'block_repeats': 1,
    },
}

#section 3.3 of paper
Level_dim_size_Spinenet = {
    2:64, 
    3:128, 
    4:256, 
    5:256,
    6:256,
    7:256
}

# dubug
output_dims = {
    2: (input_shape[0]//(2 ** 2), input_shape[1]//(2 ** 2), Level_dim_size_Spinenet[2]),
    3: (input_shape[0]//(2 ** 3), input_shape[1]//(2 ** 3), Level_dim_size_Spinenet[3]),
    4: (input_shape[0]//(2 ** 4), input_shape[1]//(2 ** 4), Level_dim_size_Spinenet[4]),
    5: (input_shape[0]//(2 ** 5), input_shape[1]//(2 ** 5), Level_dim_size_Spinenet[5]),
    6: (input_shape[0]//(2 ** 6), input_shape[1]//(2 ** 6), Level_dim_size_Spinenet[6]),
    7: (input_shape[0]//(2 ** 7), input_shape[1]//(2 ** 7), Level_dim_size_Spinenet[7])
}

Level_cores_dim_size_Resnet = {
    2:64, 
    3:128, 
    4:256, 
    5:512
}

"""
L_(1 - 7) distictions of L_i -> the height and width of the image output 
    from a block with Li is (input_height, input_width)/(2 ^ i)
    (level(L_i), (input_index1, input_index2), output)  
"""
SPINENET_struct = [
    #stem
    (2, 'basic', None, False),    
    (2, 'basic',(0), False),

    #scale permuted    
    (2, 'bottleneck', (0, 1), False),
    (4, 'residual', (0, 1), False),
    (3, 'bottleneck', (2, 3), False),
    (4, 'bottleneck', (2, 4), False),
    (6, 'residual', (3, 5), False),
    (4, 'bottleneck', (3, 5), False),
    (5, 'residual', (6, 7), False),
    (7, 'residual', (6, 8), False),
    (5, 'bottleneck', (8, 9), False),
    (5, 'bottleneck', (8, 10), False),
    (4, 'bottleneck', (5, 10), True),
    (3, 'bottleneck', (4, 10), True),
    (5, 'bottleneck', (7, 12), True),
    (7, 'bottleneck', (5, 14), True),
    (6, 'bottleneck', (12, 14), True),
]

RESNET_struct = []

print(output_dims)


class SmallSpineNet(ks.Model):
    def __init__(self, 
                 model_name= '49',
                 use_share_conv = False, 
                 input_shape = (None, 608, 608, 3),
                 kernel_initializer = 'glorot_uniform', 
                 kernel_regularizer = None, 
                 bias_regularizer = None, 
                 momentum = 0.99, 
                 epsilon = 0.001,
                 activation = 'relu',
                 projection = False,
                 downsample = False, 
                 use_bias = False, 
                 use_sync = True, 
                 min_level = 3, 
                 max_level = 7, 
                 share_conv_endpoints = 256):
        
        self._model_name = model_name
        self._use_share_conv = use_share_conv
        self._input_shape = input_shape
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._momentum = momentum
        self._epsilon = epsilon
        self._projection = projection
        self._downsample = downsample
        self._use_bias = use_bias
        self._use_sync = use_sync
        self._activation = activation
        self._scale = 1 #0.65
        self._ralpha = 0.5
        self._min_level = min_level
        self._max_level = max_level
        self._share_conv_endpoints = share_conv_endpoints
        self.count = 0

        self._blocks = {
                        'residual': ResidualBlock, 
                        'bottleneck': BottleNeckBlock
                        }

        self._spine_struct = [
                                #scale permuted    
                                (2, 'bottleneck', (0, 1), False),
                                (4, 'residual', (0, 1), False),
                                (3, 'bottleneck', (2, 3), False),
                                (4, 'bottleneck', (2, 4), False),
                                (6, 'residual', (3, 5), False),
                                (4, 'bottleneck', (3, 5), False),
                                (5, 'residual', (6, 7), False),
                                (7, 'residual', (6, 8), False),
                                (5, 'bottleneck', (8, 9), False),
                                (5, 'bottleneck', (8, 10), False),
                                (4, 'bottleneck', (5, 10), True),
                                (3, 'bottleneck', (4, 10), True),
                                (5, 'bottleneck', (7, 12), True),
                                (7, 'bottleneck', (5, 14), True),
                                (6, 'bottleneck', (12, 14), True)
                            ]

        self._spine_stem = [
                            (2, 'input', None, False),
                            (2, 'bottleneck', (0, 1), False)
        ]

        inputs = ks.layers.Input(shape = self._input_shape[1:])
        layers = self._build_stem(inputs)
        outputs = self._build_spine_block(layers)
        outputs = self._build_endpoints(outputs)

        super().__init__(inputs = [inputs], outputs = outputs)
        return 

    def _build_stem(self, inputs):
        layers = []
        x = normalized_conv(64, 
                            kernel_size = (7,7), 
                            strides = (2,2), 
                            padding = 'same', 
                            kernel_initializer = self._kernel_initializer,
                            kernel_regularizer = self._kernel_regularizer,
                            bias_regularizer = self._bias_regularizer, 
                            use_bias = self._use_bias, 
                            momentum = self._momentum, 
                            epsilon = self._epsilon, 
                            activation = self._activation)(inputs)

        x1 = ks.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        x2 = BottleNeckBlock(Level_dim_size_Spinenet[2], projection= True)(x1)
        
        #x1 = tf.identity(x1, name = f"input_layer_stem")
        #x2 = tf.identity(x2, name = f"stem2_layer")

        layers.append(x1)
        layers.append(x2)

        print(f"current layer: (CONV stem, {0})")
        print(f"input: (image, {self._input_shape})")
        print(f"layer outshape: {x1.shape} \n")

        print(f"current layer: (bottleneck stem, {1})")
        print(f"input: (downsampled, {x1.shape})")
        print(f"layer outshape: {x2.shape} \n")

        return layers
    
    def _build_spine_block(self, layers):
        net_connections = [0] * len(layers)
        endpoints = {}

        for level, layer, connections, output  in self._spine_struct:     
            filters = int(Level_dim_size_Spinenet[level] * self._scale)
            input1 = layers[connections[0]]
            input2 = layers[connections[1]]

            if connections[0] - 2 < 0:
                infunc1 = "CONV"
            else:
                infunc1 = self._spine_struct[connections[0] - 2][1]
            
            if connections[1] - 2 < 0:
                infunc2 = "CONV"
            else:
                infunc2 = self._spine_struct[connections[1] - 2][1]

            x, tshape, outshape = self._build_connections(level, 
                                                          layer, 
                                                          input1, 
                                                          input2, 
                                                          infunc1, 
                                                          infunc2, 
                                                          connections)

            if output:
                #build fpn
                for i , (layer_output, num_connections) in enumerate(zip(layers, net_connections)):
                    if num_connections == 0 and (layer_output.shape[1] == tshape[0] and layer_output.shape[3] == outshape[2]):
                        # if num_connections == 0 -> does the layer have any outgoing connections
                        # if layer_output.shape[1] == tshape[0] -> if the blocks exist on the same level, so the widths are the same
                        # if layer_output.shape[3] == input1.shape[3] -> if the layer has the same depth as the output layer
                        # question why? ask on friday
                        x.append(layer_output)
                        net_connections[i] += 1

            x = tf.math.add_n(x)
            x = ks.layers.Activation(activation=self._activation)(x)

            # build endpoints is used to construct output connections for each (width, height, depth)

            block = self._blocks[layer](filters)(x)

            if output:
                endpoints[level] = block

            #block = tf.identity(block, name = f"{layer}_{self.count}")
            print(f"current layer: ({layer}, {self.count + 2})")
            print(f"input1: ({connections[0]}, {infunc1}, {input1.shape})")
            print(f"input2: ({connections[1]}, {infunc2}, {input2.shape})")
            print(f"layer outshape: {outshape}, output layer? {output}\n")

            layers.append(block)
            net_connections.append(0)
            self.count += 1
        #print(endpoints)
        return endpoints
    
    def _build_connections(self, level, layer, input1, input2, infunc1, infunc2, connections):
        target_shape = (self._input_shape[1]//(2 ** level), self._input_shape[2]//(2 ** level), int(Level_dim_size_Spinenet[level] * self._scale))
        input1_shape = input1.shape[1:]
        input2_shape = input2.shape[1:]
        #print(input2_shape, input1_shape, target_shape, layer, level)

        x1 = resample(curr_shape=input1_shape, 
                      input_fn = infunc1,
                      target_shape=target_shape, 
                      resampling_alpha = self._ralpha, 
                      target_fn = layer)(input1)

        x2 = resample(curr_shape=input2_shape, 
                      input_fn = infunc2,
                      target_shape=target_shape, 
                      resampling_alpha = self._ralpha, 
                      target_fn = layer)(input2)

        tf.print(x1.shape)
        tf.print(x2.shape)
        #x1 = tf.identity(x1, name = f"resample_in1_{self.count}")
        #x2 = tf.identity(x2, name = f"resample_in2_{self.count}")
        return [x1, x2], target_shape, x1.shape[1:]

    def _build_endpoints(self, net):
        endpoints = {}
        for level in range(self._min_level, self._max_level + 1):
            x = net[level]
            if self._use_share_conv:
                x = normalized_conv(self._share_conv_endpoints, 
                                    kernel_size = (1,1), 
                                    strides = (1,1), 
                                    padding = 'same', 
                                    kernel_initializer = self._kernel_initializer,
                                    kernel_regularizer = self._kernel_regularizer,
                                    bias_regularizer = self._bias_regularizer, 
                                    use_bias = self._use_bias, 
                                    momentum = self._momentum, 
                                    epsilon = self._epsilon, 
                                    activation = self._activation)(x)
            endpoints[str(level)] = x
        #print(endpoints)
        return endpoints