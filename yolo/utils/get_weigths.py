import struct
import inspect
import numpy as np
import os

"""class objects to to act as place holders to build attributes from config file and store weights"""
class conv_layer(object):
    """modulates shape and stores weights"""
    def __init__(self, ltype = 'convolutional', size = 1, stride = 1, pad = 1, filters = 256, activation = 'linear', batch_normalize = 0, groups = 1, shape = None, **kwargs):
        self.type = ltype
        self.size = int(size)
        self.strides = int(stride)
        self.pad = int(pad) if self.size != 1 else 0
        self.n = int(filters)
        self.activation = activation
        self.batch_normalize = int(batch_normalize)
        self.groups = int(groups)
        self.inshape = shape
        self.nweights = int((self.inshape[-1]/self.groups) * self.n * self.size * self.size)

        print(self.nweights)
        self.baises = None
        self.weights = None
        self.scales = None
        self.rolling_mean = None
        self.rolling_variance = None
        return 
    
    @property
    def shape(self):
        if self.inshape == None:
            return None
        w = len_width(self.inshape[0], self.size, self.pad, self.strides)
        h = len_width(self.inshape[1], self.size, self.pad, self.strides)
        return (w, h, self.n)
    
    def load_weights(self, files):
        bytes_read = self.n

        self.biases = np.array(read_n_floats(self.n, files))

        if self.batch_normalize == 1:
            self.scales = np.array(read_n_floats(self.n, files))
            self.rolling_mean = np.array(read_n_floats(self.n, files))
            self.rolling_variance = np.array(read_n_floats(self.n, files))
            bytes_read += self.n * 3
        
        #used as a guide: https://github.com/thtrieu/darkflow/blob/master/darkflow/dark/convolution.py
        self.weights = np.array(read_n_floats(self.nweights, files))
        self.weights = self.weights.reshape(self.n, self.inshape[-1], self.size, self.size).transpose([2,3,1,0])
        bytes_read += self.nweights
        print(f"weights shape: {self.weights.shape}")
        return bytes_read * 4
    
    def get_weights(self):
        weights_obj = {"weights": self.weights, 
                       "baises": self.biases, 
                       "scales": self.scales, 
                       "means": self.rolling_mean, 
                       "varience": self.rolling_variance}
        return weights_obj

    def __repr__(self):
        return f"shape: {self.shape}"#str(list(self.__dict__.items()))


class upsample_layer(object):
    """object that only modulates shape"""
    def __init__(self, width = None, height = None, channels = None, strides = 2, **kwargs):
        self.inshape = (int(width), int(height), int(channels))
        self.strides = int(strides)
        return

    @property
    def shape(self):
        return (self.inshape[0] * self.strides, self.inshape[1] * self.strides, self.inshape[2])
    
    def get_weights(self):
        return None

    def __repr__(self):
        return f"shape: {self.shape}"


class place_layer(object):
    """object that neither modulates shape or stores weights"""
    def __init__(self, width = None, height = None, channels = None, **kwargs):
        self.shape = (int(width), int(height), int(channels))
        return

    def get_weights(self):
        return None

    def __repr__(self):
        return f"shape: {self.shape}"


'''
testing the size degradation and increase in the encoder and decoder
'''
def get_size(path):
    data = os.stat(path)
    return data.st_size

def len_width(n, f, p, s):
    '''
    n: height or width
    f: kernels height or width
    p: padding
    s: strides height or width
    '''
    return int(((n + 2*p - f)/s) + 1)

def len_width_up(n, f, p, s):
    '''
    n: height or width
    f: kernels height or width
    p: padding
    s: strides height or width
    '''
    return int(((n - 1) * s - 2*p + (f - 1)) + 1)

def get_cfg(file_name):
    output = []
    mydict = None
    with open(file_name) as configfile:
        i = 0
        for line in configfile:
            if line.startswith('['):
                mydict = {}
                mydict['ltype'] = line.strip('[] \n')
                output.append(mydict)
            elif mydict is not None:
                line, *_ = line.strip().split('#', 1)
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    mydict[k] = v
    return output

def read_n_floats(n, bfile):
    return list(struct.unpack("f" * n,  bfile.read(4 * n)))

def read_n_int(n,  bfile, unsigned = False):
    if unsigned:
        return list(struct.unpack("<"+"i" * n,  bfile.read(4 * n)))
    else:
        return list(struct.unpack("<"+"i" * n,  bfile.read(4 * n)))

def read_n_long(n,  bfile, unsigned = False):
    if unsigned:
        return list(struct.unpack("<"+"Q" * n,  bfile.read(8 * n)))
    else:
        return list(struct.unpack("<"+"q" * n,  bfile.read(8 * n)))




def build_conv(layer_dict, file, prevlayer):
    bytes_read = 0 
    if layer_dict['ltype'] == 'convolutional':
        print('\nconvolutional')
        layer = conv_layer(shape = prevlayer.shape, **layer_dict)
        bytes_read = layer.load_weights(file)
        print(layer)
    elif layer_dict['ltype'] == 'net':
        print('\ninput')
        layer = place_layer(**layer_dict)#(layer_dict['width'], layer_dict['height'], layer_dict['channels'])
        print(layer)
    elif layer_dict['ltype'] == 'shortcut' or layer_dict['ltype'] == 'route' or layer_dict['ltype'] == 'yolo':
        print('\nshortcut')
        layer_dict = {"width":prevlayer.shape[0], "height":prevlayer.shape[1], "channels":prevlayer.shape[2]}
        layer = place_layer(**layer_dict)#(layer_dict['width'], layer_dict['height'], layer_dict['channels'])
        print(layer)
    elif layer_dict['ltype'] == 'upsample':
        print('\nupsample')
        layer_dict = {"width":prevlayer.shape[0], "height":prevlayer.shape[1], "channels":prevlayer.shape[2], "strides": layer_dict["stride"]}
        layer = upsample_layer(**layer_dict)#(layer_dict['width'], layer_dict['height'], layer_dict['channels'])
        print(layer)
    else:
        print(layer_dict)
        layer = layer_dict
    return layer, bytes_read

def read_file(config, weights):
    bytes_read = 0

    major = read_n_int(1, weights)[0]
    bytes_read += 4
    minor = read_n_int(1, weights)[0]
    bytes_read += 4
    revision = read_n_int(1, weights)[0]
    bytes_read += 4

    if ((major * 10 + minor) >= 2):
        print("64 seen")
        iseen = read_n_long(1, weights, unsigned=True)[0]
        bytes_read += 8
    else:
        print("32 seen")
        iseen = read_n_int(1, weights, unsigned=True)[0]
        bytes_read += 8
    
    print(f"major: {major}")
    print(f"minor: {minor}")
    print(f"revision: {revision}")
    print(f"iseen: {iseen}")

    net = [None]
    for layer_dict in config:
        #try:
        layer, num_read = build_conv(layer_dict, weights, net[-1])
        if layer_dict["ltype"] != 'yolo' and layer.shape[-1] != 255:
            net.append(layer)
        
        bytes_read += num_read
        #except:
            #break
    return net, bytes_read


def load_weights(config_file, weights_file):
    config = get_cfg(config_file)
    size = get_size(weights_file)
    weights = open(weights_file, "rb")
    print(weights)
    net, bytes_read = read_file(config, weights)
    print(f"bytes_read: {bytes_read}, original_size: {size}, final_position: {weights.tell()}")
    if (bytes_read != size):
        print("error: could not read the entire weights file")
    return net, bytes_read

config = "yolov3.cfg"
weights = "yolov3.weights"
net,_ = load_weights(config, weights)