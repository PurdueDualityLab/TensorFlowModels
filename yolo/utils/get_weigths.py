import struct

def get_cfg(file_name):
    output = []
    mydict = None
    with open(file_name) as configfile:
        i = 0
        for line in configfile:
            if line.startswith('['):
                mydict = {}
                mydict['_type'] = line.strip('[] \n')
                output.append(mydict)
            elif mydict is not None:
                line, *_ = line.strip().split('#', 1)
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    mydict[k] = v
    return output

def read_n_floats(n, file):
    return list(struct.unpack("f" * n, weights.read(4 * n)))

def read_n_int(n, file, unsigned = False):
    if unsigned:
        return list(struct.unpack("<"+"i" * n, weights.read(4 * n)))
    else:
        return list(struct.unpack("<"+"i" * n, weights.read(4 * n)))

def read_n_long(n, file, unsigned = False):
    if unsigned:
        return list(struct.unpack("<"+"Q" * n, weights.read(8 * n)))
    else:
        return list(struct.unpack("<"+"q" * n, weights.read(8 * n)))

def build_conv(layer_dict):
    print(layer_dict)
    return

def read_file(config, weights):
    major = read_n_int(1, weights)[0]
    minor = read_n_int(1, weights)[0]
    revision = read_n_int(1, weights)[0]

    if ((major * 10 + minor) >= 2):
        print("64 seen")
        iseen = read_n_long(1, weights, unsigned=True)[0]
    else:
        print("32 seen")
        iseen = read_n_int(1, weights, unsigned=True)[0]
    
    print(f"major: {major}")
    print(f"minor: {minor}")
    print(f"revision: {revision}")
    print(f"iseen: {iseen}")

    for layer in config:
        print(layer)

config = get_cfg("yolov3.cfg")
weights = open("yolov3.weights", "rb")
read_file(config, weights)