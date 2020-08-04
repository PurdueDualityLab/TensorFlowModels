"""
This file contains the code to load parsed weights that are in the DarkNet
format into TensorFlow layers
"""
import itertools

def interleve_weights(block):
    """merge weights to fit the DarkResnet block style"""
    if len(block) == 0:
        return []
    weights_temp = []
    for layer in block:
        weights = layer.get_weights()
        weights = [tuple(weights[0:3]), tuple(weights[3:])]
        weights_temp.append(weights)
    top, bottom = tuple(zip(*weights_temp))
    weights = list(itertools.chain.from_iterable(top)) + \
        list(itertools.chain.from_iterable(bottom))
    return weights


def get_darknet53_tf_format(net, only_weights=True):
    """convert weights from darknet sequntial to tensorflow weave, Darknet53 Backbone"""
    combo_blocks = []
    for i in range(2):
        layer = net.pop(0)
        combo_blocks.append(layer)
    # ugly code i will document, very tired
    encoder = []
    while len(net) != 0:
        blocks = []
        layer = net.pop(0)
        while layer._type != "shortcut":
            blocks.append(layer)
            layer = net.pop(0)
        encoder.append(blocks)
    new_net = combo_blocks + encoder
    weights = []
    if only_weights:
        for block in new_net:
            if type(block) != list:
                weights.append(block.get_weights())
            else:
                weights.append(interleve_weights(block))
    print("converted/interleved weights for tensorflow format")
    return new_net, weights

def get_tiny_weights(backbone, encoder, layers):
    return

def _load_weights_dnBackbone(backbone, encoder, mtype = "darknet53"):
    # get weights for backbone
    if mtype == "darknet53":
        encoder, weights_encoder = get_darknet53_tf_format(encoder[:])
    elif mtype == "darknet53":
        encoder, weights_encoder = get_tiny_tf_format(encoder[:])

    # set backbone weights
    print(f"\nno. layers: {len(backbone.layers)}, no. weights: {len(weights_encoder)}")
    _set_darknet_weights(backbone, weights_encoder)

    print(f"\nsetting backbone.trainable to: {backbone.trainable}\n")
    return


def _load_weights_dnHead(head, decoder):
    # get weights for head
    decoder, weights_decoder = get_decoder_weights(decoder)

    # set detection head weights
    print(f"\nno. layers: {len(head.layers)}, no. weights: {len(weights_decoder)}")
    _set_darknet_weights(head, weights_decoder)

    print(f"\nsetting head.trainable to: {head.trainable}\n")
    return

# DEBUGGING
def print_layer_shape(layer):
    weights = layer.get_weights()
    for item in weights:
        print(item.shape)
    return

def _set_darknet_weights(model, weights_list):
    for i, (layer, weights) in enumerate(zip(model.layers, weights_list)):
        print(f"loaded weights for layer: {i}  -> name: {layer.name}",sep='      ',end="\r")
        #print_layer_shape(layer)
        layer.set_weights(weights)
    model.trainable = False
    return

def split_decoder(lst):
    decoder = []
    outputs = []
    for layer in lst:
        if layer._type == 'yolo':
            outputs.append(decoder.pop())
            outputs.append(layer)
        else:
            decoder.append(layer)
    return decoder, outputs

def get_decoder_weights(decoder):
    layers = [[]]
    block = []
    weights = []

    decoder, head = split_decoder(decoder)

    # get decoder weights and group them together
    for i, layer in enumerate(decoder):
        if layer._type == "route" and decoder[i - 1]._type != 'maxpool':
            layers.append(block)
            block = []
        elif (layer._type == "route" and decoder[i - 1]._type == "maxpool") or layer._type == "maxpool":
            continue
        elif layer._type == "convolutional":
            block.append(layer)
        else:
            layers.append([])
    if len(block) > 0:
        layers.append(block)

    # interleve weights for blocked layers
    for layer in layers:
        print(layer)
        weights.append(interleve_weights(layer))

    # get weights for output detection heads
    for layer in reversed(head):
        if layer != None and layer._type == "convolutional":
            weights.append(layer.get_weights())

    return layers, weights
