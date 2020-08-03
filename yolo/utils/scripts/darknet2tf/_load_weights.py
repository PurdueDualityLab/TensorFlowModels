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


def _load_weights_dnBackbone(backbone, encoder):
    # get weights for backbone
    encoder, weights_encoder = get_darknet53_tf_format(encoder[:])

    # set backbone weights
    print(f"\nno. layers: {len(backbone.layers)}, no. weights: {len(weights_encoder)}")
    _set_darknet_weights(backbone, weights_encoder)

    print(f"\nsetting backbone.trainable to: {backbone.trainable}\n")
    return


def _load_weights_dnHead(head, decoder, outputs):
    # get weights for head
    decoder, weights_decoder = get_decoder_weights(decoder, outputs)

    # set detection head weights
    print(f"\nno. layers: {len(head.layers)}, no. weights: {len(weights_decoder)}")
    _set_darknet_weights(head, weights_decoder)

    print(f"\nsetting head.trainable to: {head.trainable}\n")
    return


def _set_darknet_weights(model, weights_list):
    for i, (layer, weights) in enumerate(zip(model.layers, weights_list)):
        print(f"loaded weights for layer: {i}  -> name: {layer.name}",sep='      ',end="\r")
        layer.set_weights(weights)
    model.trainable = False
    return


def get_decoder_weights(decoder, head):
    layers = [[]]
    block = []
    weights = []

    # get decoder weights and group them together
    for layer in decoder:
        if layer._type == "route":
            layers.append(block)
            block = []
        elif layer._type == "convolutional":
            block.append(layer)
        else:
            layers.append([])
    layers.append(block)

    # interleve weights for blocked layers
    for layer in layers:
        weights.append(interleve_weights(layer))

    # get weights for output detection heads
    for layer in reversed(head):
        if layer != None and layer._type == "convolutional":
            weights.append(layer.get_weights())

    return layers, weights
