
from yolo.utils.scripts.darknet2tf.get_weights import get_darknet53_tf_format, interleve_weights

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
