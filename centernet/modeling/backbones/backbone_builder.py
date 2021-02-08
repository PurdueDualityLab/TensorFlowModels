from centernet.modeling.backbones.centernet_backbone import CenterNetBackbone
import centernet.modeling.backbones.config

def buildCenterNetBackbone(config):
    downsample = False
    n_stacks = 0
    all_filter_sizes = []
    all_rep_sizes = []
    all_strides = []
    
    for layer in config:
        name, filterSizes, repSizes, strides = layer

        if name == "Downsample":
            downsample = True
        elif name == "HourglassBlock":
            n_stacks += 1
            if len(filterSizes) != len(repSizes):
                print("Number of filter sizes and rep sizes must be equal")
                break
            all_filter_sizes.append(filterSizes)
            all_rep_sizes.append(repSizes)
            all_strides.append(strides)
        else:
            print("Invalid layer name provided")
            break
    
    backbone = CenterNetBackbone(filter_sizes=all_filter_sizes,
                                 rep_sizes=all_rep_sizes,
                                 n_stacks=n_stacks,
                                 strides=all_strides,
                                 downsample=downsample)
    
    return backbone