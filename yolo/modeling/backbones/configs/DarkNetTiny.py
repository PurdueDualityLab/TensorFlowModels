# (name, stack, numberinblock, bottleneck, filters, kernal_size, strides, padding,  activation, route, output_name)
BACKBONE = [
    ["DarkConv", None, 1, False, 16, 3, 1, "same", "leaky", -1, 0, False],  # 1
    ["DarkTiny", None, 1, True, 32, 3, 2, "same", "leaky", -1, 1, False],  # 3
    ["DarkTiny", None, 1, True, 64, 3, 2, "same", "leaky", -1, 2, False],  # 3
    ["DarkTiny", None, 1, False, 128, 3, 2, "same", "leaky", -1, 3,
     False],  # 2
    # 14 route 61 last block
    ["DarkTiny", None, 1, False, 256, 3, 2, "same", "leaky", -1, 4,
     True],  # wrong output 
    # 14 route 61 last block
    ["DarkTiny", None, 1, False, 512, 3, 2, "same", "leaky", -1, 5,
     False],  # 3
    ["DarkTiny", None, 1, False, 1024, 3, 1, "same", "leaky", -1, 5,
     True],  # 6  #route
]  # 52 layers

backbone = {
    "splits": {
        "backbone_split": 14
    },
    "backbone": BACKBONE,
}
