# (name, stack, numberinblock, bottleneck, filters, kernal_size, strides, padding,  activation, route, output_name)
BACKBONE = [
    ["DarkConv", None, 1, False, 32, 3, 1, "same", "mish", -1, 0, False],  # 1
    ["DarkRes", "csp", 1, True, 64, None, None, None, "mish", -1, 1,
     False],  # 3
    ["DarkRes", "csp", 2, False, 128, None, None, None, "mish", -1, 2,
     False],  # 2
    # 14 route 61 last block
    ["DarkRes", "csp", 8, False, 256, None, None, None, "mish", -1, 3, True],
    # 14 route 61 last block
    ["DarkRes", "csp", 8, False, 512, None, None, None, "mish", -1, 4,
     True],  # 3
    ["DarkRes", "csp", 4, False, 1024, None, None, None, "mish", -1, 5,
     True],  # 6  #route
]  # 52 layers

backbone = {
    "splits": {
        "backbone_split": 106,
        "neck_split": 138
    },
    "backbone": BACKBONE,
}
