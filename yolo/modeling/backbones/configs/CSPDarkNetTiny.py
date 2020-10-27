# (name, stack, numberinblock, bottleneck, filters, kernal_size, strides, padding,  activation, route, output_name)
BACKBONE = [
    ["DarkConv", None, 1, False, 32, 3, 2, "same", "leaky", -1, 0, False],  # 1
    ["DarkConv", None, 1, False, 64, 3, 2, "same", "leaky", -1, 1, False],  # 1
    ["CSPTiny", "csp_tiny", 1, False, 64, 3, 2, "same", "leaky", -1, 2,
     False],  # 3
    [
        "CSPTiny", "csp_tiny", 1, False, 128, 3, 2, "same", "leaky", -1, 3,
        False
    ],  # 3
    ["CSPTiny", "csp_tiny", 1, False, 256, 3, 2, "same", "leaky", -1, 4,
     True],  # 3
    ["DarkConv", None, 1, False, 512, 3, 1, "same", "leaky", -1, 5, True],  # 1
]  # 52 layers

backbone = {"splits": {"backbone_split": 28}, "backbone": BACKBONE}
