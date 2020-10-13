# (name, stack, numberinblock, bottleneck, filters, kernal_size, strides, padding,  activation, route, output_name)
backbone = [
    ("DarkConv", None, 1, False, 32, 3, 2, "same", "leaky", -1, None),  # 1
    ("DarkConv", None, 1, False, 64, 3, 2, "same", "leaky", -1, None),  # 1
    ("CSPTiny", "csp_tiny", 1, False, 64, 3, 2, "same", "leaky", -1, None),  # 3
    ("CSPTiny", "csp_tiny", 1, False, 128, 3, 2, "same", "leaky", -1, None),  # 3
    ("CSPTiny", "csp_tiny", 1, False, 256, 3, 2, "same", "leaky", -1, 4),  # 3
    ("DarkConv", None, 1, False, 512, 3, 1, "same", "leaky", -1, 5),  # 1
]  # 52 layers
