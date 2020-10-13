# (name, stack, numberinblock, bottleneck, filters, kernal_size, strides, padding,  activation, route, output_name)
backbone = [
    ("DarkConv", None, 1, False, 16, 3, 1, "same", "leaky", -1, None),  # 1
    ("darkyolotiny", None, 1, True, 32, 3, 2, "same", "leaky", -1, None),  # 3
    ("darkyolotiny", None, 1, True, 64, 3, 2, "same", "leaky", -1, None),  # 3
    ("darkyolotiny", None, 1, False, 128, 3, 2, "same", "leaky", -1, None),  # 2
    # 14 route 61 last block
    ("darkyolotiny", None, 1, False, 256, 3, 2, "same", "leaky", -1, 3),
    # 14 route 61 last block
    ("darkyolotiny", None, 1, False, 512, 3, 2, "same", "leaky", -1, 4),  # 3
    ("darkyolotiny", None, 1, False, 1024, 3, 2, "same", "leaky", -1, 5),  # 6  #route
]  # 52 layers
