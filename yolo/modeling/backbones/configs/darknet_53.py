# (name, numberinblock, filters, kernal_size, padding, strides, downsample, output)
backbone = [
    ("DarkConv", 1, 32, 3, 1, "same", False, False),  # 1
    ("DarkRes", 1, 64, None, None, None, True, False),  # 3
    ("DarkRes", 1, 128, None, None, None, True, False),  # 3
    ("DarkRes", 1, 128, None, None, None, False, False),  # 2
    ("DarkRes", 1, 256, None, None, None, True, False),  # 3
    # 14 route 61 last block
    ("DarkRes", 7, 256, None, None, None, False, True),
    ("DarkRes", 1, 512, None, None, None, True, False),  # 3
    # 14 route 61 last block
    ("DarkRes", 7, 512, None, None, None, False, True),
    ("DarkRes", 1, 1024, None, None, None, True, False),  # 3
    ("DarkRes", 3, 1024, None, None, None, False, True),  # 6  #route
]  # 52 layers
