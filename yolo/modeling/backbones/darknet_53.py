# (name, numberinblock, filters, kernal_size, downsample, output)
darknet53_config = [
    ("DarkConv", 1, 32, 3, False, False),  # 1
    ("DarkRes", 1, 64, None, True, False),  # 3
    ("DarkRes", 1, 128, None, True, False),  # 3
    ("DarkRes", 1, 128, None, False, False),  # 2
    ("DarkRes", 1, 256, None, True, False),  # 3
    # 14 route 61 last shortcut or last conv
    ("DarkRes", 7, 256, None, False, True),
    ("DarkRes", 1, 512, None, True, False),  # 3
    # 14 route 61 last shortcut or last conv
    ("DarkRes", 7, 512, None, False, True),
    ("DarkRes", 1, 1024, None, True, False),  # 3
    ("DarkRes", 3, 1024, None, False, True),  # 6  #route
]  # 52 layers
