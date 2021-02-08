# (name, filterSizes, repSizes, strides)
CENTERNET_HG104_CFG = [
    ("Downsample", None, None, None),
    ("HourglassBlock", [256, 256, 384, 384, 384, 512], [2, 2, 2, 2, 2, 4], 1),
    ("HourglassBlock", [256, 256, 384, 384, 384, 512], [2, 2, 2, 2, 2, 4], 1),
]