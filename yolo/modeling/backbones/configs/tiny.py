# (name, numberinblock, filters, kernal_size, padding, strides, downsample, output, output_name)
backbone = [
    ("DarkConv", 1, 16, 3, 1, "same", False, False, None),
    ("darkyolotiny", 1, 32, 3, 2, "same", False, False, None),
    ("darkyolotiny", 1, 64, 3, 2, "same", False, False, None),
    ("darkyolotiny", 1, 128, 3, 2, "same", False, False, None),
    ("darkyolotiny", 1, 256, 3, 2, "same", False, True, 3),
    ("darkyolotiny", 1, 512, 3, 2, "same", False, False, None),
    ("darkyolotiny", 1, 1024, 3, 1, "same", False, True, 5),
    ###########
]
