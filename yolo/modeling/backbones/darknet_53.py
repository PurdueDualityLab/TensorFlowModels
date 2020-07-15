import yolo.modeling.building_blocks as nn_blocks



# (name, numberinblock, filters, kernal_size, downsample, route, output)
darknet53 = {
    ("DarkConv", 1, 32, 3, False, False, False), #1
    ("DarkRes", 1, 64, None, True, False, False), #3
    ("DarkRes", 1, 128, None, True, False, False), #3
    ("DarkRes", 1, 128, None, False, False, False), #2
    ("DarkRes", 1, 256, None, True, False, False), #3
    ("DarkRes", 7, 256, None, False, False, False), #14 route 61 last shortcut or last conv
    ("DarkRes", 1, 512, None, True, False, False), #3
    ("DarkRes", 7, 512, None, False, False, False), #14 route 61 last shortcut or last conv
    ("DarkRes", 1, 1024, None, True, False, False), #3
    ("DarkRes", 3, 1024, None, False, False, True), #6  #route  
} #52 layers 
