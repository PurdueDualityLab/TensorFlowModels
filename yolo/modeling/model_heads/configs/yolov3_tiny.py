head = {
    5: {
        "depth": 1024,
        "upsample": None,
        "upsample_conditions": None,
        "processor": 'yolo>DarkRouteProcess',
        "processor_conditions": {
            "filters": 512,
            "repetitions": 1,
            "insert_spp": False
        },
        "output_conditions": {
            "kernel_size": (1, 1),
            "strides": (1, 1),
            "padding": "same",
            "use_bn": False,
            "activation": None
        },
        "output-extras": 0,
    },
    3: {
        "depth": 256,
        "upsample": 'yolo>DarkUpsampleRoute',
        "upsample_conditions": {
            "filters": 128
        },
        "processor": 'yolo>DarkConv',
        "processor_conditions": {
            "filters": 256,
            "use_bn": True,
            "kernel_size": (3, 3),
            "strides": (1, 1),
            "padding": 'same',
        },
        "output_conditions": {
            "kernel_size": (1, 1),
            "strides": (1, 1),
            "padding": "same",
            "use_bn": False,
            "activation": None
        },
        "output-extras": 0,
    },
}
