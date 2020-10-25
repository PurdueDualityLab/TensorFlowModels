head = {
    3: {
        "depth": 128,
        "resample": None,
        "resample_conditions": None,
        "processor": 'yolo>DarkRouteProcess',
        "processor_conditions": {
            "filters": 256,
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
    4: {
        "depth": 256,
        "resample": 'yolo>DarkRoute',
        "resample_conditions": {
            "filters": 256,
            "downsample": True
        },
        "processor": 'yolo>DarkRouteProcess',
        "processor_conditions": {
            "filters": 512,
            "repetitions": 6,
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
    5: {
        "depth": 512,
        "resample": 'yolo>DarkRoute',
        "resample_conditions": {
            "filters": 512,
            "downsample": True
        },
        "processor": 'yolo>DarkRouteProcess',
        "processor_conditions": {
            "filters": 1024,
            "repetitions": 6,
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
}
