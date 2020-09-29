head = {
    "1024": {
        "depth": 1024,
        "upsample": None,
        "upsample_conditions": None,
        "processor": 'yolo>DarkRouteProcess',
        "processor_conditions": {
            "filters": 1024,
            "repetitions": 3,
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
    "512": {
        "depth": 512,
        "upsample": 'yolo>DarkUpsampleRoute',
        "upsample_conditions": {
            "filters": 256
        },
        "processor": 'yolo>DarkRouteProcess',
        "processor_conditions": {
            "filters": 512,
            "repetitions": 3,
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
    "256": {
        "depth": 256,
        "upsample": 'yolo>DarkUpsampleRoute',
        "upsample_conditions": {
            "filters": 128
        },
        "processor": 'yolo>DarkRouteProcess',
        "processor_conditions": {
            "filters": 256,
            "repetitions": 3,
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
