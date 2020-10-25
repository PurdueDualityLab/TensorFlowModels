head = {
    5: {
        "depth": 512,
        "upsample": None,
        "upsample_conditions": None,
        "processor": 'yolo>DarkRouteProcess',
        "processor_conditions": {
            "filters": 512,
            "repetitions": 2,
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
        "upsample": 'yolo>DarkUpsampleRoute',
        "upsample_conditions": {
            "filters": 128
        },
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
}
