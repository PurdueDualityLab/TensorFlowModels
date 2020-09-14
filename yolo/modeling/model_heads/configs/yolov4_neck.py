
head = {
"1024":{
      "depth":1024,
      "upsample":None,
      "upsample_conditions":None,
      "processor": 'yolo>DarkRouteProcess',
      "processor_conditions":{"filters": 1024, "repetitions": 3, "insert_spp": True},
      "tailing_conditions":"standard",
   },
"512":{
      "depth":512,
      "upsample": 'yolo>DarkRoute',
      "upsample_conditions":{"filters": 256},
      "processor": 'yolo>DarkRouteProcess',
      "processor_conditions":{"filters": 512, "repetitions": 2, "insert_spp": False},
      "tailing_conditions":"standard",
   },
"256":{
      "depth":256,
      "upsample": 'yolo>DarkRoute',
      "upsample_conditions":{"filters": 128},
      "processor": 'yolo>DarkRouteProcess',
      "processor_conditions":{"filters": 256, "repetitions": 2, "insert_spp": False},
      "tailing_conditions":"half_standard",
   },
}
