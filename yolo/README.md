# YOLO -> You Only LOOK Once

## Model Purpose 
The yolo models were introduced in 2015 as a show case for a fast Neural Network whose main goal is to identify objects and their locations in an image using a single forward pass or a single conputation. Over the years, the YOLO team has identified model bottle necks and updated the model to get better perfomance in more visual scenarios. 

## Our Goal
With this library, our goal as an independent team is to provide the research community with Tensorflow native implmentations that have been trained and benchmarked to ensure equivalent performance to the darknet orignal implementation, while allowing for the versitility of using a more approchable Deep learning Library, specifically Tensorflow 2.x. We also hope to provide Documentation and explanations of the networks functionality to allow the YOLO model to feel less like a black box. 

## What you will find in this repo

| Object Detectors | Classifiers      |
| ---------------- | ---------------- |
| Yolo-v3          | Darknet53        |
| Yolo-v3 tiny     | CSPDarknet53     |
| Yolo-v3 spp      |
| Yolo-v4          |
| Yolo-v4 tiny     |

In addition to the native implementations, we are providing a Darknet Native Config to Tensorflow converter. This tool will take a custom config from Darknet (for any convolutional Model), and re-constuct the model in Tensorflow 2.x. If a weights file Is provided, the tool will also load the Darknet weights into the constructed tensorflow model directly. This was done to allow better future proofing with minor alterations, and to provide simple Backwards compatibility for older models like Yolo v1 and Yolo v2. 

**We will try to provide loss functions for models built from Config files, but if we are not able to find one, or have not yet implemented it, you will find the warning:

``` WARNING: Model from Config, loss function not found, for custom training, please construct, or find the loss function for this model, if the model is used as an industry standard, please post an issure request, and we will try to implement the loss function as soon as possible.``` 

## Install Instructions

```not yet defined```

## Usage Instructions

```user interfaceing not yet implemented```

## Custom Training Instructions

```user interfaceing not yet implemented```

## Tests and Benchmark Statistics

```user interfaceing not yet implemented```

## Community guidelines

```not yet implemented```

## Citations

```will get to it```