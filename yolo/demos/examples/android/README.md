# TensorFlow Lite Object Detection Android Demo

### Overview

This is a camera app that continuously detects the objects (bounding boxes and
classes) in the frames seen by your device's back camera, using a quantized
[YOLOv3 regular and YOLOv4 tiny](../../..)
model trained on the [COCO dataset](http://cocodataset.org/). These instructions
walk you through building and running the demo on an Android device.

The model files are downloaded via Gradle scripts when you build and run. You
don't need to do any steps to download TFLite models into the project
explicitly.

Application can run either on device or emulator.

## Installing the demo from a precompiled APK

See [Installation](INSTALL.md).

## Build the demo using Android Studio

### Prerequisites

*   If you don't have already, install
    **[Android Studio](https://developer.android.com/studio/index.html)**,
    following the instructions on the website.

*   You need an Android device and Android development environment with minimum
    API 21.

*   Android Studio 3.2 or later.

### Building

*   Open Android Studio, and from the Welcome screen, select Open an existing
    Android Studio project.

*   From the Open File or Project window that appears, navigate to and select
    the tensorflow-lite/examples/object_detection/android directory from
    wherever you cloned the TensorFlow Lite sample GitHub repo. Click OK.

*   If it asks you to do a Gradle Sync, click OK.

*   You may also need to install various platforms and tools, if you get errors
    like "Failed to find target with hash string 'android-21'" and similar.
    Click the `Run` button (the green arrow) or select `Run > Run 'android'`
    from the top menu. You may need to rebuild the project using `Build >
    Rebuild` Project.

*   If it asks you to use Instant Run, click Proceed Without Instant Run.

*   Also, you need to have an Android device plugged in with developer options
    enabled at this point. See
    **[here](https://developer.android.com/studio/run/device)** for more details
    on setting up developer devices.

#### Switch between inference solutions (Task library vs TFLite Interpreter)

This object detection Android reference app demonstrates two implementation
solutions,
[lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/nl_classification/android/lib_task_api)
that leverages the out-of-box API from the
[TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/object_detector),
and
[lib_interpreter](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android/lib_interpreter)
that creates the custom inference pipleline using the
[TensorFlow Lite Interpreter Java API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_java).
You can change the build variant to whichever one you want to build and run—just
go to `Build > Select Build Variant` and select one from the drop-down menu. See
[configure product flavors in Android Studio](https://developer.android.com/studio/build/build-variants#product-flavors)
for more details.

### Custom model used

This example shows you how to perform TensorFlow Lite object detection using a
custom model.

* Checkout the branch 'custom_nms'
* Run the command `python3 -m yolo.utils.export.tflite_convert`
* The models will be created with the dataset label names packed as metadata

### Additional Note

_Please do not delete the assets folder content_. If you explicitly deleted the
files, then please choose *Build*->*Rebuild* from menu to re-download the
deleted model files into assets folder.
