## Purdue YOLO Demo

_December 2020_


## Summary

Welcome to the demo of Purdue University’s first contribution to the TensorFlow Model Garden. This demo showcases our implementation of the You Only Look Once (YOLO) model, written in TensorFlow v2.0 and deployed on a mobile phone. The demo will run on Android or iOS.

In order to build the Demo, we first took the existing Tensorflow Lite Demo Apps and re-purposed them to work with both Yolov4-tiny, and Yolov3-regular. We used models trained COCO.

You can watch a video of the demo here: [https://youtu.be/sl8iN6eHCfo](https://youtu.be/sl8iN6eHCfo)

You can find a precompiled APK for the Android version of this app on [http://cam2project.net/yolo](https://www.cam2project.net/yolo)


## Technical Details


### Implementation

Our YOLO implementation follows the TensorFlow Model Garden guidelines. It is 7,705 lines of Python code (SLOC) excluding test cases. The implementation includes:



*   The TensorFlow 2.x YOLO v3 and v4 models
*   Classification and Detection Data Pipelines
*   Loss Functions
*   Utilities:
    *   IOU
    *   Custom Kmeans Anchor Box Selection
    *   Custom Single Stage Non-Max Suppression box filtering (for TF Lite)
*   Demo Files
*   Tutorials
*   Training Loops

We built the Android and iOS mobile app on top of Google’s previous app [1], which showcased object detection in TensorFlow Lite using a pre-trained MobileNet SSD model from the Object Detection API. We more-or-less preserved the original application, but made the specific image processing model into a dynamically configurable property. We also added support to flip the camera. Now the app can also perform object detection with YOLO, using front- or rear-facing cameras.

In the demo, you can select either YOLO v3 [2] or YOLO v4 Tiny [3]. YOLO v4 Tiny is a newer, smaller (“tiny”) model than v3. It uses Cross Stage Partial Networks to improve accuracy and derive deeper features using connections across different blocks in the model.


### Performance

On a desktop computer:



*   Our YOLO v3 model can process 120 frames per second.
*   The original C implementation of the model (DarkNet) can process 140 frames per second on the same machine.

On our mobile device (iPhone 11):



*   Our YOLO v4 Tiny implementation can process 15 frames per second using 2 threads.
*   Our YOLO v3 Regular implementation can process 1 frame per second using 2 threads.

The YOLO family of models is designed to be energy efficient. However, we have not yet assessed the energy costs of our implementation.


## Can I try the demo on my own device?

Certainly! We have documented the installation instructions and app walkthrough for:



*   [Android](android/)
*   [iOS](ios/)

We have tested the demo on:



*   Android:
    *   Oneplus 7T (Android 10)
    *   Samsung Galaxy S10 (Android 10)
    *   (Android 11)
*   iOS:
    *   iPhone 11 Pro Max (iOS 13)
    *   iPhone 10s Max (iOS 14.2)


## Contributors



*   **Google**: Jaeyoun Kim, Abdullah Rashwan
*   **Purdue University**: Vishnu Banna, Anirudh Vegesana, Akhil Chinnakotla, Tristan Yan, Naveen Vivek, James Davis, Yung-Hsiang Lu
*   **Loyola University Chicago**: George K. Thiruvathukal


## References

[1] TensorFlow Lite Object Detection Demo: [https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection)

[2] YOLO v3: _Joseph Redmon and Ali Farhadi. Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767, 2018._ ([https://arxiv.org/pdf/2004.10934.pdf](https://arxiv.org/pdf/2004.10934.pdf))

[3] YOLO v4 tiny: _Alexey Bochkovskiy, Chien-Yao Wang, and HongYuan Mark Liao. Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934, 2020_ ([https://arxiv.org/pdf/1804.02767.pdf](https://arxiv.org/pdf/1804.02767.pdf))


## Acknowledgments

We thank Google for their generous financial and technical support.
