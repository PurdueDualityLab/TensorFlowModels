# Yolo Object Detectors, You Only Look Once

[![Paper](http://img.shields.io/badge/Paper-arXiv.1804.02767-B3181B?logo=arXiv)](https://arxiv.org/abs/1804.02767)

This repository is the unofficial implementation of the following paper. However, we spent painstaking hours ensuring that every aspect that we constructed was the exact same as the original paper and the original repository. 

* YOLOv3: An Incremental Improvement: [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)

## Description

> :memo: Provide description of the model.  
>  
> * Provide brief information of the algorithms used.  
> * Provide links for demos, blog posts, etc.  

Yolo v1 the original implementation was released in 2015 providing a ground breaking algorithem that would quickly process images, and locate objects in a single pass through the detector. The orignal implementation based used a backbone derived from state of the art object classifier of the time, like [GoogLeNet](https://arxiv.org/abs/1409.4842) and [VGG](https://arxiv.org/abs/1409.1556). More Attention was given to the novel Yolo Detection head that allowed for classification with a single pass of an image. Though limited, the network could predict up to 90 bounding boxes per image, and was tested for about 80 classes per box. Also, the model could only make prediction at one scale. These attributes caused yolo v1 to be more limited, and less verisitle, so as the year passed, the Developers continued to update and develop this model. 

Yolo v3 and v4 serve as the most up to date and capable versions of the Yolo network group. These model uses a custom backbone called Darknet53 that uses knowledge gained from the ResNet paper to improve its predictions. The new backbone also allows for objects to be detected at multiple scales. As for the new detection head, the model now predicts the bounding boxes using a set of anchor box priors (Anchor Boxes) as suggestions. The multiscale predictions in combination with the Anchor boxes allows for the network to make up to 1000 object predictions on a single image. Finally, the new loss function forces the network to make better prediction by using Intersection Over Union (IOU) to inform the models confidence rather than relying on the mean squared error for the entire output. 


## History

> :memo: Provide a changelog.

## Authors or Maintainers

> :memo: Provide maintainer information.  

* Vishnu Samardh Banna ([@GitHub vishnubanna](https://github.com/vishnubanna))
* Full name ([@GitHub username](https://github.com/username))

## Models in the library

| Object Detectors | Classifiers      |
| :--------------: | :--------------: |
| Yolo-v3          | Darknet53        |
| Yolo-v3 tiny     | CSPDarknet53     |
| Yolo-v3 spp      |
| Yolo-v4          |
| Yolo-v4 tiny     |

* For all Standard implementations, we provided scripts to load the weights into the Tensorflow implementation provided that you have a yolo**.cfg file, and the corresponding yolo**.weights file.

## Table of Contents

> :memo: Provide a table of contents to help readers navigate a lengthy README document.

## Requirements

[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB)](https://www.python.org/downloads/release/python-380/)

> :memo: Provide details of the software required.  
>  
> * Add a `requirements.txt` file to the root directory for installing the necessary dependencies.  
>   * Describe how to install requirements using pip.  
> * Alternatively, create INSTALL.md.  

To install requirements:

```setup
pip install -r requirements.txt
```

## Results

[![TensorFlow Hub](https://img.shields.io/badge/TF%20Hub-Models-FF6F00?logo=tensorflow)](https://tfhub.dev/...)

> :memo: Provide a table with results. (e.g., accuracy, latency)  
>  
> * Provide links to the pre-trained models (checkpoint, SavedModel files).  
>   * Publish TensorFlow SavedModel files on TensorFlow Hub (tfhub.dev) if possible.  
> * Add links to [TensorBoard.dev](https://tensorboard.dev/) for visualizing metrics.  
>  
> An example table for image classification results  
>  
> ### Image Classification  
>  
> | Model name | Download | Top 1 Accuracy | Top 5 Accuracy |  
> |------------|----------|----------------|----------------|  
> | Model name | [Checkpoint](https://drive.google.com/...), [SavedModel](https://tfhub.dev/...) | xx% | xx% |  

## Dataset

> :memo: Provide information of the dataset used.  

## Training

> :memo: Provide training information.  
>  
> * Provide details for preprocessing, hyperparameters, random seeds, and environment.  
> * Provide a command line example for training.  

Please run this command line for training.

```shell
python3 ...
```

## Evaluation

> :memo: Provide an evaluation script with details of how to reproduce results.  
>  
> * Describe data preprocessing / postprocessing steps.  
> * Provide a command line example for evaluation.  

Please run this command line for evaluation.

```shell
python3 ...
```

## References

> :memo: Provide links to references.  

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> :memo: Place your license text in a file named LICENSE in the root of the repository.  
>  
> * Include information about your license.  
> * Reference: [Adding a license to a repository](https://help.github.com/en/github/building-a-strong-community/adding-a-license-to-a-repository)  

This project is licensed under the terms of the **Apache License 2.0**.

## Citation

> :memo: Make your repository citable.  
>  
> * Reference: [Making Your Code Citable](https://guides.github.com/activities/citable-code/)  

If you want to cite this repository in your research paper, please use the following information.
