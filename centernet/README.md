> :memo: A README.md template for releasing a paper code implementation to a GitHub repository.
>
> * Template version: 1.0.2020.170
> * Please modify sections depending on needs.

# CenterNet

[![Paper](http://img.shields.io/badge/Paper-arXiv.1904.07850-B3181B?logo=arXiv)](https://arxiv.org/abs/1904.07850)

This repository is the unofficial implementation of the following paper.

* Paper title: [Objects as Points](https://arxiv.org/abs/1904.07850)

## Description

CenterNet [1] builds upon CornerNet [2], an anchor-free model for object
detection.

Many other models, such as YOLO and RetinaNet, use anchor boxes. These anchor
boxes are predefined to be close to the aspect ratios and scales of the objects
in the training dataset. Anchor-based models do not predict the bounding boxes
of objects directly. They instead predict the location and size/shape
refinements to a predefined anchor box. The detection generator then computes
the final confidences, positions, and size of the detection.

CornerNet eliminates the need for anchor boxes. RetinaNet needs thousands of
anchor boxes in order to cover the most common ground truth boxes [2]. This adds
unnecessary complexity to the model which slow down training and create
imbalances in positive and negative anchor boxes [2]. Instead, CornerNet creates
heatmaps for each of the corners and pools them together in order to get the
final detection boxes for the objects. CenterNet removes even more complexity
by using the center instead of the corners, meaning that only one set of
heatmaps (one heatmap for each class) is needed to predict the object. CenterNet
proves that this can be done without a significant difference in accuracy.

## History

> :memo: Provide a changelog.

## Authors or Maintainers

> :memo: Provide maintainer information.

* Full name ([@GitHub username](https://github.com/username))
* Full name ([@GitHub username](https://github.com/username))

## Table of Contents

> :memo: Provide a table of contents to help readers navigate a lengthy README document.

## Requirements

[![TensorFlow 2.4](https://img.shields.io/badge/TensorFlow-2.4-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.4.0)
[![Python 3.7](https://img.shields.io/badge/Python-3.7-3776AB)](https://www.python.org/downloads/release/python-370/)

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

[1] CenterNet: Xingyi Zhou, Dequan Wang, and Philipp Krähenbühl. Objects as Points. arXiv preprint arXiv:1904.07850, 2019. (https://arxiv.org/abs/1904.07850)
[2] CornerNet: Hei Law and Jia Deng. CornerNet: Detecting Objects as Paired Keypoints. arXiv preprint arXiv:1808.01244, 2018. (https://arxiv.org/abs/1808.01244)

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
