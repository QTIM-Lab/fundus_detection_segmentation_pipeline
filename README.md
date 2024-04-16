# Detection and Segmentation Pipeline

By Scott Kinder
scott.kinder@cuanschutz.edu

# Overview

This repo is broken down into 2 main parts:

1. [detection](./detection/) Detection
2. [segmentation](./segmentation/) Segmentation

# Usage

It is sequential, so Part 1 creates the dataset for Part 2 to use

Both modules have instructions to follow from start to finish

To begin, start with [detection](./detection/) and get a YOLO model to create a cropped dataset which you can then train or evaluate on with your [segmentation](./segmentation/) model in step 2

# Prior Assumptions

## Data

We assume you have a set of full fundus images and segmentation labels. In our case, we had full fundus photos and segmentations of cup and discs from Drishti-GS, RIGA, Refuge-1, and RIM-ONE DL datasets. We preprocessed our segmentation labels into .png images with 3 channels, which represented the background, cup, and disc (as implicitly RGB for visualization) respectively.

## Python environment

There are requirements.txt files [here](./requirements.txt). This repo really just uses a couple standard Deep Learning libaries

- Pytorch
- OpenCV (cv2), PIL
- numpy, pandas, matplotlib
- ultralytics (for YOLO model)
- albumentations (for image augmentations)

I highly doubt that in the event you get a missing package that just pip installing it wouldn't work, but feel free to contact me if you think something is missing otherwise I think that's mostly it, and the requirements.txt files
