# Detection and Segmentation Pipeline

By Scott Kinder
scott.kinder@cuanschutz.edu

# Architecture

## Inference pipeline architecture

It is comprised of 2 main steps, detection and segmentation:

![End-to-end](./docs/img/end_to_end_pipeline.png)

## End-to-end segmentation recovery process

Once the segmentation has been acquired, it must be recovered back onto the original process via the following steps. This will undo the steps in the inference pipeline to get the segmentation back onto the original image:


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

![End-to-end](./docs/img/pub_data_map.png)

We assume you have a set of full fundus images and segmentation labels. In our case, we had full fundus photos and segmentations of cup and discs from 9 public datasets.

- [Chákṣu](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9898274/)
- [Drishti-GS](https://ieeexplore.ieee.org/document/6867807)
- [G1020](https://arxiv.org/abs/2006.09158)
- [ORIGA](https://pubmed.ncbi.nlm.nih.gov/21095735/)
- [REFUGE-1](https://refuge.grand-challenge.org/)
- [RIM-ONE-DL](https://www.ias-iss.org/ojs/IAS/article/view/2346)
- [RIGA](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10579/2293584/Retinal-fundus-images-for-glaucoma-analysis-the-RIGA-dataset/10.1117/12.2293584.full#_=_) (Composed of Bin Rushed, Magrabi, and Messidor)

## Python environment

There are requirements.txt files [here](./requirements.txt). This repo really just uses a couple standard Deep Learning libaries

- Pytorch
- OpenCV (cv2), PIL
- numpy, pandas, matplotlib
- ultralytics (for YOLO model)
- albumentations (for image augmentations)

I highly doubt that in the event you get a missing package that just pip installing it wouldn't work, but feel free to contact me if you think something is missing otherwise I think that's mostly it, and the requirements.txt files
