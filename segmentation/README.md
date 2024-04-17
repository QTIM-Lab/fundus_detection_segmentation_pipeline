# Segmentation

This module shows how to do segmentation training and inference to collect Cup-Disc Ratios (CDR) using [Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former).

## Pretrained Mask2Former segmentation model

To use our model, the model weights can be found in a dropbox here:

[Link to model weights on dropbox](https://www.dropbox.com/scl/fi/otpvalopjfrzmqhahztfj/model.pt?rlkey=gmdtmp4jedmyxepvw1n7q38sc&dl=0)

You can now specify those weights to run inference (see [step 3](#3-run-inference-and-collect-cdrs) and the [infer](./scripts/infer/) module)

If asked to specify dataset mean and standard deviation, use the following values:

mean: 0.768, 0.476, 0.290

std: 0.220, 0.198, 0.166

# Overview

The full module is broken down into 4 main parts, supported by 4 .py scripts:

1. Collect the mean and standard deviation of dataset - [find_mean_std_multiprocess.py](./scripts/preprocess/find_mean_std_multiprocess.py)
2. Train the MaskFormer model - [train_mask2former.py](./scripts/train/train_mask2former.py)
3. Run inference on a .csv of image paths to collect CDRs and visualize results - [infer_segment_cdr.py](./scripts/infer/infer_segment_cdr.py)
4. Evaluate predictions to get Jaccard and Dice score - [evaluate_metrics_mp.py](./scripts/evaluate/evaluate_metrics_mp.py)

The docs go in this order

# 1. Collect Mean and Std of Dataset

The first step that you should do is collect the mean and standard deviation of your dataset.

Typically normalization is used for this model and therefore we need the mean and std for the data

To get the statistics, run the [find_mean_std_multiprocess.py](./scripts/preprocess/find_mean_std_multiprocess.py) script:

```bash
python segmentation/scripts/preprocess/find_mean_std_multiprocess.py \
    --csv /path/to/data.csv \
    --csv_path_col_name image_path \
    --img_size 512 \
    --num_processes 16
```

That will print to the console the results, i.e.

Number of images: 836

Mean: [0.76766771 0.47570729 0.2902386 ]

Standard Deviation: [0.2180807  0.19569278 0.16597114]

# 2. Train the MaskFormer Model

Now that you have the Mean and Std of your data, you are ready to train the model

To train, use the [train_mask2former.py](./scripts/train/train_mask2former.py) script:

```bash
python segmentation/scripts/train/train_mask2former.py \
    --train_csv /path/to/train.csv \
    --val_csv /path/to/val.csv \
    --csv_img_path_col image_path \
    --csv_label_path_col label_path \
    --output_directory /path/to/outputs \
    --dataset_mean 0.768 0.476 0.290 \
    --dataset_std 0.220 0.198 0.166 \
    --lr 0.00002 \
    --batch_size 8 \
    --jitters 0.5 0.5 0.25 0.1 0.75 \
    --num_epochs 200 \
    --patience 10 \
    --num_val_outputs_to_save 7 \
    --num_workers 16
```

It expects:
- input_csv to be a csv with all training data in rows with paths to images and segmentation labels, and provide the column names
- you to input your datasets mean and std
- specify other hyperparams

# 3. Run Inference and Collect CDRs

Once the model is finished training, and you have the model weights in the output directory ready to go, you can run inference quickly with multiprocessing to collect CDRs and visualize the outputs

To run inference, collect a result .csv with CDRs, and outputs of inference segmentations, use the [infer_segment_cdr.py](./scripts/infer/infer_segment_cdr.py) script:

```bash
python segmentation/scripts/infer/infer_segment_cdr.py \
    --model_path /path/to/segmentation/model.pt \
    --input_root_dir /path/to/images \
    --input_csv /path/to/data.csv \
    --csv_path_col_name image_path \
    --output_root_dir /path/to/output \
    --batch_size 8 \
    --dataset_mean 0.768 0.476 0.290 \
    --dataset_std 0.220 0.198 0.166 \
    --num_processes 16 \
    --cuda_num 0
```

It expects:
- input_csv to be a csv with all image files that you want to run inference on in rows with file names, and to provide the column name for that

# 4. Evaluate Jaccard and Dice

To evaluate quantitative segmentation metrics, use the scripts below

I recommend using low number of processes, even if CPU resources are available, as using large files (i.e. original full fundus images) can consume a lot of RAM on each thread and can crash the program

```bash
# For Cup
python segmentation/scripts/evaluate/evaluate_metrics_mp.py \
    --prediction_folder /path/to/predictions \
    --label_folder /path/to/labels \
    --csv_path /path/to/data.csv \
    --output_folder /path/to/output \
    --disclude_datasets rimone \
    --num_processes 4

# For Disc
python segmentation/scripts/evaluate/evaluate_metrics_mp.py \
    --prediction_folder /path/to/predictions \
    --label_folder /path/to/labels \
    --csv_path /path/to/data.csv \
    --output_folder /path/to/output \
    --disclude_datasets rimone \
    --eval_disc \
    --num_processes 4
```