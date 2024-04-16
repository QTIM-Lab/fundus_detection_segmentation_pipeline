# Evaluate Jaccard and Dice Metrics

## Usage

```bash

# Get Jaccard and Dice with Multiprocessing

# For Cup
python evaluate_metrics_mp.py \
    --prediction_folder /path/to/predictions \
    --label_folder /path/to/labels \
    --csv_path /path/to/data.csv \
    --output_folder /path/to/output \
    --disclude_datasets rimone \
    --num_processes 4

# For Disc
python evaluate_metrics_mp.py \
    --prediction_folder /path/to/predictions \
    --label_folder /path/to/labels \
    --csv_path /path/to/data.csv \
    --output_folder /path/to/output \
    --disclude_datasets rimone \
    --eval_disc \
    --num_processes 4

```