# Preprocessing

## Find Mean and Std of dataset

Useful for training new models. For pre-trained models, should use given normalization statistics

### Usage

```bash
python find_mean_std_multiprocess.py \
    --csv /path/to/test.csv \
    --csv_path_col_name image_path \
    --img_size 512 \
    --num_processes 16
```