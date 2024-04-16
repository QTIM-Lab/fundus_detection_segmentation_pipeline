import argparse
import os

from segmentation_train_and_inference.utils.evaluate_utils import get_eval_df, parallel_process, calculate_and_save_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Jaccard and Dice scores on predicted and labeled image with multiprocessing")
    parser.add_argument("--prediction_folder", type=str, help="Path to the folder containing predictions")
    parser.add_argument("--label_folder", type=str, help="Path to the folder labels")
    parser.add_argument("--csv_path", type=str, help="Path to csv file with images, labels, and dataset names")
    parser.add_argument("--output_folder", type=str, help="Path to csv file with images, labels, and dataset names")
    parser.add_argument("--eval_disc", action='store_true', help="Whether to evaluate disc disc. Otherwise just evaluate cup")
    parser.add_argument('--disclude_datasets', nargs='+', type=str, help='Datasets to not consider (i.e. RIM-ONE for recovered)')
    parser.add_argument("--num_processes", type=int, help="Number of processes for multiprocessing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prediction_folder = args.prediction_folder
    label_folder = args.label_folder
    csv_path = args.csv_path
    output_folder = args.output_folder
    eval_disc = args.eval_disc
    disclude_datasets = args.disclude_datasets
    num_processes = args.num_processes

    # Make output folder if not exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the dataframe of data to consider
    df = get_eval_df(csv_path, disclude_datasets)

    # Process data in parallel
    results = parallel_process(data=df.to_dict('records'), 
                               prediction_folder=prediction_folder, 
                               label_folder=label_folder, 
                               eval_disc=eval_disc,
                               num_processes=num_processes)

    # Calculate and save the statistics
    calculate_and_save_stats(results, output_folder)
