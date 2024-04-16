import torch
import numpy as np
import pandas as pd
import torchmetrics
import os
import multiprocessing as mp
from PIL import Image


def image_pil_to_tensor(image_pil, eval_disc):
    # Load image as PIL image
    image = np.array(image_pil)
    print('image shape: ', image.shape)

    # Convert PIL image to tensor and flatten to 1D
    tensor = torch.tensor(image).view(-1, 3)
    
    # Define RGB to class mapping (adjust values based on your image)
    if eval_disc:
        class_mapping = {(255, 0, 0): 0, (0, 255, 0): 1, (0, 0, 255): 1}
    else:
        class_mapping = {(255, 0, 0): 0, (0, 255, 0): 0, (0, 0, 255): 1}
    
    # Map RGB values to class labels
    class_labels = torch.tensor([class_mapping[tuple(rgb.tolist())] for rgb in tensor])
    
    # Reshape to 2D
    class_labels = class_labels.view(image.shape[1], image.shape[0])
    
    return class_labels


def grayscale_path_to_image_pil(image_path):
    # Load image as PIL image
    gray_img = np.array(Image.open(image_path).convert('L'))
    # Create a 3-channel image with the same shape as the grayscale image
    return grayscale_to_image_pil(gray_img)


def grayscale_to_image_pil(gray_image):
    # Create a 3-channel image with the same shape as the grayscale image
    image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)

    # Assign red, green, and blue colors based on intensity values
    image[gray_image == 0] = [255, 0, 0]   # Values that were 0 become red
    image[gray_image == 127] = [0, 255, 0]  # Values that were 127 become green
    image[gray_image == 255] = [0, 0, 255]  # Values that were 255 become blue

    return Image.fromarray(image)

def process_row(row, prediction_folder, label_folder, eval_disc, jaccard_metric, dice_metric):
    image_filename = row['image_filename']
    label_filename = row['label_filename']

    # Get paths to predicted and label
    pred_path = os.path.join(prediction_folder, image_filename)
    label_path = os.path.join(label_folder, label_filename)

    # Get the PIL images
    color_prediction = grayscale_path_to_image_pil(pred_path)
    color_label = grayscale_path_to_image_pil(label_path)

    # Convert to tensors with appropriate labeling (disc = disc+cup, cup = cup)
    pred_tensor = image_pil_to_tensor(color_prediction, eval_disc).float()
    label_tensor = image_pil_to_tensor(color_label, eval_disc)

    # Check if no label, in which case metric cannot be calculated
    if torch.all(label_tensor == 0):
        jaccard_value = -np.inf
        dice_value = -np.inf
        print(f'Filename: {image_filename} has empty label, cannot calculate metric')
    else:
        # Calculate the jaccard and dice scores using torchmetrics implementation
        jaccard_value = jaccard_metric(pred_tensor, label_tensor)
        dice_value = dice_metric(pred_tensor, label_tensor)

    print(f'Filename: {image_filename}, Jaccard: {jaccard_value}, Dice: {dice_value}')
    return (row['dataset_name'], image_filename, label_filename, jaccard_value, dice_value)

def parallel_process(data, prediction_folder, label_folder, eval_disc, num_processes):
    # Define metrics
    jaccard_metric = torchmetrics.classification.BinaryJaccardIndex()
    dice_metric = torchmetrics.classification.Dice(average='micro')
    pool = mp.Pool(num_processes)
    results = pool.starmap(process_row, [(row, prediction_folder, label_folder, eval_disc, jaccard_metric, dice_metric) for row in data])  # Pass additional_param to process_row
    pool.close()
    pool.join()
    return results

def calculate_and_save_stats(res, output_folder):
    # Convert results to DataFrame
    result_df = pd.DataFrame(res, columns=['dataset_name', 'image_path', 'label_path', 'jaccard_value', 'dice_value'])

    individual_scores_csv_path = os.path.join(output_folder, 'individual_scores.csv')
    dataset_scores_csv_path = os.path.join(output_folder, 'dataset_scores.csv')

    # Save the individual scores
    result_df.to_csv(individual_scores_csv_path, index=False)

    # Get rid of negative values, which came from nan/empty labels
    result_df = result_df[(result_df['jaccard_value'] >= 0) & (result_df['dice_value'] >= 0)]

    # Calculate the mean and standard deviation for each dataset
    dataset_stats = result_df.groupby('dataset_name').agg({'jaccard_value': ['mean', 'std'], 'dice_value': ['mean', 'std']}).reset_index()
    
    # Rename columns for clarity
    dataset_stats.columns = ['dataset_name', 'mean_jaccard', 'std_jaccard', 'mean_dice', 'std_dice']

    # Save the dataset statistics
    dataset_stats.to_csv(dataset_scores_csv_path, index=False)

def get_eval_df(csv_path, disclude_datasets):
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Do not consider discluded datasets
    if disclude_datasets is not None:
        for discluded in disclude_datasets:
            print('discluding: ', discluded)
            df = df[df['dataset_name'] != discluded]

    return df
