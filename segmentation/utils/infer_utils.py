from PIL import Image
import os
import csv
import cv2
import numpy as np


def get_color_seg(seg_map, palette):
    """
    Get the color segmentation map as defined by the palette
    """

    color_segmentation_map = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8) # height, width, 3
    
    for label, color in enumerate(palette):
        color_segmentation_map[seg_map - 1 == label, :] = color

    # Convert to BGR
    return color_segmentation_map[..., ::1]

def get_label_color_seg(label_seg_path):
    """
    Get a color segmentation map given how we encode seg maps (0=bg, 127=disc, 255=cup)
    """

    label_gray_seg = np.array(Image.open(label_seg_path).convert('L'))

    # Create a blank color image of the same size as the grayscale image
    label_color_seg = np.zeros((label_gray_seg.shape[0], label_gray_seg.shape[1], 3), dtype=np.uint8)

    # Set pixels corresponding to grayscale values to the corresponding colors
    label_color_seg[label_gray_seg == 0] = (255, 0, 0)    # Red
    label_color_seg[label_gray_seg == 127] = (0, 255, 0)  # Green
    label_color_seg[label_gray_seg == 255] = (0, 0, 255)  # Blue

    return label_color_seg

def get_label_gray_seg(label_col_seg):
    """
    Create grayscale segmentation map
    """
    
    # Create a grayscale image array
    grayscale = np.zeros_like(label_col_seg[:, :, 0])  # Same shape as the height and width of the original image

    # Assign pixel values based on color
    grayscale[np.all(label_col_seg == [255, 0, 0], axis=-1)] = 0   # Red pixels
    grayscale[np.all(label_col_seg == [0, 255, 0], axis=-1)] = 127 # Green pixels
    grayscale[np.all(label_col_seg == [0, 0, 255], axis=-1)] = 255 # Blue pixels

    return grayscale


def get_contours(color_seg, channel_idx):
    """
    get contours given the color segmentation map, for the desired channel
    i.e. channel 1 is disc, channel 2 is cup, as defined by palette
    """

    contours, _ = cv2.findContours(color_seg[:,:,channel_idx], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contours = max(contours, key=cv2.contourArea)

    return contours

def find_largest_difference(contour_dict):
    """
    Find the largest difference between y values at any x coordinate
    """

    max_difference = 0
    max_x, max_y1, max_y2 = None, None, None

    for x, (min_y, max_y) in contour_dict.items():
        difference = max_y - min_y
        if difference > max_difference:
            max_difference = difference
            max_x = x
            max_y1 = min_y
            max_y2 = max_y

    return max_x, max_y1, max_y2

def min_max_y_contour(contour):
    """
    Find the min and max y values for all x values in the contour
    """

    # Create an empty dictionary to store min and max y values for each x pixel
    min_max_y_dict = {}

    # Iterate through each point in the contour
    for point in contour:
        x, y = point

        # Check if x already exists in the dictionary
        if x in min_max_y_dict:
            # Update min and max y values if necessary
            min_max_y_dict[x][0] = min(min_max_y_dict[x][0], y)
            min_max_y_dict[x][1] = max(min_max_y_dict[x][1], y)
        else:
            # Create a new entry for x with initial min and max y values
            min_max_y_dict[x] = [y, y]

    # Fill in the missing x values
    filled_dict = {}
    x_values = sorted(min_max_y_dict.keys())
    for i in range(len(x_values) - 1):
        x_start, x_end = x_values[i], x_values[i + 1]
        y_start, y_end = min_max_y_dict[x_start], min_max_y_dict[x_end]

        # Fill in missing x values between x_start and x_end
        for x in range(x_start + 1, x_end):
            filled_dict[x] = [min(y_start[0], y_end[0]), max(y_start[1], y_end[1])]

    # Merge original dictionary with filled dictionary
    min_max_y_dict.update(filled_dict)

    return min_max_y_dict

def get_max_vertical_diameter(seg_contours, seg_map=None, line_thickness=1):
    """
    get the max vertical diameter given contours, and draw it on the seg map if provided
    """

    sorted_contours = np.squeeze(seg_contours)

    if len(sorted_contours) <= 1:
        return 0

    min_max_values = min_max_y_contour(sorted_contours)
    x_max, y1_max, y2_max = find_largest_difference(min_max_values)

    longest_length = abs(y2_max - y1_max)

    if longest_length > 0 and seg_map is not None:
        cv2.line(seg_map, (x_max, y1_max), (x_max, y2_max), (255,255,255), line_thickness)

    return longest_length


def get_data_rows(res_queue):
    """
    Get all data from the result queue into list (for rows in csv)
    """

    rows_list = []

    while not res_queue.empty():
        row = res_queue.get()
        rows_list.append(row)

    return rows_list

def write_output_csv(output_csv_path, row_data):
    """
    Write the filtered data to a new CSV file
    """

    with open(output_csv_path, 'w', newline='') as output_csv:
        fieldnames = row_data[0].keys()
        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_data)

def get_cdr_from_color_seg(seg_color):
    """
    Get Cup-Disc Ratio from a Color Seg map
    
    Assumes bg is red channel 0, disc is green channel 1, cup blue channel 2
    """

    # Get disc and cup contours
    disc_contours = get_contours(seg_color, channel_idx=1)
    cup_contours = get_contours(seg_color, channel_idx=2)

    # Get disc and cup max vertical diameter
    disc_diameter = get_max_vertical_diameter(disc_contours, seg_map=None)
    cup_diameter = get_max_vertical_diameter(cup_contours, seg_map=None)

    # Get CDR based on disc and cup diameter
    cdr = 0 if disc_diameter == 0 else cup_diameter / disc_diameter

    return cdr

def process_segmentation(pred_segmentation_map, filename, result_queue, palette, item_index, output_dir, label_segmentation_map):
    """
    Multiprocess call script to process a segmentation output for an associated image filename
    """
    
    # Get the color seg map given the model seg output (which is gray scale)
    pred_color_seg = get_color_seg(pred_segmentation_map, palette)

    pred_cdr = get_cdr_from_color_seg(pred_color_seg)
    label_cdr = pred_cdr if label_segmentation_map is None else get_cdr_from_color_seg(label_segmentation_map)

    # Convert the NumPy array to a PIL image
    # seg_output = Image.fromarray(pred_color_seg.astype('uint8'))
    # seg_output = Image.fromarray(pred_segmentation_map.astype('uint8'))
    seg_output = get_label_gray_seg(pred_color_seg)
    seg_output = Image.fromarray(seg_output.astype('uint8'))

    # Save the image
    seg_output.save(os.path.join(os.path.join(output_dir, 'outputs'), filename))

    row = {
        'pred_cdr': pred_cdr,
        'label_cdr': label_cdr,
        'image_file': filename
    }

    result_queue.put(row)

    print(f"processed output for (file, num): ({filename}, {item_index}), with pred cdr: {pred_cdr}, label cdr: {label_cdr}")
