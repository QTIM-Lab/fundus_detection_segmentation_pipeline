import numpy as np
import os
from PIL import Image

from detection_and_crop.utils.detection_utils import cut_xy


def get_best_coords_and_conf(result_i, threshold):
    best_coords = None
    best_conf = None
    # Sometimes, it will make no predictions (no objects i.e. all white image)
    # Sometimes makes more than 1, which isn't a big problem is first (best) one
    # is good...
    if result_i.boxes.conf.numel() > 0:
        best_conf = result_i.boxes.conf[0].item()
        # check it is above threshold
        if best_conf >= threshold:
            best_coords = result_i.boxes.xyxy.detach().cpu().numpy()[0]
    return best_coords, best_conf

def get_rescaled_h_w(img, output_image_size, ht, wd, pad):
    # Rescale the original image such that the disc boundary is square, and
    # sized output_img_size - padding * 2
    # if the entire image is going to be 512,512, then the disc extent
    # should be 512 - padding*2
    scale_factor_h = (output_image_size - (pad * 2)) / ht
    scale_factor_w = (output_image_size - (pad * 2)) / wd
    rescaled_h = int(img.height * scale_factor_h)
    rescaled_w = int(img.width * scale_factor_w)
    return scale_factor_h, scale_factor_w, rescaled_h, rescaled_w

def get_new_xy_wh(x_val, y_val, sf_w, sf_h, output_image_size, pad):
    new_x = int((x_val * sf_w) - pad)
    new_y = int((y_val * sf_h) - pad)
    new_w = output_image_size
    new_h = output_image_size
    return new_x, new_y, new_w, new_h

def keep_xywh_within_bounds(x, y, w, h, img_resized, file_name):
    if x < 0:
        print(f"hit: {img_resized.width}, {w}, {x}, {file_name}")
        # find amount it is being cut off by, if it is less than y
        if x < y:
            cut_diff = abs(x)
        else:
            cut_diff = abs(y)
        x, y, w, h = cut_xy(x, y, w, h, cut_diff)
    elif y < 0:
        print(f"hit: {img_resized.height}, {h}, {y}, {file_name}")
        # find amount it is being cut off by, if it is less than y
        cut_diff = abs(y)
        x, y, w, h = cut_xy(x, y, w, h, cut_diff)
    elif img_resized.width < x + w:
        print(f"hit: {img_resized.width}, {w}, {x}, {file_name}")
        cut_width = (x + w) - img_resized.width
        cut_height = (y + h) - img_resized.height
        if (cut_width < cut_height):
            cut_diff = cut_height
        else:
            cut_diff = cut_width
        x, y, w, h = cut_xy(x, y, w, h, cut_diff)
    elif img_resized.height - h < y:
        print(f"hit: {img_resized.height}, {h}, {y}, {file_name}")
        cut_diff = (y + h) - img_resized.height
        x, y, w, h = cut_xy(x, y, w, h, cut_diff)

    return x, y, w, h

def get_cropped_image_and_info(best_coordinates, img, output_image_size, pad, file_name):
    x = best_coordinates[0]
    y = best_coordinates[1]
    w = best_coordinates[2] - best_coordinates[0]
    h = best_coordinates[3] - best_coordinates[1]

    # Rescale the original img such that the disc boundary is square, and
    # sized output_image_size - pad * 2
    # if the entire img is going to be 512,512, then the disc extent
    # should be 512 - pad*2
    scale_factor_h, scale_factor_w, rescaled_h, rescaled_w = get_rescaled_h_w(img=img, 
                                                                            output_image_size=output_image_size, 
                                                                            ht=h, 
                                                                            wd=w, 
                                                                            pad=pad)

    image_resized = img.resize((rescaled_w, rescaled_h))

    image_array_resized = np.array(image_resized)

    new_x, new_y, new_w, new_h = get_new_xy_wh(x_val=x, 
                                                y_val=y, 
                                                sf_w=scale_factor_w, 
                                                sf_h=scale_factor_h, 
                                                output_image_size=output_image_size, 
                                                pad=pad)
    
    new_x, new_y, new_w, new_h = keep_xywh_within_bounds(x=new_x, 
                                                            y=new_y, 
                                                            w=new_w, 
                                                            h=new_h, 
                                                            img_resized=image_resized, 
                                                            file_name=file_name)

    # Crop the images
    cropped_image_array = image_array_resized[new_y:new_y+new_h, new_x:new_x+new_w]

    return cropped_image_array, new_x, new_y, new_w, new_h, rescaled_w, rescaled_h


def process_result(queue, res_i, image, filename, output_directory, batch_number, item_number, threshold=1e-6, output_img_size=512, padding=25, label_image=None, label_output_directory=None):
    best_coords, best_conf = get_best_coords_and_conf(res_i, threshold)
    
    # if there was one, get the best one
    if best_coords is not None:
        cropped_image_arr, new_x, new_y, new_w, new_h, rescaled_w, rescaled_h = get_cropped_image_and_info(best_coordinates=best_coords, 
                                                                                                           img=image, 
                                                                                                           output_image_size=output_img_size, 
                                                                                                           pad=padding, 
                                                                                                           file_name=filename)

        # Convert back to PIL Image
        cropped_image = Image.fromarray(cropped_image_arr)

        if cropped_image.width != output_img_size or cropped_image.height != output_img_size:
            print("must have had a cut off image because on edge")
            print(f"orig width heights: {cropped_image.width}, {cropped_image.height}")
            cropped_image = cropped_image.resize((output_img_size, output_img_size))

        # Image should be good now!

        output_path = os.path.join(output_directory, filename)
        # Split the file path into base and extension
        output_base, ext = os.path.splitext(output_path)
        # Ensure the extension is lowercase and ends with ".png"
        output_path = output_base + ".png"
        cropped_image.save(output_path)

        res_row = {
            'filename': filename,
            'conf': best_conf,
            'x': best_coords[0],
            'y': best_coords[1],
            'w': best_coords[2],
            'h': best_coords[3]
        }
        queue.put(res_row)

        if label_image is not None:
            label_image_resized = label_image.resize((rescaled_w, rescaled_h))
            label_array_resized = np.array(label_image_resized)
            cropped_label_arr = label_array_resized[new_y:new_y+new_h, new_x:new_x+new_w]
            cropped_label = Image.fromarray(cropped_label_arr)

            if cropped_label.width != output_img_size or cropped_label.height != output_img_size:
                cropped_label = Image.fromarray(cropped_label_arr)
                cropped_label = cropped_label.resize((output_img_size, output_img_size))

            cropped_label_arr = np.array(cropped_label)

            # Define your threshold values
            thresholds = [64, 191]

            # Apply thresholding
            cropped_label_arr[cropped_label_arr < thresholds[0]] = 0
            cropped_label_arr[(cropped_label_arr >= thresholds[0]) & (cropped_label_arr <= thresholds[1])] = 127
            cropped_label_arr[cropped_label_arr > thresholds[1]] = 255

            # Convert back to PIL Image
            cropped_label = Image.fromarray(cropped_label_arr)

            # get the filename with split and join it with output dir
            label_output_path = os.path.join(label_output_directory, filename)
            # Split the file path into base and extension
            label_base, ext = os.path.splitext(label_output_path)
            # Ensure the extension is lowercase and ends with ".png"
            label_output_path = label_base + ".png"
            cropped_label.save(label_output_path)

    else:
        print('=!=!=============No best coords for: %s=============!=!=', filename)
    
    print(f"Processed image: {filename}, in (batch, item): ({batch_number}, {item_number})")