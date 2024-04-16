from PIL import Image
import numpy as np
import os
import torch

from detection_and_crop.utils.detection_utils import cut_xy
from detection_and_crop.utils.create_yolocropped_utils import get_best_coords_and_conf, get_cropped_image_and_info
from segmentation_train_and_inference.utils.infer_utils import get_color_seg


def get_gray_from_color(color_seg):
    gray_image = np.zeros_like(color_seg[:,:,0])
    gray_image[color_seg[:,:,0] == 255] = 0
    gray_image[color_seg[:,:,1] == 255] = 127
    gray_image[color_seg[:,:,2] == 255] = 255
    return gray_image


def yolo_inference_function(model, img_file_path, threshold=1e-6, output_img_size=512, padding=25):
    image = Image.open(img_file_path)
    orig_height = image.height
    orig_width = image.width
    orig_wh = [orig_width, orig_height]
    # print(f'orig height: {orig_height}, orig width: {orig_width}')
    res = model([image])
    res_i = res[0]

    best_coords, best_conf = get_best_coords_and_conf(res_i, threshold)
    
    # if there was one, get the best one
    if best_coords is not None:
        cropped_image_arr, new_x, new_y, new_w, new_h, rescaled_w, rescaled_h = get_cropped_image_and_info(best_coordinates=best_coords, 
                                                                                                           img=image, 
                                                                                                           output_image_size=output_img_size, 
                                                                                                           pad=padding, 
                                                                                                           file_name=img_file_path)

        recovered_wh = [rescaled_w, rescaled_h]
        
        best_coords_x0 = new_x
        best_coords_x1 = new_x+new_w
        best_coords_y0 = new_y
        best_coords_y1 = new_y+new_h
        recovered_xyxy = [best_coords_x0, best_coords_y0, best_coords_x1, best_coords_y1]

        # Convert back to PIL Image
        cropped_image = Image.fromarray(cropped_image_arr)

        recovered_resize = [512,512]

        if cropped_image.width != 512 or cropped_image.height != 512:
            print("must have had a cut off image because on edge")
            print(f"orig width heights: {cropped_image.width}, {cropped_image.height}")
            recovered_resize = [cropped_image.width, cropped_image.height]
            cropped_image = cropped_image.resize((output_img_size, output_img_size))

        # Image should be good now!
            
        return cropped_image, orig_wh, recovered_wh, recovered_xyxy, recovered_resize

    else:
        print('=!=!=============No best coords for: %s=============!=!=', img_file_path)



def seg_inference_function(model, img_pil, transform, preprocessor, palette, device):
    width, height = img_pil.size
    transformed_image = transform(image=np.array(img_pil))['image'].transpose(2,0,1)

    # prepare image for the model
    inputs = preprocessor(images=transformed_image, return_tensors="pt").to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # predict segmentation maps
    target_sizes = [(height, width)]
    predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                target_sizes=target_sizes)

    # Get predicted map in numpy on cpu
    pred_segmentation_map = predicted_segmentation_maps[0].cpu().numpy()

    # Map it to our colors based on palette
    pred_color_seg = get_color_seg(pred_segmentation_map, palette)

    # Once we have colors, convert to grayscale
    gray_img = get_gray_from_color(pred_color_seg)

    return gray_img


def handle_file(filename, yolo_model, seg_model, transform, preprocessor, palette, device, output_dir):
    print(filename)
    try:
        cropped_img, orig_wh, recovered_wh, recovered_xyxy, recovered_resize = yolo_inference_function(yolo_model, filename)
    except Exception as e:
        print(f"Could not find {filename}!")
        print('error: ', e)
        exit(0)
        return filename

    cropped_img_np = np.array(cropped_img)

    img_pil = Image.fromarray(cropped_img_np)

    seg_output = seg_inference_function(seg_model, img_pil, transform, preprocessor, palette, device)

    seg_output = Image.fromarray(seg_output).resize((recovered_resize[0], recovered_resize[1]))
    seg_output = np.array(seg_output)

    # Create a NumPy array filled with zeros
    recovered_seg = np.zeros((int(recovered_wh[1]), int(recovered_wh[0])), dtype=np.uint8)

    recovered_x0 = int(recovered_xyxy[0])
    recovered_x1 = int(recovered_xyxy[2])
    recovered_y0 = int(recovered_xyxy[1])
    recovered_y1 = int(recovered_xyxy[3])

    recovered_seg[recovered_y0:recovered_y1, recovered_x0:recovered_x1] = seg_output

    recovered_seg_pil = Image.fromarray(recovered_seg).resize((orig_wh[0], orig_wh[1]))

    image = np.array(recovered_seg_pil)
    adjusted_image = np.zeros_like(image)  # Create an array of zeros with the same shape as the input image
    adjusted_image[image < 64] = 0         # Values below 64 are set to 0
    adjusted_image[(image >= 64) & (image <= 191)] = 127  # Values between 64 and 191 are set to 127
    adjusted_image[image > 191] = 255      # Values above 191 are set to 255

    recovered_seg_pil = Image.fromarray(adjusted_image)
    
    output_path = os.path.join(output_dir, os.path.basename(filename))
    recovered_seg_pil.save(output_path)
