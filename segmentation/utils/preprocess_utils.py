from multiprocessing import Pool
import numpy as np
from PIL import Image


def calculate_mean_std_parallel(img_path):
    img_array = np.array(Image.open(img_path)) / 255
    sum_channels = np.sum(img_array, axis=(0, 1))
    sum_squared_channels = np.sum(img_array ** 2, axis=(0, 1))

    return sum_channels, sum_squared_channels

def calculate_mean_std(image_paths, image_size=(512,512), num_processes=None):
    num_images = len(image_paths)
    sum_channels = np.zeros((3,), dtype=np.float64)
    sum_squared_channels = np.zeros((3,), dtype=np.float64)

    with Pool(processes=num_processes) as pool:
        results = pool.map(calculate_mean_std_parallel, image_paths)

    for sum_ch, sum_squared_ch in results:
        sum_channels += sum_ch
        sum_squared_channels += sum_squared_ch

    mean_channels = sum_channels / (num_images * image_size[1] * image_size[0])
    std_channels = np.sqrt(
        (sum_squared_channels / (num_images * image_size[1] * image_size[0])) - mean_channels ** 2
    )

    return mean_channels, std_channels
