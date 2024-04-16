import numpy as np


def square_crop_with_padding(image_pil, label_pil, rect, padding, max_offset=0.5):
    """Crops a square region from the image, containing the given rectangle with padding.

    Args:
        image: The image to crop.
        x1, y1, x2, y2: Coordinates of the rectangle to include in the crop.
        padding: Amount of padding to add around the rectangle.

    Returns:
        The cropped square image.
    """
    image = np.array(image_pil)
    label = np.array(label_pil)
    # Calculate dimensions of the rectangle and desired crop
    x1, y1, x2, y2 = rect

    # Calculate dimensions of the rectangle and desired crop
    rect_width = x2 - x1
    rect_height = y2 - y1
    crop_size = max(rect_width, rect_height) + 2 * padding

    # Generate random offsets within specified limits
    offset_x = int(np.random.uniform(-max_offset * rect_width, max_offset * rect_width))
    offset_y = int(np.random.uniform(-max_offset * rect_height, max_offset * rect_height))

    # Adjust top-left corner to account for offsets and stay within image bounds
    # top = max(0, min(y1 - padding + offset_y, image.shape[0] - crop_size))
    # left = max(0, min(x1 - padding + offset_x, image.shape[1] - crop_size))

    # # Adjust bottom and right edges
    # bottom = min(top + crop_size, image.shape[0])
    # right = min(left + crop_size, image.shape[1])

    top = y1 - padding + offset_y
    left = x1 - padding + offset_x
    bottom = top + crop_size
    right = left + crop_size

    if top < 0 or left < 0 or bottom > image.shape[0] or right > image.shape[1]:
        return None, None

    # Ensure square crop size based on available area
    crop_size = min(bottom - top, right - left)

    # Perform the square crop with the potentially offset rectangle
    crop = image[top:top+crop_size, left:left+crop_size]
    crop_label = label[top:top+crop_size, left:left+crop_size]

    return crop, crop_label

def square_crop_with_padding_no_label(image_pil, rect, padding, max_offset=0.5):
    """Crops a square region from the image, containing the given rectangle with padding.

    Args:
        image: The image to crop.
        x1, y1, x2, y2: Coordinates of the rectangle to include in the crop.
        padding: Amount of padding to add around the rectangle.

    Returns:
        The cropped square image.
    """
    image = np.array(image_pil)
    # Calculate dimensions of the rectangle and desired crop
    x1, y1, x2, y2 = rect

    # Calculate dimensions of the rectangle and desired crop
    rect_width = x2 - x1
    rect_height = y2 - y1
    crop_size = max(rect_width, rect_height) + 2 * padding

    # Generate random offsets within specified limits
    offset_x = int(np.random.uniform(-max_offset * rect_width, max_offset * rect_width))
    offset_y = int(np.random.uniform(-max_offset * rect_height, max_offset * rect_height))

    top = y1 - padding + offset_y
    left = x1 - padding + offset_x
    bottom = top + crop_size
    right = left + crop_size

    if top < 0 or left < 0 or bottom > image.shape[0] or right > image.shape[1]:
        return None, None

    # Ensure square crop size based on available area
    crop_size = min(bottom - top, right - left)

    # Perform the square crop with the potentially offset rectangle
    crop = image[top:top+crop_size, left:left+crop_size]

    return crop
