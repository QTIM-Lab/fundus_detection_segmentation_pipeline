import os
from PIL import Image
import cv2
import numpy as np

class YoloInferenceDataset:
    def __init__(self, root_directory, batch_size, label_directory=None):
        self.root_directory = root_directory
        self.batch_size = batch_size
        self.img_files = [file for file in os.listdir(root_directory) if file.lower().endswith((".png", ".jpg"))]
        self.current_index = 0
        self.label_directory = label_directory

    def __len__(self):
        return len(self.img_files)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.img_files):
            raise StopIteration

        batch_files = self.img_files[self.current_index:self.current_index + self.batch_size]
        batch_images = []
        label_images = []
        for file in batch_files:
            # Open the image using PIL
            image = Image.open(os.path.join(self.root_directory, file))
            image_arr = np.array(image)

            # Convert the image to grayscale
            gray = np.array(image.convert('L'))

            # Threshold the image
            _, thresholded = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            x = 0
            y = 0
            w = image.width
            h = image.height
            # an all black image wont have contours and it throws error
            if len(contours) > 0:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Get the bounding box of the largest contour
                x, y, w, h = cv2.boundingRect(largest_contour)
        

            # Crop the image to the bounding box region
            cropped_image = image_arr[y:y+h, x:x+w, :]

            batch_images.append(Image.fromarray(cropped_image))

            if self.label_directory is not None:
                label = Image.open(os.path.join(self.label_directory, file))
                label_arr = np.array(label)
                cropped_label = label_arr[y:y+h, x:x+w]
                label_images.append(Image.fromarray(cropped_label))

        self.current_index += self.batch_size

        return batch_images, batch_files, label_images
