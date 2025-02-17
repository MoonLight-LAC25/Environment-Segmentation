#!/usr/bin/python3

from PIL import Image
import numpy as np
import cv2

import sys
import os

# # Get the absolute path of the parent directory
current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    
# # Add the parent directory to sys.path
sys.path.append(current_dir)

from unet import Unet
from image_utils import ImageUtils

class Lunar_UNet(Unet):
    def __init__(self, **kwargs):
        lunar_kwargs = {'count':False, 'input_shape':[640, 640], 'num_classes':3, 'classes':["background","surface","rocks"], 'model_path':"logs/moon/run1/best_epoch_weights.pth"}
        combined_args = {**lunar_kwargs, **kwargs}
        super().__init__(**combined_args)
        self.image_utils = ImageUtils()

    def get_class_mask(self, mask, class_pixels_rgb):
        mask = np.array(mask)
        mod_img = np.zeros(
            [np.shape(mask)[0], np.shape(mask)[1], np.shape(mask)[2]]
        )
        if mask.ndim == 3:
            for height in range(mask.shape[0]):
                for width in range(mask.shape[1]):
                    if (mask[height][width] == class_pixels_rgb).all(): 
                      mod_img[height][width] = mask[height][width][0]
        return Image.fromarray(np.uint8(mod_img))
    
    def get_binary_rock_mask(self, image):
        all_mask = self.detect_image(image)
        rock_mask = self.get_class_mask(all_mask, [108, 59, 42])
        binary_rock_mask = self.image_utils.binarize_image(masked_img=np.array(rock_mask))
        return binary_rock_mask

if __name__ == "__main__":
    # Test initialisation for model
    lunar_unet = Lunar_UNet(count=False, input_shape=[640, 640], num_classes=3, classes=["background","surface","rocks"], model_path="logs/moon/run1/best_epoch_weights.pth")

    # Example Usage
    # image = Image.open("path")
    # binary_rock_mask = lunar_unet.get_binary_rock_mask(image)


    