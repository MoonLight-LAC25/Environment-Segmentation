#!/usr/bin/python3

from PIL import Image
import numpy as np

import sys
import os
import cv2 

# # Get the absolute path of the parent directory
current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    
# # Add the parent directory to sys.path
sys.path.append(current_dir)

from unet import Unet
from image_utils import ImageUtils

class Lunar_UNet(Unet):
    def __init__(self, **kwargs):
        lunar_kwargs = {'count':False, 'input_shape':[640, 640], 'num_classes':4, 'classes':["background","surface","rocks","lander"], 'model_path':"team_code/Perceptron/MAC_SLAM_Lunar/Model/UNet/best_epoch_weights.pth"}
        combined_args = {**lunar_kwargs, **kwargs}
        super().__init__(**combined_args)
        self.image_utils = ImageUtils()
        self.prev_mask = None
        self.mask_store = None

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
        #return Image.fromarray(np.uint8(mod_img))
    @staticmethod
    def erode_class(segmentation_map, target_class, bg_class, kernel_size=3, iterations=1):
        mask = (segmentation_map == target_class).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
        new_segmentation_map = segmentation_map.copy()
        new_segmentation_map[(mask == 1) & (eroded_mask == 0)] = bg_class
        return new_segmentation_map
    
    def set_mask(self, image):
        try:
            self.this_mask = self.erode_class(np.uint8(self.detect_image(image)), target_class=2, bg_class=1)
            self.mask_store = self.this_mask
        except Exception as e:
            print(f"Error in set_mask: {e}")
            raise
    
    def get_binary_rock_mask(self):
        if self.this_mask is None:
            raise ValueError("Mask not set. Call set_mask() first to get rock mask.")
        return self.this_mask == 2
    
    def get_binary_lander_mask(self):
        if self.this_mask is None:
            raise ValueError("Mask not set. Call set_mask() first to get lander mask.")
        return self.this_mask == 3
    
    def clear_mask(self):
        self.this_mask = None

if __name__ == "__main__":
    # Test initialisation for model
    lunar_unet = Lunar_UNet(count=False, input_shape=[640, 640], num_classes=3, classes=["background","surface","rocks"], model_path="logs/moon/run1/best_epoch_weights.pth")
    img_utils = ImageUtils()

    image = Image.open("dataset/test/images/00017.png")
    all_mask = lunar_unet.detect_image(image)
    # all_mask.save("all_classes_mask.png")

    # get binary rock mask
    rock_mask = lunar_unet.get_class_mask(all_mask, [108, 59, 42])
    binary_rock_mask = Image.fromarray(img_utils.binarize_image(masked_img=np.array(rock_mask)))
    binary_rock_mask.save("binary_rocks_mask.png")


    