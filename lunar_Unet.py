#!/usr/bin/python3

from PIL import Image
import numpy as np

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
    
    def get_mask(self, image):
        self.mask = self.detect_image(image)
    
    def get_binary_rock_mask(self):
        #all_mask = self.detect_image(image)
        return self.mask == 2
    
    def get_binary_lander_mask(self):
        return self.mask == 3

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


    