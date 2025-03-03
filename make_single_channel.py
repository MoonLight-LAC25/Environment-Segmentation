# to make multi-channel masks into single channel masks
import numpy as np
#import cupy as np
from PIL import Image
from labelme import utils
import os
from tqdm import tqdm
import multiprocessing


#filepath = "img_03.png"
#savepath = "img_03_sc.png"

#@jit(target_backend='cuda')
def get_mod_mask(npa):
    # mask_color_type_1=[255, 106, 77]
    # mask_color_type_2=[250, 250, 55]
    mask_color_type_1 = [81, 0, 81]
    mask_color_type_2 = [108, 59, 42]
    mask_color_type_3 = [110, 190, 160]
    if npa.ndim == 3:
        mod_img = np.zeros([np.shape(npa)[0], np.shape(npa)[1]])
        for height in range(npa.shape[0]):
            for width in range(npa.shape[1]):
                # in each pixel
                mask_channels = npa[height][width] # get the RGB channels
                for _ in range(mask_channels.shape[0]):
                    # making the value of each pixel of the image is the type to which the pixel belongs. here I check the color of the pixel and assign it a value accordingly
                    if (mask_channels == mask_color_type_1).all():
                        mod_img[height][width] = 1
                    elif (mask_channels == mask_color_type_2).all():
                        mod_img[height][width] = 2
                    elif (mask_channels == mask_color_type_3).all():
                        mod_img[height][width] = 3
                    else:
                        mod_img[height][width] = 0
        return mod_img
            

def process_masks_multithread(load_folderpath, save_folderpath):
    multichannel_mask_names = os.listdir(load_folderpath)
    
    print("There are "+str(len(multichannel_mask_names))+" to convert. Loading:")

    multichannel_masks = []

    # load all masks
    for filepath in tqdm(multichannel_mask_names, total=len(multichannel_mask_names)):
        image = Image.open(load_folderpath+filepath)
        npa = np.array(image)
        multichannel_masks.append(npa)
        # print("Dimensions of ", filepath, ":", npa.ndim)

    print("Running using all available cores...")
    pool = multiprocessing.Pool()
    single_channel_masks = tqdm(pool.imap(get_mod_mask, multichannel_masks), total=len(multichannel_masks))

    counter = 0
    for filepath in tqdm(multichannel_mask_names, total=len(multichannel_mask_names)):
        # mod_img = get_mod_mask(npa)
        savepath = save_folderpath+filepath
        utils.lblsave(savepath, single_channel_masks[counter])
        counter+=1
        
    print("Saved masks to", save_folderpath)

def process_masks(load_folderpath, save_folderpath):
    multichannel_mask_names = os.listdir(load_folderpath)
    
    print("There are "+str(len(multichannel_mask_names))+" to convert. Processing:")
    # load all masks
    for filepath in tqdm(multichannel_mask_names, total=len(multichannel_mask_names)):
        image = Image.open(load_folderpath+filepath)
        npa = np.array(image)
        mod_img = get_mod_mask(npa)
        savepath = save_folderpath+filepath
        utils.lblsave(savepath, mod_img)
    
    print("Saved masks to", save_folderpath)

            
if __name__ == "__main__":
    
    load_folderpath = "/home/abhi/Documents/Lunar/dataset/train_val/multichannel_masks/"
    save_folderpath = "/home/abhi/Documents/Lunar/dataset/train_val/masks/"

    process_masks(load_folderpath=load_folderpath, save_folderpath=save_folderpath)   
    
        
