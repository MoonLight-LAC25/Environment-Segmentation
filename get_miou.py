import os

from PIL import Image
from tqdm import tqdm

from unet import Unet
from utils.utils_metrics import compute_mIoU, show_results

''' You need to pay attention to the following points when evaluating indicators: 1. The image generated by this file is a grayscale image. Because the value is relatively small, there is no display effect when looking at the image in jpg format, so the image that is almost completely black is normal. 2. This file calculates the miou of the verification set. Currently, the library uses the test set as the verification set and does not divide the test set separately. 3. Only models trained according to voc format data can use this file to calculate miou.
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    # miou_mode is used to specify the content calculated when the file is run.
    # miou_mode is 0, which represents the entire miou calculation process, including obtaining prediction results and calculating miou.
    # miou_mode is 1, which means only the prediction results are obtained.
    # miou_mode is 2, which means only miou is calculated.
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   Number of categories + 1, such as 2 + 1
    #------------------------------#
    num_classes     = 3
    #--------------------------------------------#
    #   The type of distinction is the same as in json to dataset
    #--------------------------------------------#
    name_classes    = ["background","top_layer","under_layer"]
    #-------------------------------------------------------#
    # Point to the folder where the VOC data set is located
    # Default points to the VOC data set in the root directory
    #-------------------------------------------------------#
    VOCdevkit_path  = 'datasets/Dataset_0220_bule'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        unet = Unet()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)