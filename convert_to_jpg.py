# convert images to jpg

import cv2
import os

dirname = "/home/student/AIMS_CLOUD_1/Abhi/dataset/train_val/og_png_imgs/"
save_dirname = "/home/student/AIMS_CLOUD_1/Abhi/dataset/train_val/imgs/" #needs to exist
# save_file_prefix = "img_000"

print("Loading images...")

images = sorted(os.listdir(dirname))

print("Images loaded: ")
# print(images)
print("__________________________________________")

print("saving images in .jpg format...")

counter = 0

for imagename in images:
    if imagename.endswith(".png"):
        print("converting", imagename)
        image = cv2.imread(dirname+imagename)
        # cv2.imwrite(save_dirname+save_file_prefix+str(counter)+".jpg", image)
        imname = imagename.split('.')[0]
        cv2.imwrite(save_dirname+imname+".jpg", image)
        
        # counter = counter + 1

print("saved images in: "+save_dirname)


