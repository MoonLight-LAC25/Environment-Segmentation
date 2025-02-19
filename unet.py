import colorsys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.unet import Unet as unet


# ------------------------------------------------#
# Using your own trained model prediction requires modifying 2 parameters.
# Both model_path and num_classes need to be modified!
# If there is a shape mismatch
# Be sure to pay attention to the modification of model_path and num_classes during training.
# ------------------------------------------------#
class Unet(object):
    _defaults = {
        # ------------------------------------------------------------------#
        # model_path points to the weight file in the logs folder
        # After training, there are multiple weight files in the logs folder. Just select the one with the lower loss in the verification set.
        # A lower loss on the verification set does not mean a higher miou, it only means that the weight has better generalization performance on the verification set.
        # ------------------------------------------------------------------#
        "model_path": "",
        # --------------------------------#
        # The number of classes that need to be distinguished +1 (for background)
        # --------------------------------#
        "num_classes": 3,
        # --------------------------------#
        # Backbone network to be used. Options: "vgg", "resnet50"
        # --------------------------------#
        "backbone": "vgg",
        # --------------------------------#
        # Enter the size of the image
        # --------------------------------#
        "input_shape": [640, 640],
        # ------------------------------------------------#
        # The mix_type parameter is used to control the way the detection results are visualized.
        #
        # When mix_type = 0, it means that the original image and the generated image are mixed.
        # mix_type = 1 means only retaining the masks
        # When mix_type = 2, it means that only the background is deducted and only the target in the original image is retained.
        # ------------------------------------------------#
        "mix_type": 1,
        # --------------------------------#
        # Whether to use Cuda
        # Set to False if there is no GPU available
        # --------------------------------#
        "cuda": True,
    }

    # --------------------------------------------------#
    # Initialize UNET
    # --------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # --------------------------------------------------#
        # Set different colors for the picture frame
        # --------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (81, 0, 81), (108, 59, 42), (110, 190, 160), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # --------------------------------------------------#
        # Create the model
        # --------------------------------------------------#
        self.generate()
    
    # ---------------------------------------------------#
    #   Set up the netwoek and device
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        self.net = unet(num_classes = self.num_classes, backbone=self.backbone)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   Segment the image
    # ---------------------------------------------------#
    def detect_image(self, images):
        orininal_w, orininal_h = self.input_shape[1], self.input_shape[0]
        nw, nh = self.input_shape[1], self.input_shape[0]

        with torch.no_grad():
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            #   The image is passed into the network for prediction
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # ---------------------------------------------------#
            #   Get the type of each pixel
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            # --------------------------------------#
            #   Cut off the gray bar part
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            # ---------------------------------------------------#
            #   Get the type of each pixel
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            # ------------------------------------------------#
            #   Convert new picture to image form
            # ------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            # ------------------------------------------------#
            #   Mix the new image with the original image
            # ------------------------------------------------#
            image   = Image.blend(old_img, image, 0.3)

        elif self.mix_type == 1:
            return pr

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            # ------------------------------------------------#
            #   Convert new picture to image form
            # ------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))

        return image
