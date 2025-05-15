"""
Inference script for predicting malignancy of lung nodules
"""
import numpy as np
import dataloader
import torch
import torch.nn as nn
from torchvision import models
from models.model_2d import ResNet18
from models.model_3d import I3D
from models.model_vit import DeiTSmall
from models.model_mobilenet_v3_large import MobileNetLarge
from models.model_convnexttiny import ConvNextTiny
from models.model_convnexttinyv2 import ConvNextTinyV2
from models.model_medicalnets import MedicalNetModel
import os
import math
import logging
import timm

from collections import OrderedDict


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

# define processor
class MalignancyProcessor:
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(self, mode="2D", suppress_logs=False, model_name="LUNA25-baseline-2D"):

        self.size_px = 224 #64
        self.size_mm = 50

        self.model_name = model_name
        self.mode = mode
        self.suppress_logs = suppress_logs

        if not self.suppress_logs:
            logging.info("Initializing the deep learning system")

        if self.mode == "2D":
            self.model_2d = ResNet18(weights=None).cuda()
        elif self.mode == "vit":
            self.model_vit = DeiTSmall(pretrained=False).cuda()
        elif self.mode == "mobilenetv3L":
            self.model_mobilenet_v3_large = MobileNetLarge(pretrained=False).cuda()
        elif self.mode == "convnexttiny":
            self.model_mobilenet_v3_large = ConvNextTiny(pretrained=False).cuda()
        elif self.mode == "convnexttinyv2":
            self.model_mobilenet_v3_large = ConvNextTinyV2(pretrained=False).cuda()
        elif self.mode == "3D":
            self.model_3d = I3D(num_classes=1, pre_trained=False, input_channels=3).cuda()
        elif self.mode == "MedicalNetResnet10":
            self.model_medicalnets = MedicalNetModel(model_depth=10, pretrained=False).cuda()
        elif self.mode == "MedicalNetResnet18":
            self.model_medicalnets = MedicalNetModel(model_depth=18, pretrained=False).cuda()
        elif self.mode == "MedicalNetResnet34":
            self.model_medicalnets = MedicalNetModel(model_depth=34, pretrained=False).cuda()
        elif self.mode == "MedicalNetResnet50":
            self.model_medicalnets = MedicalNetModel(model_depth=50, pretrained=False).cuda()
        elif self.mode == "MedicalNetResnet101":
            self.model_medicalnets = MedicalNetModel(model_depth=101, pretrained=False).cuda()
        elif self.mode == "MedicalNetResnet152":
            self.model_medicalnets = MedicalNetModel(model_depth=152, pretrained=False).cuda()
        elif self.mode == "MedicalNetResnet200":
            self.model_medicalnets = MedicalNetModel(model_depth=200, pretrained=False).cuda()

        self.model_root = "/opt/app/resources/"
        # self.model_root = "results"

    def define_inputs(self, image, header, coords):
        self.image = image
        self.header = header
        self.coords = coords

    def extract_patch(self, coord, output_shape, mode):

        patch = dataloader.extract_patch(
            CTData=self.image,
            coord=coord,
            srcVoxelOrigin=self.header["origin"],
            srcWorldMatrix=self.header["transform"],
            srcVoxelSpacing=self.header["spacing"],
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            coord_space_world=True,
            mode=mode,
        )

        # ensure same datatype...
        patch = patch.astype(np.float32)

        # clip and scale...
        patch = dataloader.clip_and_scale(patch)
        return patch

    # def add_prefix_to_state_dict(self, state_dict, prefix="model."):
    #     new_state_dict = OrderedDict()

    #     for k, v in state_dict.items():
    #         if not k.startswith(prefix):
    #             new_key = prefix + k
    #         else:
    #             new_key = k  # already prefixed
    #         new_state_dict[new_key] = v
    #     return new_state_dict


    def _process_model(self, mode):

        if not self.suppress_logs:
            logging.info("Processing in " + mode)

        if mode == "2D":
            output_shape = [1, self.size_px, self.size_px]
            model = self.model_2d
        elif mode =="vit":
            output_shape = [1, self.size_px, self.size_px]
            model = self.model_vit
        elif mode == "mobilenetv3L":
            output_shape = [1, self.size_px, self.size_px]
            model = self.model_mobilenet_v3_large
        elif mode == "convnexttiny":
            output_shape = [1, self.size_px, self.size_px]
            model = self.model_convnexttiny
        elif mode == "convnexttinyv2":
            output_shape = [1, self.size_px, self.size_px]
            model = self.model_convnexttinyv2
        elif mode == "MedicalNetResnet10" or mode == "MedicalNetResnet18" or mode == "MedicalNetResnet34" \
            or mode == "MedicalNetResnet50" or mode == "MedicalNetResnet101" or mode == "MedicalNetResnet152" \
                or mode == "MedicalNetResnet200":
            output_shape = [self.size_px, self.size_px, self.size_px]
            model = self.model_medicalnets
        else:
            output_shape = [self.size_px, self.size_px, self.size_px]
            model = self.model_3d

        nodules = []

        for _coord in self.coords:

            patch = self.extract_patch(_coord, output_shape, mode=mode)
            nodules.append(patch)

        nodules = np.array(nodules)
        nodules = torch.from_numpy(nodules).cuda()


        print(f"DEBUG: Loading model from {os.path.join(self.model_root, self.model_name, 'best_metric_model.pth')}")
        
        ckpt = torch.load(
            os.path.join(
                self.model_root,
                self.model_name,
                "best_metric_model.pth",
            )
        )
        
        # Handle if it contains a full checkpoint dict
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        
        # Add "model." prefix to keys
        # ckpt = self.add_prefix_to_state_dict(ckpt, prefix="model.")

        
        model.load_state_dict(ckpt)
        model.eval()
        logits = model(nodules)
        logits = logits.data.cpu().numpy()

        logits = np.array(logits)
        return logits

    def predict(self):

        logits = self._process_model(self.mode)

        probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return probability, logits