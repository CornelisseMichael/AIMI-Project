"""
Inference script for predicting malignancy of lung nodules
"""
import numpy as np
import dataloader
import torch
import torch.nn as nn
from torchvision import models
from models.model_2d import ResNet18, ResNet50
from models.model_3d import I3D
from models.model_vit import DeiTSmall
from models.model_mobilenet_v3_large import MobileNetLarge
from models.model_convnexttiny import ConvNextTiny
from models.model_convnexttinyv2 import ConvNextTinyV2
from models.model_medicalnets import MedicalNetModel
from models.ensemble_model import EnsembleWrapper
from models.EnsembleInference import Ensemble
from models.Ensemble3d_deit import Ensemble3d_deit
from models.SwinTiny import SwinTiny
from models.SwinBase import SwinBase
from models.DeitSmall import DeiTSmall
from models.DeitBase import DeiTBase
import os
import math
import logging
import timm
import experiment_config as config
from collections import OrderedDict
config = config.Configuration()
DEVICE = config.DEVICE


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

    def __init__(self, mode="ensemble", suppress_logs=False, model_name="LUNA25-baseline-2D"):

        # model folder
        self.model_root = "/opt/app/resources/"
        #self.model_root = "results"

        
        self.size_px = 224 # Change this to reflect the input size
        self.size_mm = 50

        self.model_name = model_name
        self.mode = mode
        self.suppress_logs = suppress_logs

        if not self.suppress_logs:
            logging.info("Initializing the deep learning system")

        # Add your model initialization here
        if self.mode == "2D":
            self.model_2d = ResNet18(weights=None).cuda()
        elif self.mode == "vit":
            self.model_vit = DeiTSmall(pretrained=False).cuda()
        elif self.mode == "mobilenetv3L":
            self.model_mobilenet_v3_large = MobileNetLarge(pretrained=False).cuda()
        elif self.mode == "convnexttiny":
            self.model_convnexttiny = ConvNextTiny(pretrained=False).cuda()
        elif self.mode == "convnexttinyv2":
            self.model_convnexttinyv2 = ConvNextTinyV2(pretrained=False).cuda()
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
        elif config.MODE == "ensemble": # 2D model Ensemble
            if config.MODEL =="swin_tiny":
                base = SwinTiny(pretrained=False, num_classes=1)
            elif config.MODEL =="swin_base":
                base = SwinBase(pretrained=False, num_classes=1)
            elif config.MODEL == "deit_base":
                base = DeiTBase(num_classes=1, pretrained=False)
            elif config.MODEL =="resnet50":
                base = ResNet50()
            elif config.MODEL =="deit_small":
                base = DeiTSmall(num_classes=1, pretrained=False)
                model = EnsembleWrapper(base, num_images=config.NUM_IMAGES, num_outputs=1)
                model_name = f"LUNA25-deit_small_model_{i+1}"
                ckpt_path = os.path.join('final_ensemble_models', model_name, "best_metric_model.pth")
                #print(f"ckpt_path {ckpt_path}")
                ckpt = torch.load(ckpt_path,map_location=torch.device(DEVICE))

                if "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]
    
                ckpt = self.remove_prefix_to_state_dict(ckpt, prefix="module.")
                model.load_state_dict(ckpt)
                model.eval()
                self.ensemble_model = model.to(DEVICE)
            elif config.MODEL =="deit_small1":
                base_models = []
                for i in range (5):
                    base = DeiTSmall(num_classes=1, pretrained=False)
                    model = EnsembleWrapper(base, num_images=config.NUM_IMAGES, num_outputs=1)
                    model_name = f"LUNA25-deit_small_model_{i+1}"
                    ckpt_path = os.path.join('final_ensemble_models', model_name, "best_metric_model.pth")
                    print(f"ckpt_path {ckpt_path}")
                    ckpt = torch.load(ckpt_path,map_location=torch.device(DEVICE))

                    if "state_dict" in ckpt:
                        ckpt = ckpt["state_dict"]
        
                    ckpt = self.remove_prefix_to_state_dict(ckpt, prefix="module.")
                    model.load_state_dict(ckpt)
                    model.eval()
                    base_models.append(model)
                self.ensemble_model = Ensemble(base_models).to(DEVICE)
        elif config.MODE =="ensemble_deit_3d": # ensemble deit and 3d model
            deit_model_name = "LUNA25-deit_small_ensemble-ensemble-20250525-3"
            model_name_3d = "LUNA25-3D-baseline-3D-20250519-1"

            # First the DeiT model
            base = DeiTSmall(num_classes=1, pretrained=False)
            deit_model = EnsembleWrapper(base, num_images=config.NUM_IMAGES, num_outputs=1)

            ckpt_deit_model = torch.load(
                    os.path.join(
                        self.model_root,
                        deit_model_name,
                        "best_metric_model.pth",
                    ),
                        map_location=torch.device(DEVICE)
                    )
            if "state_dict" in ckpt_deit_model:
                ckpt_deit_model = ckpt_deit_model["state_dict"]
            ckpt_deit_model = self.remove_prefix_to_state_dict(ckpt_deit_model, prefix="module.")
            deit_model.load_state_dict(ckpt_deit_model)
            deit_model.eval()

            #now the 3d model
            model_3d = I3D(num_classes=1, pre_trained=False, input_channels=3).to(DEVICE)

            ckpt_3D_model = torch.load(
                    os.path.join(
                        self.model_root,
                        model_name_3d,
                        "best_metric_model.pth",
                    ),
                        map_location=torch.device(DEVICE)
                    )
            if "state_dict" in ckpt_3D_model:
                ckpt_3D_model = ckpt_3D_model["state_dict"]
            ckpt_3D_model = self.remove_prefix_to_state_dict(ckpt_3D_model, prefix="module.")
            model_3d.load_state_dict(ckpt_3D_model)
            model_3d.eval()

            # now ensemble them
            self.ensemble_model = Ensemble3d_deit(model_3d,  deit_model).to(DEVICE)

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

    def add_prefix_to_state_dict(self,state_dict, prefix="model."):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[prefix + k] = v
        return new_state_dict

    def remove_prefix_to_state_dict(self, state_dict, prefix="module."):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith(prefix):  
                k = k[len(prefix):]
            new_state_dict[k] = v
        return new_state_dict


    def _process_model(self, mode):

        if not self.suppress_logs:
            logging.info("Processing in " + mode)

        # Add your model with the correct output shape here
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
        elif mode =="ensemble":
            model =  self.ensemble_model
            output_shape = [1, self.size_px, self.size_px]
        elif mode == "ensemble_deit_3d":
            model = self.ensemble_model
            output_shape_deit = [1, self.size_px, self.size_px]
            output_shape_3d = [64, 64,64]
        else:
            output_shape = [self.size_px, self.size_px, self.size_px]
            model = self.model_3d


        
        if mode == "ensemble":
            print("mode is ensemble")
            nodules = []
            for _coord in self.coords: #N 
                image_patches = [] 
                for _ in range(config.NUM_IMAGES):  # B
                    patch = self.extract_patch(_coord, output_shape, mode=mode)  # C, H, W
                    image_patches.append(patch)
                nodules.append(image_patches)
            nodules = np.array(nodules)
            nodules = torch.from_numpy(nodules).to(DEVICE)
 
        elif mode == "ensemble_deit_3d" :
        # first deit nodules
            deit_nodules = []
            for _coord in self.coords: #N 
                image_patches = [] 
                for _ in range(config.NUM_IMAGES):  # B
                    patch = self.extract_patch(_coord, output_shape_deit, mode="ensemble")  # C, H, W
                    image_patches.append(patch)
                deit_nodules.append(image_patches)
            deit_nodules = np.array(deit_nodules)
            deit_nodules = torch.from_numpy(deit_nodules).to(DEVICE)
        # then 3d nodules
            nodules_3d = []
            for _coord in self.coords:
                patch = self.extract_patch(_coord, output_shape_3d, mode="3D")
                nodules_3d.append(patch)
            nodules_3d = np.array(nodules_3d)
            nodules_3d = torch.from_numpy(nodules_3d).to(DEVICE)

            logits = model(nodules_3d,deit_nodules)
            logits = logits.data.cpu().numpy()
            logits = np.array(logits)
            return logits  
            
        else:
            nodules = []
            for _coord in self.coords:
                patch = self.extract_patch(_coord, output_shape, mode=mode)
                nodules.append(patch)
            nodules = np.array(nodules)
            nodules = torch.from_numpy(nodules).to(DEVICE)
        print("model is", model)
        logits = model(nodules)
        logits = logits.data.cpu().numpy()
        logits = np.array(logits)
        return logits


    def predict(self):
        logits = self._process_model(self.mode)
        probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return probability, logits

if __name__ == "__main__":
    processor = MalignancyProcessor(mode = "ensemble_deit_3d")
