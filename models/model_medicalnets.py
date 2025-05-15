import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.conv_seg = nn.Sequential(
                                        nn.ConvTranspose3d(
                                        512 * block.expansion,
                                        32,
                                        2,
                                        stride=2
                                        ),
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(
                                        32,
                                        32,
                                        kernel_size=3,
                                        stride=(1, 1, 1),
                                        padding=(1, 1, 1),
                                        bias=False), 
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(
                                        32,
                                        num_seg_classes,
                                        kernel_size=1,
                                        stride=(1, 1, 1),
                                        bias=False) 
                                        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_seg(x)

        return x

def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

######################################################################
###### UP UNTIL HERE IS COPIED FROM THE MEDICALNET GITHUB REPO ######
###### SEE THEIR /models/resnet.py
######################################################################

class MedicalNetModel(nn.Module):
    def __init__(self, model_depth=50, num_classes=1, pretrained=False, 
                 pretrained_path="pretrained_resnets/resnet_152.pth",
                 input_D=64, input_H=128, input_W=128, input_channels=1):
        super(MedicalNetModel, self).__init__()
        
        self.config_depth = input_D
        self.config_height = input_H
        self.config_width = input_W
        self.input_channels = input_channels
        
        # Need to modify the first convolutional layer to accept the right number of input channels
        # So we initialize model architecture (backbone) based on depth
        self.backbone = None
        print(f"DEBUG: Creating backbone with depth {model_depth}")

        if model_depth == 10:
            self.backbone = resnet10(sample_input_D=self.config_depth, 
                                    sample_input_H=self.config_height,
                                    sample_input_W=self.config_width, 
                                    num_seg_classes=1)
        elif model_depth == 18:
            self.backbone = resnet18(sample_input_D=self.config_depth, 
                                    sample_input_H=self.config_height,
                                    sample_input_W=self.config_width, 
                                    num_seg_classes=1)
        elif model_depth == 34:
            self.backbone = resnet34(sample_input_D=self.config_depth, 
                                    sample_input_H=self.config_height,
                                    sample_input_W=self.config_width, 
                                    num_seg_classes=1)
        elif model_depth == 50:
            self.backbone = resnet50(sample_input_D=self.config_depth, 
                                    sample_input_H=self.config_height,
                                    sample_input_W=self.config_width, 
                                    num_seg_classes=1)
        elif model_depth == 101:
            self.backbone = resnet101(sample_input_D=self.config_depth, 
                                     sample_input_H=self.config_height,
                                     sample_input_W=self.config_width, 
                                     num_seg_classes=1)
        elif model_depth == 152:
            self.backbone = resnet152(sample_input_D=self.config_depth, 
                                     sample_input_H=self.config_height,
                                     sample_input_W=self.config_width, 
                                     num_seg_classes=1)
        elif model_depth == 200:
            self.backbone = resnet200(sample_input_D=self.config_depth, 
                                     sample_input_H=self.config_height,
                                     sample_input_W=self.config_width, 
                                     num_seg_classes=1)

        print(f"DEBUG: Backbone created successfully")
              
        # If we need a different number of input channels, replace the first conv layer
        if input_channels != 1:
            print(f"DEBUG: Replacing first conv layer to accept {input_channels} channels")
            self.backbone.conv1 = nn.Conv3d(
                input_channels,
                64,
                kernel_size=7,
                stride=(2, 2, 2),
                padding=(3, 3, 3),
                bias=False
            )
        
        # Get the expansion factor from the last layer based on model depth
        # Has to do with BasicBlock vs Bottleneck classes
        expansion = 1
        if model_depth >= 50:
            print("DEBUG: Changing expansion factor from 1 to 4")
            expansion = 4
            
        # Replace the final layer with a classifier for binary classification
        # The exact features depend on the structure
        num_features = 512 * expansion
        print(f"DEBUG: Creating classifier with {num_features} input features")

        self.classifier = nn.Linear(num_features, num_classes)
        
        # Load pretrained weights if specified
        if pretrained:
            print(f"DEBUG: Loading pretrained weights")
            self._load_pretrained_weights(pretrained_path)

            
        print(f"DEBUG: MedicalNetModel initialization complete")
    
    def _load_pretrained_weights(self, pretrained_path):
        """
        Load pretrained weights from MedicalNet
        """
        print(f"Loading pretrained weights from {pretrained_path}")
        
        try:
            # Load the state dict
            pretrained_dict = torch.load(pretrained_path)
            
            # Filter out unnecessary keys
            if 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']
            
            # Get the model's current state dict
            model_dict = self.backbone.state_dict()
            
            # Create a set of all model keys for tracking
            all_model_keys = set(model_dict.keys())
            loaded_keys = set()
            
            # Map pretrained keys to model keys
            mapped_dict = {}
            
            # Remove 'module.' prefix from pretrained keys (common in DataParallel models)
            for k, v in pretrained_dict.items():
                if k.startswith('module.'):
                    new_key = k[7:]  # Remove 'module.' prefix
                else:
                    new_key = k
                    
                # Check if key exists in model
                if new_key in model_dict:
                    # Check if shapes match
                    if model_dict[new_key].shape == v.shape:
                        mapped_dict[new_key] = v
                        loaded_keys.add(new_key)
                    else:
                        print(f"DEBUG: Shape mismatch for {new_key}: pretrained {v.shape} vs model {model_dict[new_key].shape}")
            
            # # Identify missing keys # Note, the 18 layers that it does not load from for segmentation (start with conv_seg.)
            # missing_keys = all_model_keys - loaded_keys
            # print(f"DEBUG: Missing keys ({len(missing_keys)}):")
            # for key in missing_keys:
            #     print(f"  {key}")
            
            # Update the model's state dict with pretrained weights
            model_dict.update(mapped_dict)
            self.backbone.load_state_dict(model_dict)
            
            print(f"Successfully loaded pretrained weights from {pretrained_path}")
            print(f"Loaded {len(mapped_dict)}/{len(model_dict)} layers from pretrained model")
            
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print(f"Failed path was: '{pretrained_path}'")
            print("Continuing with randomly initialized weights")

    def forward(self, x):
        # print(f"DEBUG: Forward pass - Input shape: {x.shape}")
        
        try:
            # Extract features directly from backbone layers
            # Skip the segmentation head (conv_seg)
            # First process input
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            # Then extract features with the resnet layers
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            # print(f"DEBUG: After backbone layers shape: {x.shape}")
            
            # Global average pooling over the spatial dimensions (H, W, D)
            features = torch.mean(x, dim=(2, 3, 4))
            # print(f"DEBUG: After pooling shape: {features.shape}")
            
            # Classification
            # print(f"DEBUG: Applying classifier")
            output = self.classifier(features)
            # print(f"DEBUG: Output shape: {output.shape}")
            
            return output
        except Exception as e:
            print(f"DEBUG: Error in forward pass: {e}")
            import traceback
            traceback.print_exc()
            raise e