import torch
import torch.nn as nn
from monai.networks.nets import UNet

class ResNet3D(nn.Module):
    def __init__(self, num_classes=1, spatial_dims=3, in_channels=1):  # in_channels=1 for grayscale medical images
        super(ResNet3D, self).__init__()
        self.model = UNet(
                        spatial_dims=spatial_dims,
                        in_channels=num_classes,
                        out_channels=2,
                        channels=(16, 32, 64, 128, 256),
                        strides=(2, 2, 2, 2),
                        num_res_units=2,
                    )


    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    volume = torch.randn(4, 1, 64, 128, 128)  # [B, C, D, H, W]
    model = ResNet3D(num_classes=1)
    output = model(volume)
    print(f"ResNet3D output shape is: {output.shape}")  # torch.Size([4, 1])
