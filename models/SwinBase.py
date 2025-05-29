import torch
import torch.nn as nn
import timm  # assuming timm is already installed from .whl

class SwinBase(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):  # Set pretrained=False for offline use
        super(SwinBase, self).__init__()
        self.model = timm.create_model("swin_base_patch4_window7_224", 
                                       pretrained=pretrained, 
                                       num_classes = num_classes)
    def forward(self, x):
        return self.model(x)

# Example test
if __name__ == "__main__":
    image = torch.randn(4, 3, 224, 224)  # Required input size for Swin
    model = SwinBase()
    output = model(image)
    print(output.shape)