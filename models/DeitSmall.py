import torch
import torch.nn as nn
import timm  # assuming timm is already installed from .whl

class DeiTSmall(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):  # Set pretrained=False for offline use
        super(DeiTSmall, self).__init__()
        self.model = timm.create_model('deit_small_patch16_224', pretrained=pretrained)

        # Replace the classification head
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Example test
if __name__ == "__main__":
    image = torch.randn(4, 3, 224, 224)  # Required input size for DeiT
    model = DeiTSmall()
    output = model(image)
    print(output.shape)