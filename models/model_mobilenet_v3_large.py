import torch
import torch.nn as nn
import timm  # assuming timm is already installed from .whl

class MobileNetLarge(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):  # Set pretrained=False for offline use
        super(MobileNetLarge, self).__init__()
        self.model = timm.create_model('mobilenetv3_large_100', pretrained=pretrained)

        # Replace the classification head
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    image = torch.randn(4, 3, 224, 224)  # Required input size for MobileNetV3L
    model = MobileNetLarge()
    output = model(image)
    print(output.shape)