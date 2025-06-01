import torch
import torch.nn as nn
from models.SwinTiny import SwinTiny
from models.DeitSmall import DeiTSmall
class EnsembleWrapper(nn.Module):
    def __init__(self, base_model, num_images= 9, num_outputs=1):
        """
        base_model_class: a callable that returns an instance of a model (e.g., SwinTiny or ResNet50)
        num_outputs: output size after the final FC layer
        """
        super().__init__()
        self.base_model = base_model  # shared weights
        self.num_images = num_images
        self.fc = nn.Linear(self.num_images * num_outputs, num_outputs)  # combine 9 outputs

    def forward(self, x):
        """
        x: Tensor of shape (B, 9, C, H, W)
        """
        #print(x.shape)
        B, N, C, H, W = x.shape
        #print(B,N,C,H,W)
        assert N == self.num_images, f"Input must have {self.num_images} images per sample"

        # Flatten into 9 separate inputs to the base model
        x = x.view(B * N, C, H, W)
        out = self.base_model(x)  # shape: (B*9, num_outputs)

        # Reshape and combine
        out = out.view(B, N * out.shape[-1])  # shape: (B, 9*num_outputs)
        out = self.fc(out)  # shape: (B, num_outputs)

        return out
if __name__ == "__main__":
    # Fake input: batch of 2 sets of 9 images
    input_tensor = torch.randn(2, 9, 3, 224, 224)

    basemodel = SwinTiny()

    #basemodel = DeiTSmall(num_classes=1)

    model = EnsembleWrapper(basemodel, num_outputs=1)
    output = model(input_tensor)
    print(output.shape)  # Should be [2, 1]
