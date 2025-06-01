import torch
import torch.nn as nn
#from models.SwinTiny import SwinTiny
from models.DeitSmall import DeiTSmall
from models.ensemble_model import EnsembleWrapper
class Ensemble(nn.Module):
    def __init__(self, base_models, num_images= 9, num_outputs=1):
        """
        base_model_class: a callable that returns an instance of a model (e.g., SwinTiny or ResNet50)
        num_outputs: output size after the final FC layer
        """
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.num_images = num_images


    def forward(self, x):
        """
        x: Tensor of shape (B, N, C, H, W), where N = num_images
        """
        B, N, C, H, W = x.shape
        assert N == self.num_images, f"Expected {self.num_images} images, got {N}"

        #x = x.view(B * N, C, H, W)

        all_outputs = []
        for model in self.base_models:
            with torch.no_grad():  # inference-only
                out = model(x)  # (B, num_outputs), assume num_outputs=1
                print(f"out is {out}")
                #out = out.view(B, N, -1).mean(dim=1)  # (B, num_outputs)
                all_outputs.append(out)

        # Average over all models' outputs
        final_output = torch.stack(all_outputs, dim=0).mean(dim=0)  # (B, num_outputs)
        return final_output
if __name__ == "__main__":
    # Fake input: batch of 2 sets of 9 images
    input_tensor = torch.randn(2, 9, 3, 224, 224)

    basemodel1 = DeiTSmall(num_classes=1)
    basemodel2 = DeiTSmall(num_classes=1)
    basemodel3 = DeiTSmall(num_classes=1)
    basemodel4 = DeiTSmall(num_classes=1)
    basemodel5 = DeiTSmall(num_classes=1)
    basemodel6 = DeiTSmall(num_classes=1)

    model = Ensemble([basemodel1, basemodel2, basemodel3, basemodel4, basemodel5, basemodel6])
    output = model(input_tensor)
    print(output.shape)  # Should be [2, 1]
