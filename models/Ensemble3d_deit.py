import torch
import torch.nn as nn
#from models.SwinTiny import SwinTiny
from models.DeitSmall import DeiTSmall
from models.ensemble_model import EnsembleWrapper
from models.model_3d import I3D
class Ensemble3d_deit(nn.Module):
    def __init__(self, model_3D, deit_model, num_images= 9, num_outputs=1):
        """
        base_model_class: a callable that returns an instance of a model (e.g., SwinTiny or ResNet50)
        num_outputs: output size after the final FC layer
        """
        super().__init__()
        #self.base_models = nn.ModuleList(base_models)
        self.model_3D = model_3D
        self.deit_model = deit_model
        self.num_images = num_images

    def forward(self, x_3d, x_deit):
        """
        x: Tensor of shape (B, N, C, H, W), where N = num_images
        """
        #print("x_deit.shape", x_deit.shape)
        #print("x_3d", x_3d.shape)
        B, N, C, H, W = x_deit.shape
        
        assert N == self.num_images, f"Expected {self.num_images} images, got {N}"

        #x = x.view(B * N, C, H, W)

        with torch.no_grad():
            out_deit = self.deit_model(x_deit).view(-1) 
            out_3D = self.model_3D(x_3d).view(-1) 
        #print("out_deit", out_deit.shape, " valeus" ,out_deit )
        #print("out_3D", out_3D.shape, "values" , out_3D)

        # Average over the outputs
        final_output = torch.stack([out_deit,out_3D] , dim=0).mean(dim=0)  # (B, num_outputs)
        return final_output
if __name__ == "__main__":
    # Fake input: batch of 2 sets of 9 images / 1 3d image
    deit_tensor = torch.randn(2, 9, 3, 224, 224)
    tensor_3D = torch.rand(1, 1, 64, 64, 64)
    

    deit_model = DeiTSmall(num_classes=1)
    model_3D = I3D(num_classes=1, input_channels=3, pre_trained=True, freeze_bn=True)
    model = Ensemble3d_deit(model_3D, deit_model)
    output = model(tensor_3D, deit_tensor)
    print(output.shape)  # Should be [2, 1]
