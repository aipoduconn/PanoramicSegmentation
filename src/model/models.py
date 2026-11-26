import torchvision.models as models 
from torch import nn
import torch
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights,deeplabv3_resnet50,DeepLabV3_ResNet50_Weights,DeepLabV3_ResNet101_Weights,deeplabv3_resnet101



class PanSegModel(nn.Module):
    def __init__(self,model_name,num_class):
        super(PanSegModel, self).__init__()
        
        self.model=None
        if model_name=="deeplabv3_resnet50":
            weights = DeepLabV3_ResNet50_Weights.DEFAULT 
            self.model = deeplabv3_resnet50(weights=weights)
            self.model.classifier[4]=torch.nn.Conv2d(256, num_class, kernel_size=(1, 1), stride=(1, 1))
            self.model.aux_classifier[4]=torch.nn.Conv2d(256, num_class, kernel_size=(1, 1), stride=(1, 1))
            
        elif model_name=="deeplabv3_resnet101":
            weights = DeepLabV3_ResNet101_Weights.DEFAULT 
            self.model = deeplabv3_resnet101(weights=weights)
            self.model.classifier[4]=torch.nn.Conv2d(256, num_class, kernel_size=(1, 1), stride=(1, 1))
            self.model.aux_classifier[4]=torch.nn.Conv2d(256, num_class, kernel_size=(1, 1), stride=(1, 1))
    
    def forward(self,x):
        return self.model(x)
        
            
            

