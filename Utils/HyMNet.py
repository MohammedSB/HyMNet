import torch
import torch.nn as nn
import torch.nn.functional as F

class HyMNet(nn.Module):
    def __init__(self, image_model=None, tabular_model=None, fusion_model=None):
        super(HyMNet, self).__init__()
        self.image_model = image_model
        self.tabular_model = tabular_model
        self.fusion_model = fusion_model
    
    def forward(self, img, features=None):
        
        # forward tree
        if self.fusion_model and self.image_model == None and self.tabular_model == None: # Train just fusion
            output = self.fusion_model(features)
        elif self.fusion_model: # Train all paths
            image_features = self.image_model(img)
            tabular_features = self.tabular_model(features)
            combined_features = torch.cat((image_features, tabular_features), 1)
            output = self.fusion_model(combined_features)
        elif self.image_model:
            output = self.image_model(img)
        else:
            output = self.tabular_model(features)
            
        return output
    
