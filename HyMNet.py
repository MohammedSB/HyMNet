import torch
import torch.nn as nn

from torchvision.models import densenet201, DenseNet201_Weights
    
def get_densenet201(device="cuda:0", freeze=False, with_mlp=False, outputs=1):
    model = densenet201(weights=DenseNet201_Weights.DEFAULT)
    
    if with_mlp:
        model.classifier = nn.Sequential(
            model.classifier,
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1000, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=64, out_features=outputs),
        )
    else:
        num_features = model.classifier.in_features
        features = list(model.classifier.children())[:-1]
        features.extend([nn.Linear(num_features, outputs)])
        model.classifier = nn.Sequential(*features)
        
    model = model.to(device)

    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False  
    
    return model

def get_demographic_model():
    demographic_model = nn.Sequential(
        nn.Linear(in_features=2, out_features=8),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=8, out_features=32),
    )
    return demographic_model

def get_fusion_model():
    fusion_model = nn.Sequential(
        nn.Linear(in_features=64, out_features=128),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(in_features=128, out_features=32),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(in_features=32, out_features=16),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(in_features=16, out_features=1),
    )
    return fusion_model


class HyMNet(torch.nn.Module):
    def __init__(self, device="cuda") -> None:
        super(HyMNet, self).__init__()
        self.image_model = get_densenet201(device=device, freeze=True, with_mlp=False, outputs=32)
        self.demographic_model = get_demographic_model()
        self.fusion_model = get_fusion_model()
    
    def forward(self, x):
        image_embeddings = x[0]
        demographic_embeddings = x[1]
        combined_embeddings = torch.cat((image_embeddings, demographic_embeddings), 1)
        output = get_fusion_model(combined_embeddings)
        return output