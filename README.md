# [HyMNet: a Multimodal Deep Learning System for Hypertension Classification using Fundus Photographs and Cardiometabolic Risk Factors](https://pubmed.ncbi.nlm.nih.gov/39593740/)
HyMNet is a multimodal hypertension classification model that takes as input a fundus image, age and sex featrues to classify hypertension. With the aid of the readily available age and gender features, this model achieves a higher performance than models trained solely on fundus images.

## How can I use HyMNet?
HyMNet needs three inputs: (1) a fundus image, (2) age, and (3) sex (male/female). 

Below is sample code for how to get a classification from the model.

First clone the github repository and install the requirements.
```
pip install -r requirements.txt
```
To load the model see the sample below.
```python
from Utils.Utils import *
from Utils.Blacksmith import * 

from Utils.HyMNet import HyMNet
from timm.models.layers import trunc_normal_
import Utils.ViT as ViT 
import torch.nn as nn

tabular_model = nn.Sequential(
    nn.Linear(in_features=2, out_features=8),
    nn.LeakyReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=8, out_features=32),
)

image_model = get_retfound(<path_to_retfound_model>, image_size=512,
                          classes=8).requires_grad_(True)

fusion_model = nn.Sequential(
    nn.Linear(in_features=40, out_features=128),
    nn.LeakyReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(in_features=128, out_features=32),
    nn.LeakyReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(in_features=32, out_features=16),
    nn.LeakyReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(in_features=16, out_features=1),
)

model = HyMNet(image_model=image_model, tabular_model=tabular_model, fusion_model=fusion_model).to(device)
state_dict = torch.load("JointFusion_finetune.pth") # you have to download the model first
state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
model.load_state_dict(state_dict)

image = <load_the_image> # pytorch tensor
features = torch.tensor([50, 1]) # 50 year old, male

output = model(image, features)
prob = nn.Sigmoid()(output.squeeze())
prediction = prob > 0.5 # HTN if 1, 0 otherwise
```

If you want to use the function based implementation of the model (along with all the utilites in this repository), see reference code in `Utils.ipynb`, `Blacksmith.ipynb`, and `FusionModels.ipynb`.

## Pretrained HyMNet weights
To download the pretrained weights for HyMNet as well as all the other models in the study, click on the button below.

[[`Click me to download`]](https://drive.google.com/drive/folders/1wT8xKUZvXbubOOSPjD_S11jbqLebexuN?dmr=1&ec=wgc-drive-hero-goto)

## Reference
If you find this repository useful for your work, please cite using the citation below.

```
@Article{bioengineering11111080,
    AUTHOR = {Baharoon, Mohammed and Almatar, Hessa and Alduhayan, Reema and Aldebasi, Tariq and Alahmadi, Badr and Bokhari, Yahya and Alawad, Mohammed and Almazroa, Ahmed and Aljouie, Abdulrhman},
    TITLE = {HyMNet: A Multimodal Deep Learning System for Hypertension Prediction Using Fundus Images and Cardiometabolic Risk Factors},
    JOURNAL = {Bioengineering},
    VOLUME = {11},
    YEAR = {2024},
    NUMBER = {11},
    ARTICLE-NUMBER = {1080},
    URL = {https://www.mdpi.com/2306-5354/11/11/1080},
    PubMedID = {39593740},
    ISSN = {2306-5354},
    DOI = {10.3390/bioengineering11111080}
}
```