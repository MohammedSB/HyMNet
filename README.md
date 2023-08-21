# HyMNet: a Multimodal Deep Learning System for Hypertension Classification using Fundus Photographs and Cardiometabolic Risk Factors
HyMNet is a multimodal hypertension classification model that takes as input a fundus image, age and sex featrues to classify hypertension. With the aid of the readily available age and gender features, this model achieves a higher performance than models trained solely on fundus images.

## How can I use HyMNet
HyMNet needs three inputs: (1) a fundus image, (2) age, and (3) sex (male/female). The input of the model needs to be in the shape of `[fundus image, [age, sex]]`. The model outputs a prediction logit that needs to be passed through a sigmoid function before classification. For optimial performance, the inputs needs to be preprocessed according to the pipeline detailed in the paper. See function `get_htn_dataframe` in `Utils.ipynb` and the fundus image transformation in `FundusModel.ipynb`.

Below is sample code for how to get a classification from the model.

First clone the github repository. Then:
```python
import torch
from HyMNet import HyMNet

model = HyMNet()
model.load_state_dict(torch.load("<path_to_model_weights>"))

prediction_logit = model(x)
prediction = torch.nn.Sigmoid()(prediction_logit)
```

If you want to use the function based implementation of the model (along with all the utilites in this repository), see reference code in `Utils.ipynb`, `Blacksmith.ipynb`, and `FusionModels`.

## Download pretrained HyMNet weights
[[`Click me to download`]](https://drive.google.com/uc?export=download&id=1Np3QJIAKS1LOo_RwL6lLFWRdyC4ItIB1)

This is my work done in collaboration with Dr. Abdulrhman Aljouie as an intern at King Abdullah International Medical Research Center (KAIMRC).
