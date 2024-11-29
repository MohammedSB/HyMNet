# import cv2
from PIL import Image
import matplotlib.pyplot as plt
from inspect import getmembers, isfunction 
import argparse
import math
import random
import re

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.transform import resize

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, cross_validate, StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, precision_recall_curve, roc_auc_score
from sklearn.utils import shuffle

import torch
import torch.nn as nn
from torch import Tensor, optim
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torch.nn.modules.loss import BCEWithLogitsLoss
from torchvision.datasets import ImageFolder
import torchvision.models

import itertools

import torchvision

import xgboost as xgb

import json
import os
import cv2
import sys
from os import listdir
from os.path import splitext
from tqdm import tqdm
from pathlib import Path
import logging

import warnings
from timm.models.layers import trunc_normal_
import Utils.ViT as vit 


def set_seed(seed = 0):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def showimage(myimage, figsize=[10,10]):
    if (myimage.ndim>2):  #This only applies to RGB or RGBA images (e.g. not to Black and White images)
        myimage = myimage[:,:,::-1] #OpenCV follows BGR order, while matplotlib likely follows RGB order
         
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(myimage, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def visualize_metrics(metrics):
    lenEpochs = len(metrics[0]) 
    epochs = [i for i in range(lenEpochs)]
    train = [( metrics[0][i]["Average Loss"]  ) for i in metrics[0]]
    val = [( metrics[1][i]["Average Loss"]  ) for i in metrics[0]]
    plt.plot(epochs, train, label = "Train")
    plt.plot(epochs, val, label = "Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend() 
    
def show_metrics(metrics):
   
    print(f'\
    Average Loss:  {metrics["Average Loss"]:>8f},\
    Accuracy: {metrics["Accuracy"]:>0.2f}%,\
    Correct Counter: {metrics["Correct"]}/{metrics["Size"]},\
    F1 Score:{metrics["F1 Score"]: 0.2f},\
    Precision:{metrics["Precision"]: 0.2f},\
    Recall: {metrics["Recall"]: 0.2f},\
    PR: {metrics["PR"]: 0.2f},\
    AUROC: {metrics["AUROC"]: 0.2f}\n'
    )
    

def binary_logits_prediction(logit):
    sig = 1/(1 + np.exp(-logit))
    return classify(sig)
    
def classify(sig):
    return int(sig>=0.5)


def get_processed_dataframe(path, normalize=False, standardize=True):    
    
    # Hypertension
    htn = pd.read_csv(path["HTNPath"] + r"/HTN_Project_HTN_CSV.csv")

    # Clean Hypertension
    htn.columns = htn.columns.str.strip()
    htn["Eye"] = htn["Eye"].str.strip()

    # Add "File Path" column
    htn["Image Path"] = path["HTNPath"] + os.sep + htn['ImageName']

    # non-Hypertension
    non_htn = pd.read_csv(path["NonHTNPath"] + r"/HTN_Project_Normal_CSV.csv")
    
    # Clean non-Hypertension
    non_htn.columns = non_htn.columns.str.strip()
    non_htn["Eye"] = non_htn["Eye"].str.strip()

    # Add "Image Path" column
    non_htn["Image Path"] = path["NonHTNPath"] + os.sep + non_htn['ImageName']
    
    # Add the two df's
    df = pd.concat([htn, non_htn]) 
    
    # Fix issue with age
    df.loc[df["MRN"] == 2982450, "Age"] = 63

    # Clean again
    df = df.dropna()
    df["Gender"] = df["Gender"].str.strip()
    df.reset_index(inplace=True, drop=True)
    
    # One hot encoding for Gender 
    df["Gender"] = (df["Gender"]=="M").astype("int64")
        
    if normalize:
        df["Age"]=(df["Age"]-df["Age"].min())/(df["Age"].max()-df["Age"].min())
    if standardize:
        df["Age"] = (df["Age"] - df["Age"].mean())/df["Age"].std(ddof=0)
    
    return df

def build_tabular_dataset(model, dataset, device, method=None):
    
    x, y = [], []
    
    for sample in dataset:

        img, features, target = sample["image"], sample["features"], sample["label"]
        img, features, target = img.to(device).float(), features.to(device).float(), \
                                        target.to(device).float()

        img, features = img.unsqueeze(0), features.unsqueeze(0)
    
        if method == "late_fusion" or method == "lf":
            output_image = model.image_model(img)
            combined = torch.cat((output_image, features), 1)
        elif method == "voting_fusion" or method == "vf":
            
            # image output
            output_image = model.image_model(img)
            
            # tabular output
            output_tabular = model.tabular_model(features)
            
            # fusion output
            output_fusion = model.fusion_model(img, features)
            
            combined = torch.cat((output_fusion, output_image, output_tabular), 1)

        x += combined.tolist()
        y += target.unsqueeze(0).tolist()
        
    return np.array(x), np.array(y)

def get_mrns(df):
    
    df = df.drop_duplicates(subset='MRN', keep='first')
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=0)

    # Create a new column in df that represents the combined conditions of 'HTN' and 'DM'
    df['HTN_DM'] = df['HTN'].astype(str) + df['DM'].astype(str)

    for train_index, temp_index in sss.split(df, df['HTN_DM']):
        df_train = df.iloc[train_index]
        temp = df.iloc[temp_index]

    # split into a 50% validation set and a 50% test set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    for val_index, test_index in sss.split(temp, temp['HTN_DM']):
        df_val = temp.iloc[val_index]
        df_test = temp.iloc[test_index]

    # Now, df_train, df_val, and df_test are your training, validation, and test sets
    train_mrn = df_train["MRN"].to_list()
    val_mrn = df_val["MRN"].to_list()
    test_mrn = df_test["MRN"].to_list()
    
    return train_mrn, val_mrn, test_mrn
    
class HypertensionDataset(Dataset):
    def __init__(self, path, split="train", train_transform=None, test_transform=None, tabular_only=False):
        
        self.path = path
        self.split = split
        self.tabular_only = tabular_only
        
        self.train_transform = train_transform
        self.test_transform = test_transform
         
        self.df = get_processed_dataframe(path)
        
        self.train_mrn, self.val_mrn, self.test_mrn = get_mrns(self.df)
        if self.split == "train":
            self.df = self.df[self.df['MRN'].isin(self.train_mrn)]
        elif self.split == "val":
            self.df = self.df[self.df['MRN'].isin(self.val_mrn)]
        elif self.split == "test":
            self.df = self.df[self.df['MRN'].isin(self.test_mrn)]
        else:
            print(f"Wrong split {self.split}")
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        
        # Get image
        if not self.tabular_only:
            image_path = self.df["Image Path"].iloc[idx]
            img = Image.open(image_path)
        
            if self.train_transform is not None:
                img = self.train_transform(img)
            elif self.test_transform is not None:
                img = self.test_transform(img)
        else:
            img = torch.zeros(1)
       
        sample = {
            "image": img,
            "features": torch.from_numpy(self.df.iloc[idx][["Age", "Gender"]].values.astype("float64")),
            "label": torch.tensor(self.df["HTN"].iloc[idx]),
            "dm": self.df["DM"].iloc[idx]
        }
        
        return sample
    
class InputOutputDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        sample = {
            "image": torch.tensor([]),
            "features": torch.tensor(self.x[idx]),
            "label": torch.tensor(self.y[idx]),
        }
        return sample
                

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
    
def get_retfound(checkpoint="/home/baharoon/HTN/RETFound_cfp_weights.pth", image_size=512, classes=1, global_pool=True):
    # call the model
    model = vit.__dict__['vit_large_patch16'](
        num_classes=classes,
        drop_path_rate=0.2,
        global_pool=global_pool,
        img_size=image_size,
        
    )
    
    # load RETFound weights
    checkpoint = torch.load(checkpoint, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)

#     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)

    return model