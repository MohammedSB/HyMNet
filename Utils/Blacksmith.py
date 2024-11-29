from .Utils import *
from .Utils import classify

import torch
import torch.nn
import torch.nn.functional as F
from tqdm import tqdm

import sklearn
    
def one_epoch(dataloader, model, device, criterion, optimizer=None, scheduler=None, show_output=True):
    
    metrics = dict()
    
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    count, total_loss, correct = 0, 0, 0
    preds, probs, targets = [], [], []

    with torch.set_grad_enabled(training):
        for sample in tqdm(dataloader, desc="Batch in Progress", ascii=False, ncols = 100, disable=not show_output):
            
            img, features, target = sample["image"], sample["features"], sample["label"]
            img, features, target = img.to(device).float(), features.to(device).float(),\
                                    target.to(device).float()
            
            output = model(img, features) 
        
            target = target.unsqueeze(dim=-1)
                
            loss = criterion(output, target)

            # backward            
            if training:
                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()
                
            if scheduler:
                scheduler.step()

            # bookkeeping
            sig = torch.nn.Sigmoid()  
            total_loss += loss.item() * len(img)
            prob = sig(output.detach())
            pred = (prob > 0.5).float()

            pred = pred.cpu()
            target = target.cpu()  # Move target to CPU before comparison

            correct += (pred == target).sum().item()

            probs.append(prob.cpu())
            preds.append(pred)
            targets.append(target)
            
    probs = torch.cat(probs)
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    metrics["Loss"] = total_loss
    metrics["Average Loss"] = total_loss / len(dataloader.dataset)
    metrics["Correct"] = correct
    metrics["Accuracy"] = (correct / len(dataloader.dataset)) * 100
    metrics["Size"] = len(dataloader.dataset)

    # Precision, Recall, F1 Score, and AUROC
    targets_1d = targets.squeeze(1).numpy()
    preds_1d = preds.squeeze(1).numpy()
    probs_1d = probs.squeeze(1).numpy()
    
    precision, recall, thresholds = precision_recall_curve(targets_1d, preds_1d)
    metrics["PR"] = sklearn.metrics.auc(recall, precision)
    
    metrics["Precision"] = precision_score(targets_1d, preds_1d, zero_division=1)
    metrics["Recall"] = recall_score(targets_1d, preds_1d, zero_division=1)
    metrics["F1 Score"] = f1_score(targets_1d, preds_1d)
    metrics["AUROC"] = roc_auc_score(targets_1d, probs_1d)

    # y_prob, y_true, and y_pred
    metrics["y_prob"] = probs
    metrics["y_true"] = targets
    metrics["y_pred"] = preds
    
    return metrics
    
def train_val(epochs, model, criterion, optimizer, train_loader, val_loader, device, scheduler, early_stop=None, show_output=True, save_model=False):
    print("Beginning Training: \n")
    metrics_train = dict()
    metrics_val = dict()
    metrics = dict()    
    best_auroc = None
    best_model = None
    
    for epoch in range(1, epochs + 1):
        
        print(f'Epoch {epoch}/{epochs}')
        metrics = one_epoch(train_loader, model, device, criterion, optimizer, scheduler, show_output=show_output)
        print("Train Set:")
        show_metrics(metrics)

        metrics_train[epoch] = metrics
        
        metrics = one_epoch(val_loader, model, device, criterion, show_output=show_output)
        print("Validation Set:")
        show_metrics(metrics)
        
        if best_auroc == None or metrics["AUROC"] > best_auroc:
            best_model = model.state_dict()

        metrics_val[epoch] = metrics
                
    metrics = [metrics_train, metrics_val]
    
    if save_model:
        return metrics, best_model
    return metrics

def train(epochs, model, criterion, optimizer, train_loader, device):
    print("Beginning Training: \n")
    metrics_train = dict()
    metrics = dict()    
    
    for epoch in range(1, epochs + 1):
        
        print(f'Epoch {epoch}/{epochs}')
        metrics = one_epoch(train_loader, model, device, criterion, optimizer)
        print("Train Set:")
        show_metrics(metrics)

        metrics_train[epoch] = metrics
        
    return metrics_train


def test(model, criterion, test_loader, device, show_output=True):
    
    metrics = one_epoch(test_loader, model, device, criterion, show_output=show_output)
    if show_output:
        show_metrics(metrics)  
    
    return metrics

