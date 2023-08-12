# import
from torch import nn
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchmetrics.functional import auc
import io
import matplotlib.pyplot as plt
import time
import copy
from glob import glob
from tqdm import tqdm
import warnings
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score,recall_score,roc_auc_score,roc_curve
warnings.simplefilter('ignore')

# training model function
def train_model(model, criterion, optimizer, scheduler, name, num_epochs=10):
    try:
        os.mkdir(f'./modelPerformance/{name}')
    except:
        print('Dosya var')

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    # todo

# main function
if __name__ == '__main__':
    # dictionary of models
    modeller = {
            'resnext':models.resnext50_32x4d(pretrained=True)
    }
    try:
        os.mkdir(f'./modelPerformance')
    except:
        print('Dosya var')

    # vgg included
    for name,model in modeller.items():
        model_ft = model

        if 'vgg' in name:
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, len(class_names)), nn.Softmax())
        else:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, len(class_names)), nn.Softmax())

        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=0.9)

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        # training
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, name=name, num_epochs=EPOCH)







