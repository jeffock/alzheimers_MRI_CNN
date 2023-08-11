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
    pass

# main function
if __name__ == '__main__':
    pass







