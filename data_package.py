import mne as mne
import numpy as np
import pandas as pd
import torch as torch
import time
from tqdm import tqdm
from openpyxl import load_workbook

from preprocessing import Preprocessing

import torch
import torch.nn as nn
 
from subDataset import subDataset
import torch.utils.data.dataloader as DataLoader

def get_dataset():
    labels = pd.read_csv('/home/dhz/bci-data/ds003478-download/participants.tsv', sep='\t', header=0)
    col = labels.columns
    col = col[[0, 1, 2, 3, 5, 6, 7, 8]]
    labels = labels.drop(col, axis=1)

    labels = np.array(labels)

    labels[37] = 0.
    
    train_data = []
    for i in range(30):
        train_data.append(Preprocessing(i + 1).getData())

    train_data = np.array(train_data)
    train_labels = labels[0 : 30]

    train_dataset = subDataset(train_data, train_labels)

    train_loader = DataLoader.DataLoader(train_dataset,batch_size= 1, shuffle = False)
    
    test_data = []
    for i in range(20):
        test_data.append(Preprocessing(30 + i + 1).getData())

    test_data = np.array(test_data)
    test_labels = labels[30 : 50]
    
    test_dataset = subDataset(test_data, test_labels)
    test_loader = DataLoader.DataLoader(test_dataset,batch_size= 1, shuffle = False)
    
    return train_loader, test_loader