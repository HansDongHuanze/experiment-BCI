import torch
from torch import nn
from vit_pytorch import ViT

torch.set_default_tensor_type(torch.DoubleTensor)



model = ViT(
    image_size = 512,    # image size
    patch_size = 32,     # patch size
    num_classes = 4,  # class num
    dim = 1024,          # dimention of position embedding
    depth = 6,           # layer of blocks in encoder and decoder
    heads = 16,          # num of head in multi-head
    mlp_dim = 8,
    dropout = 0.1, 
    emb_dropout = 0.1
)

model = model.cuda()                                                                                           

import sys
sys.path.append('..')
from data_extractor import *

Extractor('A01T')
Extractor('A02T')

raw_data = load_eeg('A01T')
labels, train_data = data_normalization(raw_data)
raw_data_eva = load_eeg('A02T')
labels_eva, eva_data = data_normalization(raw_data_eva)

import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    # 构造函数
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    # 返回数据集大小
    def __len__(self):
        return self.data_tensor.size(0)
    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]
    
dataset = MyDataset(train_data, labels)
BATCH_SIZE = 32
data_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

dataset_eva = MyDataset(eva_data, labels_eva)
data_loader_eva = DataLoader(dataset_eva, batch_size = BATCH_SIZE, shuffle = True)

LR = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), LR)

for epoch in range(100):
    model.train()
    running_loss = 0.0
    running_len = 0.
    acc_num = 0
    with tqdm(data_loader, unit = 'batch', ncols = 0, total = len(data_loader)) as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch(Training) [{epoch + 1}/{100}]")
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            pres = model(data)
            loss = criterion(pres, target)
            loss = loss.cuda()
            
            running_loss += loss.item()
            running_len += len(pres)
            
            loss.backward()
            optimizer.step()

            predicted = (pres == pres.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)
            
            tepoch.set_postfix(loss = running_loss)
            
    with tqdm(data_loader_eva, unit = 'batch', ncols = 0, total = len(data_loader_eva)) as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch(Evaluation) [{epoch + 1}/{100}]")
            data = data.cuda()
            target = target.cuda()
            pres = model(data)

            predicted = (pres == pres.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)
            
            acc_num = acc_num + torch.sum(torch.sum(predicted == target, dim = 1) / 4).item()
            
            accuracy = acc_num / running_len
            
            tepoch.set_postfix(acc = accuracy)