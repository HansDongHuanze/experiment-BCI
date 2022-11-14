import torch
from torch import nn
from vit_pytorch import ViT

torch.set_default_tensor_type(torch.DoubleTensor)

BATCH_SIZE = 32

model = ViT(
    image_size = 1024,    # 图像大小
    patch_size = 128,     # patch大小（分块的大小）
    num_classes = 8,  # imagenet数据集1000分类
    dim = 512,          # position embedding的维度
    depth = 6,           # encoder和decoder中block层数是6
    heads = 16,          # multi-head中head的数量为16
    mlp_dim = 8,
    dropout = 0.1,       # 
    emb_dropout = 0.1
)

model = model.cuda()

import sys
sys.path.append('..')
from data_extractor_v2 import *

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
    
def training_and_eva(train_data, labels, eva_data, labels_eva):
    dataset = MyDataset(train_data, labels)
    data_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

    dataset_eva = MyDataset(eva_data, labels_eva)
    data_loader_eva = DataLoader(dataset_eva, batch_size = BATCH_SIZE, shuffle = True)
    
    LR = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), LR)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        running_len = 0.
        acc_num = 0
        
        with tqdm(data_loader, unit = 'batch', ncols = 0, total = len(data_loader)) as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch(Training) [{epoch + 1}/{10}]")
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
            
        with torch.autograd.no_grad():
            model.eval()
            with tqdm(data_loader_eva, unit = 'batch', ncols = 0, total = len(data_loader_eva)) as tepoch:
                for data, target in tepoch:
                    tepoch.set_description(f"Epoch(Evaluation) [{epoch + 1}/{10}]")
                    data = data.cuda()
                    target = target.cuda()
                    pres = model(data)

                    predicted = (pres == pres.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)
            
                    acc_num = acc_num + torch.sum(torch.sum(predicted == target, dim = 1) / 8).item()
            
                    accuracy = acc_num / running_len
            
                    tepoch.set_postfix(acc = accuracy)
                
if __name__ == '__main__':
    database_training = [[], [], [], [],
           [], [], [], []]

    database_eva = [[], [], [], [],
           [], [], [], []]

    for i in range(1, 109, 2):
        for line in database_training:
            line.clear()
        for line in database_eva:
            line.clear()
        extractor_training = Extractor("S{0:0=3d}".format(i), database_training)
        extractor_eva = Extractor("S{0:0=3d}".format(i + 1), database_eva)
        labels, train_data = data_normalization(database_training)
        labels_eva, eva_data = data_normalization(database_eva)
        training_and_eva(train_data, labels, eva_data, labels_eva)