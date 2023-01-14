from conv import *
from fcn import *
from client import *

import numpy as np
import pandas as pd
import torch as torch
import torch.nn.functional as F
import crypten

torch.set_printoptions(precision=16)
torch.set_default_tensor_type(torch.DoubleTensor)

from tqdm import tqdm, trange
import time
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

input_batch = 3
input_channel = 10
input_size = 128
filter_channel = 10
filter_size = 3
act = 'ReLU'
pooling_size = 2

conv_list = []
conv_list.append(Conv(input_batch, 3, input_size, filter_channel, filter_size, act, pooling_size))
input_size = (int)(input_size / 2)

while input_size > 1:
    conv_list.append(Conv(input_batch, input_channel, input_size, filter_channel, filter_size, act, pooling_size))
    input_size = (int)(input_size / 2)

fcn_list = []
output_size = 16

fcn1 = FCN(input_batch, filter_channel, output_size)
fcn_list.append(fcn1)

while output_size > 2:
    fcn_list.append(FCN(input_batch, output_size, (int)(output_size / 2)))
    output_size = (int)(output_size / 2)

def train(data):
    hidden_list = []
    hidden_list.append(conv_list[0].forward(data))
    
    for i in range(1, len(conv_list)):
        temp = conv_list[i].forward(hidden_list[-1])
        hidden_list.append(temp)

    hidden_list[-1] = hidden_list[-1].reshape(input_batch, 1, filter_channel)
        
    for fcn_layer in fcn_list:
        temp = fcn_layer.forward(hidden_list[-1])
        hidden_list.append(temp)
        
    return hidden_list[-1]

if __name__ == '__main__':
    clients = []
    for i in range(25):
        clients.append(Client(i + 1))
    
    for epoch in range(100):
        losses = 0.
        step = 1
        accuracy =[]
        
        loop = tqdm(range(25))
        loop.set_description(f'Epoch [{epoch + 1}/{100}]')
        
        for i in loop:
            client = clients[i]

            data = client.get_data()

            prediction = train(data)

            temp = client

            loss = temp.loss(prediction)

            learning_rate = 1e-2
            prev = client.loss_grad()

            for i in range(len(fcn_list)):
                weight, prev = fcn_list[len(fcn_list) - 1 - i].backpropagation(prev, client, learning_rate)

            prev = prev.reshape(input_batch, filter_channel, 1, 1)

            for i in range(len(conv_list)):
                weight, prev = conv_list[len(conv_list) - 1 - i].backpropagation(prev, client, learning_rate)

            losses += loss.item()

            
            loop.set_postfix(loss = losses / step)
            step += 1