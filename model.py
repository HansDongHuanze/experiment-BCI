from conv import *
from fcn import *
from client import *

import numpy as np
import pandas as pd
import torch as torch
import torch.nn.functional as F
import crypten

import matplotlib.pyplot as plt

torch.set_printoptions(precision=16)
torch.set_default_tensor_type(torch.DoubleTensor)

from tqdm import tqdm, trange
import time
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

input_batch = 2
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

def accuracy_f(predictions, client):
    MSE = client.mse_cal(predictions)
    return MSE
    
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

def start(ite):
    clients = []
    test_clients = []
    accs = []
    loss_hist = []
    for i in range(30):
        client = Client(i + 1)
        clients.append(client)
        
    for i in range(20):
        client = Client(30 + i + 1)
        test_clients.append(client)
    
    for epoch in range(ite):
        losses = 0.
        step = 0
        accuracy = 0.
        mses = 0.
        
        loop = tqdm(range(len(clients)))
        loop.set_description(f'Epoch [{epoch + 1}/{ite}]')
        
        for i in loop:
            step += 1
            client = clients[i]

            data = client.get_data()

            prediction = train(data)
            
            if client.acc(prediction):
                accuracy += 1.

            temp = client

            loss = temp.loss(prediction)

            learning_rate = 1e-2
            
            prev = client.loss_grad()

            for i in range(len(fcn_list)):
                weight, prev = fcn_list[len(fcn_list) - 1 - i].backpropagation(prev, client, learning_rate)

            prev = prev.reshape(input_batch, filter_channel, 1, 1)

            for i in range(len(conv_list)):
                weight, prev = conv_list[len(conv_list) - 1 - i].backpropagation(prev, client, learning_rate)

            losses = losses + loss.item()
            if step % 30 == 0:
                for c in test_clients:
                    p = train(c.get_data())
                    mses += accuracy_f(p, c)
                loop.set_postfix(loss = losses / step, mse = mses / 20)
            
        accs.append(mses / step)
        loss_hist.append(losses / step)
        
    plt.figure(figsize=(20, 10), dpi=100)
    iterate = range(1, ite + 1)
    
    plt.plot(iterate, accs, c='red', label="accuracy")
    plt.plot(iterate, loss_hist, c='green', linestyle='--', label="loss")
    plt.scatter(iterate, accs, c='red')
    plt.scatter(iterate, loss_hist, c='green')
    plt.legend(loc='best')
    plt.xlabel("iteration", fontdict={'size': 16})
    plt.ylabel("details", fontdict={'size': 16})
    plt.title("accuracy and average loss", fontdict={'size': 20})
    plt.show()
        
if __name__ == '__main__':
    start(100)