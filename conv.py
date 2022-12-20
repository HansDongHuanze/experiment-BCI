import numpy as np
import pandas as pd
import torch as torch
import torch.nn.functional as F
import crypten
torch.set_printoptions(precision=16)
torch.set_default_tensor_type(torch.DoubleTensor)

'''

construct a class called Conv.

When initializing, need to include following parameters:
    input batch(optional)
    input size/shape
    filter channel
    filter size/shape
    activation function
    pooling size/shape

'''

class Conv(object):
    def __init__(self, input_batch, input_channel, input_size, filter_channel, filter_size, act, pooling_size):
        self.__input_batch = input_batch
        self.__input_channel = input_channel
        self.__input_size = input_size
        self.__filter_channel = filter_channel
        self.__filter_size = filter_size
        self.__act = act
        self.__pooling_size = pooling_size
        self.__filt = torch.from_numpy(np.random.rand(self.__filter_channel, self.__input_channel, self.__filter_size, self.__filter_size) * 2. - 1.).cuda()
        self.__bias = torch.from_numpy(np.random.rand(self.__input_batch, self.__filter_channel, self.__input_size, self.__input_size) * 2. - 1.).cuda()
    
    def forward(self, x_enc):
        self.conv_res = x_enc.conv2d(self.__filt, padding = 1)
        self.relu_res = self.conv_res.relu()
        self.pooling_res = self.relu_res.max_pool2d(self.__pooling_size)
        return self.pooling_res
    
    def backpropagation(self, prev, client, rate):
        filt, dc = client.conv_grad(prev, self.pooling_res, self.relu_res, self.conv_res, rate, self.__input_batch, self.__input_channel, self.__filter_channel, self.__filter_size, self.__filt)
        self.__filt = filt
        return dc
