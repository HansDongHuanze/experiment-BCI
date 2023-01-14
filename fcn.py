import numpy as np
import pandas as pd
import torch as torch
import torch.nn.functional as F
import crypten
torch.set_printoptions(precision=16)
torch.set_default_tensor_type(torch.DoubleTensor)

'''

construct a class called FCN.

When initializing, need to include following parameters:
    input batch(optional)
    input size/shape
    output size

'''

class FCN(object):
    def __init__(self, input_batch, input_size, output_size):
        self.__input_batch = input_batch
        self.__input_size = input_size
        self.__output_size = output_size
        self.__weight = torch.from_numpy(np.random.rand(self.__input_batch, self.__input_size, self.__output_size) * 0.2 - 0.1).cuda()
        self.__bias = torch.from_numpy(np.random.rand(self.__input_batch, 1, self.__output_size) * 0.2 - 0.1).cuda()
    
    def forward(self, x_enc):
        self.x_enc = x_enc
        self.mat_res = x_enc.matmul(self.__weight) + self.__bias
        self.sig_res = self.mat_res.sigmoid()
        return self.sig_res
    
    def backpropagation(self, prev, client, rate):
        weight, bias, aft = client.fcn_grad(prev, rate, self.x_enc, self.__input_batch, self.__input_size, self.__output_size, self.__weight, self.mat_res, self.sig_res, self.__bias)
        self.__weight = weight
        self.__bias = bias
        return weight, aft
