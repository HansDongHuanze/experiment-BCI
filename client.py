import numpy as np
import pandas as pd
import torch as torch
import torch.nn.functional as F
import crypten
from data import DataExtract
torch.set_printoptions(precision=16)
torch.set_default_tensor_type(torch.DoubleTensor)

class Client(object):
    def __init__(self, num):
        self.num = num
        crypten.init()
        self.__plain_dataset, self.__labels = DataExtract(self.num).extract()
        self.__labels = self.__labels * 0.01
        self.crypt_dataset = crypten.cryptensor(self.__plain_dataset).cuda()
        
    def get_data(self):
        return self.crypt_dataset
    
    def conv_grad(self, prev, p_res, r_res, c_res, rate, batch, input_channel, input_size, filter_channel, filter_size, filt, bias):
        p_plain = p_res.get_plain_text()
        pool_raw = prev.mul(p_plain.to(torch.double)).repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        
        relu_plain = r_res.get_plain_text()
        dp_dr = torch.eq(p_plain.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3), relu_plain, out=torch.tensor([1.,0.]).cuda())
        
        conv_plain = c_res.get_plain_text()
        dr_dc = torch.eq(relu_plain, conv_plain, out=torch.tensor([1.,0.]).cuda())
        
        dp_dc = prev.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).mul(dp_dr).matmul(dr_dc)
        x_pad = self.crypt_dataset.pad([1, 1, 1, 1]).get_plain_text()

        for bat in range(batch):
            for chan in range(filter_channel):
                for line in range(filter_size):
                    for col in range(filter_size):
                        # filter update start
                        window = x_pad[bat, :, line : (line + input_size), col : (col + input_size)]
                        temp = 0.
                        for block in window:
                            # temp = temp + 10e-04 * dp_dc.mul(block.to(torch.double)).sum() / pow((batch * filter_channel * filter_size * filter_size), input_channel)
                            temp = temp + 10e-04 * dp_dc.mul(block.to(torch.double)).sum()
                            
                        for dim in range(input_channel):
                            filt[chan, dim, line, col] = filt[chan, dim, line, col] - temp * rate
                        # filter update finish
                        
        aft = torch.zeros((batch, input_channel, input_size, input_size)).cuda()
        
        for bat in range(batch):
            for in_chan in range(input_channel):
                bef = torch.zeros((input_size, input_size)).cuda()
                for chan in range(filter_channel):
                    base = torch.zeros(input_size, input_size).cuda()
                    fil_sum = 0.
                    for line in range(filter_size):
                        for col in range(filter_size):                        
                            # gradient update start
                            base_i = 10e-04 * torch.ones((input_size, input_size)).cuda()
                            base_i = filt[chan, in_chan, line, col] * base_i
                            base = base + base_i
                            fil_sum += filt[chan, in_chan, line, col]
                            # gradient update finish
                    base = base / fil_sum
                    bef = bef + dp_dc[bat, chan, :, :].mul(base)
                aft[bat, in_chan] = bef
        bias = bias + rate * dp_dc
        return filt, bias, aft
    
    def fcn_grad(self, prev, rate, x_enc, input_batch, input_size, output_size, weight, mat_res, sig_res, bias):
        sig = sig_res.get_plain_text().to(torch.double)
        gradient = prev.matmul(sig.reshape(input_batch, output_size, 1).matmul((1. - sig)))
        
        dW = gradient.mul(x_enc.get_plain_text().reshape(input_batch, input_size, 1))
        weight = weight - rate * dW
        bias = bias - rate * gradient
        
        aft = gradient.matmul(weight.reshape(input_batch, output_size, input_size))
        return weight, bias, aft
    
    def loss(self, prediction):
        loss = 0.
        self.pred = prediction.get_plain_text()
        for line in range(3):
            for col in range(2):
                loss += 1. / 2. * (self.__labels[line, col] - self.pred[line, 0, col]) * (self.__labels[line, col] - self.pred[line, 0, col])
        return loss
    
    def loss_grad(self):
        label = self.__labels.reshape(3, 1, 2)
        loss_grad = self.pred - torch.from_numpy(label).cuda()
        return loss_grad