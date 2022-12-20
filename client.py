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
        self.crypt_dataset = crypten.cryptensor(self.__plain_dataset).cuda()
        
    def get_data(self):
        return self.crypt_dataset
    
    def conv_grad(self, prev, p_res, r_res, c_res, rate, batch, input_channel, filter_channel, filter_size, filt):
        p_plain = p_res.get_plain_text()
        pool_raw = prev.matmal(p_plain).repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        
        relu_plain = r_res.get_plain_text()
        dp_dr = torch.eq(pool_raw, relu_plain, out=torch.tensor([1.,0.]).cuda())
        
        conv_plain = c_res.get_plain_text()
        dr_dc = torch.eq(relu_plain, conv_plain, out=torch.tensor([1.,0.]).cuda())
        
        dp_dc = dp_dr.matmul(dr_dc)
        x_pad = self.crypt_dataset.pad([1, 1, 1, 1]).get_plain_text()

        for bat in range(batch):
            for chan in range(filter_channel):
                for line in range(filter_size):
                    for col in range(filter_size):
                        window = x_pad[bat, :, line : (line + 128), col : (col + 128)]
                        temp = 0.
                        for block in window:
                            temp = temp + 10e-04 * dc.matmul(block.to(torch.double)).sum() / pow((batch * filter_channel * filter_size * filter_size), input_channel)
                        for dim in range(input_channel):
                            filt[chan, dim, line, col] = filt[chan, dim, line, col] + temp * rate
                    
        return filt, dc
        