import numpy as np
import pandas as pd
import torch as torch
import torch.nn.functional as F
import crypten
torch.set_printoptions(precision=16)
torch.set_default_tensor_type(torch.DoubleTensor)
crypten.init()

class network():
    def __init__(self, in_size, hid_size, out_size, rate):
        assert len(hid_size) > 0, 'a non-empty vector is expected'
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size
        self.rate = rate
        self.cons_net()
        print(f'network initialization finished')
        
    def cons_net(self):
        self.biases = torch.from_numpy(np.random.rand(len(self.hid_size) + 1))
        self.weights = []
        
        w_in = torch.from_numpy(np.random.rand(self.in_size, self.hid_size[0]))
        self.weights.append(w_in)
        
        if len(self.hid_size) > 1:
            for index in range(len(self.hid_size) - 1):
                w = torch.from_numpy(np.random.rand(self.hid_size[index], self.hid_size[index + 1]))
                self.weights.append(w)
        
        w_out = torch.from_numpy(np.random.rand(self.hid_size[-1], self.out_size))
        self.weights.append(w_out)
        
    def forward(self, x_enc):
        self.zh = []
        self.h = []
        
        zh1 = x_enc.matmul(self.weights[0]) + self.biases[0]
        self.zh.append(zh1)
        h1 = zh1.sigmoid()
        self.h.append(h1)
        
        index = 1
        
        for weight in self.weights[1:-1]:
            zh_temp = self.h[-1].matmul(weight) + self.biases[index]
            self.zh.append(zh_temp)
            h_temp = zh_temp.sigmoid()
            self.h.append(h_temp)
            index = index + 1
        
        self.zo = self.h[-1].matmul(self.weights[-1]) + self.biases[-1]
        o = self.zo.sigmoid()
        return o
    
    def backPropagation(self, o, t, x_enc):
        dE_do = o - crypten.cryptensor(t)
        do_dzo = o * (1 - o)

        dE_dzo = (dE_do * do_dzo).get_plain_text()

        dzo_dw = self.h[-1].get_plain_text()

        dE_dw = dzo_dw.T.matmul(dE_dzo)

        self.weights[-1] = self.weights[-1] - dE_dw  * self.rate

        dE_dzh = dE_dzo

        for i in range(len(self.h) - 1, 0, -1):
            dzhiii_dhii = self.weights[i + 1]
            dhii_dzhii = self.h[i].get_plain_text().T.matmul(1 - self.h[i].get_plain_text())
    
            dE_dhii = dzhiii_dhii.matmul(dE_dzh.T.double())
            dE_dzh = dE_dhii.T.matmul(dhii_dzhii.double())
    
            dzhi_dwi = self.h[i - 1].get_plain_text()
    
            dE_dwi = dzhi_dwi.T.double().matmul(dE_dzh)
    
            self.weights[i] = self.weights[i] - dE_dwi * self.rate
    
        dzh3_dh2 = self.weights[1]
        dh2_dzh2 = self.h[0].get_plain_text().T.matmul(1 - self.h[0].get_plain_text())

        dE_dh2 = dzh3_dh2.matmul(dE_dzh.T.double())
        dE_dzh = dE_dh2.T.matmul(dh2_dzh2.double())

        dzh1_dw1 = x_enc.get_plain_text()

        dE_dw1 = dzh1_dw1.T.double().matmul(dE_dzh)

        self.weights[0] = self.weights[0] - dE_dw1 * self.rate