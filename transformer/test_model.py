import numpy as np
import pandas as pd
import torch as torch
import torch.nn.functional as F
import crypten
from network import Network
import network
torch.set_printoptions(precision=16)
torch.set_default_tensor_type(torch.DoubleTensor)
crypten.init()

if __name__ == '__main__':
    nn = Network(4, [5,6,7,5,6], 3, 0.1)
    x = [[3.,2.,3.,1.]]
    x_enc = crypten.cryptensor(x)
    t = torch.tensor([[1.,0.,0.]])

    for i in range(100):
        o = nn.forward(x_enc)
        o_plaintext = o.get_plain_text()[0]
        loss = nn.crossEntropy(o_plaintext, t[0])
        print(f'loss of epoch[{i + 1}/100] is: {loss}')
        nn.backPropagation(o, t, x_enc)