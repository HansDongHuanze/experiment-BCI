'''

This python file is to extract data from EEGLAB and transfer them into array form

'''
import mne as mne
import numpy as np
import pandas as pd
import torch as torch

data_root_dir = '/home/dhz/bci-data/ds003478-download/'

class DataExtract(object):
    def __init__(self, num):
        self.num = num
    
    def extract(self):

        sub_name = "sub-{0:0=3d}".format(self.num)
        file_dir = data_root_dir + sub_name + '/eeg/' + sub_name + '_task-Rest_run-01_eeg.set'

        data = mne.io.read_raw_eeglab(file_dir)

        data.pick_channels(['C3', 'CZ', 'C4'])
        signals, times = data.get_data(return_times = True)

        seg = data.annotations
        dataframe = seg.to_data_frame()
        newframe = dataframe.groupby(dataframe.description)

        eves = ['1', '3', '5']
        signals = []

        for i in eves:
            temp = newframe.get_group(i)
            head = temp.head(1).index[0]
            start = seg[head]['onset']
            signal = data.copy().crop(tmin=start, tmax=start + 1024 * 16 * 0.002 - 0.002)
            signal = signal.get_data(return_times = False)

            signals.append(signal)

        signals = np.array(signals)

        results = []

        for eve in signals:
            results.append(eve.reshape((3, 128, 128)))
        results = np.array(results)
        res = np.array(results)
        
        labels = pd.read_csv('/home/dhz/bci-data/ds003478-download/participants.tsv', sep='\t', header=0)
        col = labels.columns
        col = col[[0, 1, 2, 3, 6, 7, 8]]
        labels = labels.drop(col, axis=1)
        
        labels = np.array([labels, labels, labels])
        
        return torch.from_numpy(res) * 10e06 + 300, labels