'''

This python file is to extract data from EEGLAB and transfer them into array form

'''
import mne as mne
import numpy as np
import pandas as pd

data_root_dir = '/home/dhz/bci-data/ds003478-download/'

class DataExtract(object):
    def __init__(self, num):
        self.num = num
    
    def extract(self):
        
        res = []
        
        for sub_num in range(1, self.num + 1):

            sub_name = "sub-{0:0=3d}".format(sub_num)
            file_dir = data_root_dir + sub_name + '/eeg/' + sub_name + '_task-Rest_run-01_eeg.set'
            
            data = mne.io.read_raw_eeglab(file_dir)
            
            if sub_num == 1:
                self.channels = data.ch_names
            
            data.pick_channels(self.channels)
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
                results.append(eve.reshape((66, 128, 128)))
            results = np.array(results)
            
            res.append(results)
        res = np.array(res)
        
        labels = pd.read_csv('/home/dhz/bci-data/ds003478-download/participants.tsv', sep='\t', header=0)
        col = labels.columns
        col = col[[0, 1, 2, 3, 6, 7, 8]]
        labels = labels.drop(col, axis=1)
        
        labels = np.array(labels[:self.num])
        
        return res, labels