'''
data-extractor_v2.py

Trying to extract data from BCI MI dataset
Target result: EEG, labels, channels, frequency
'''

# get dependency
import numpy as np
import scipy.io
import torch

from mne.io import concatenate_raws, read_raw_edf
import mne

import matplotlib.pyplot as plt
import matplotlib as mpl

class Extractor(object):
    # list of channel
    __chan_list = ['C3', 'Cz', 'C4']
    
    __event_dict = {1: 'left fist', 2: 'right fist', 
                3: 'imagine left fist', 4: 'imagine right fist', 
                5: 'both fist', 6: 'both feet', 
                7: 'imagine both fist', 8: 'imagine both feet'}
    
    # construct Extractor
    def __init__(self, filename, database):
        
        self.database = database
        self.filename = filename
        for i in range(2, 14):
            self.__ext(i)
            self.eeg_classification((i - 2) % 4)
        # self.eeg_save(self.eeg_database)
        
    def __ext(self, num):
        
        raw = read_raw_edf("/home/dhz/bci-data/files/" + self.filename + "/" + self.filename + "R{0:0=2d}".format(num + 1) + ".edf",preload=False)
        
        self.events_from_annot, self.event_dict = mne.events_from_annotations(raw)
        self.data, self.times = raw[[8, 10, 12],:]
        
    
    def eeg_classification(self, loop):
        for index in range(len(self.events_from_annot)):
            if self.events_from_annot[index][2] == 1:
                continue
            elif index == len(self.events_from_annot) - 1:
                self.database[loop * 2 + self.events_from_annot[index][2] 
                              - 2].append(self.data[:, self.events_from_annot[index][0] : self.data.shape[1]])
            else:
                self.database[loop * 2 + self.events_from_annot[index][2] 
                              - 2].append(self.data[:, self.events_from_annot[index][0] : self.events_from_annot[index + 1][0]])
        return self.database
                
    def get_data(self):
        return self.database
    
    def eeg_save(self, eeg_dataset):
        np.save("/home/dhz/bci-data/files/" + self.filename + "/" + self.filename + "R{0:0=2d}".format(7) + '.npy', eeg_dataset)
        
def signal_overview(signals, times, names):
    '''
    show the overview of EEG signals
    
    Parameters
    ---------
    signals: 2-d array
        the EEG signal data
    names: 1-d array
        the list of channels name
    '''
    plt.figure(figsize = (18, 3))
    for i in range(len(names)):
        plt.subplot(1, len(names), i + 1)
        plt.plot(times, signals[i].T)
        plt.title(names[i])
    plt.show()
    
def load_eeg(filename):
    return np.load('/home/dhz/bci-data/files/' + filename + '.npy', allow_pickle = True).item()

def data_normalization(eeg_data):
    labels = []
    data = []
    for l in eeg_data:
        index = 0
        for i in l:
            l_temp = [0., 0., 0., 0., 0., 0., 0., 0.]
            l_temp[index] = 1.
            labels.append(l_temp)
            i = [np.pad(x, (0, 1024 - len(x)), 'constant') for x in i]
            data.append([x.reshape((len(x), 1)) for x in i])
        index = index + 1
            
    return torch.tensor(labels), torch.tensor(data)