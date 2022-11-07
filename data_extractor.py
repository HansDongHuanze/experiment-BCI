'''
data-extractor.py

Trying to extract data from dataset 2
Target result: EEG, labels, channels, frequency
'''

# get dependency
import numpy as np
import scipy.io
import torch

import matplotlib.pyplot as plt
import matplotlib as mpl

class Extractor(object):
    # list of channel
    __chan_list = ['C3', 'Cz', 'C4']
    
    __event_dict = {276: 'Idling EEG', 277: 'Idling EEG', 768: 'Start of a trial',
            769: 'Cue onset left (class 1)', 770: 'Cue onset right (class 2)',
            771: 'Cue onset foot (class 3)', 772: 'Cue onset tongue (class 4)',
            783: 'Cue unknown', 1023: 'Rejected trial', 1072: 'Eye movements',
            32766: 'Start of a new run'}
    
    # construct Extractor
    def __init__(self, filename):
        
        self.filename = filename
        self.__ext()
        self.eeg_dataset = self.eeg_classification()
        self.eeg_save(self.eeg_dataset)
        
    def __ext(self):
        
        # load matlab files
        t = scipy.io.loadmat('/home/dhz/bci-data/data/' + self.filename + '.mat', struct_as_record=True)
        
        if self.filename == 'A01T' or self.filename == 'A01E':
            signals = t['s']
        
            self.typ = t['h']['EVENT'][0][0]['TYP'][0][0] # type of the event
            self.pos = t['h']['EVENT'][0][0]['POS'][0][0] # position of the event happened in timeline
            self.dur = t['h']['EVENT'][0][0]['DUR'][0][0] # duration of the event
        
            artifact_selection = t['h']['ArtifactSelection'][0][0]
        
        else:
            signals = t['s' + self.filename]
        
            self.typ = t['h' + self.filename]['EVENT'][0][0]['TYP'][0][0] # type of the event
            self.pos = t['h' + self.filename]['EVENT'][0][0]['POS'][0][0] # position of the event happened in timeline
            self.dur = t['h' + self.filename]['EVENT'][0][0]['DUR'][0][0] # duration of the event
        
            artifact_selection = t['h' + self.filename]['ArtifactSelection'][0][0]
        
        C3 = signals[:, 7]
        Cz = signals[:, 9]
        C4 = signals[:, 11]
        
        self.signal_list = [C3, Cz, C4]
        
    def eeg_classification(self):
        '''
        put EEG signals into small pieces
    
        Parameters
        ---------
        typ: 1-d array
            the EEG signal type list
        pos: 1-d array
            the EEG signal position list
        dur: 1-d array
            the EEG signal duration list
        
        Return
        ------
        result: 3-d array
            classified EEG signal, from 769 to 772
        '''
        tag_class = [769, 770, 771, 772]
        result = {769: [], 770: [], 771: [], 772: []}
        for i in range(len(self.typ)):
            if(self.typ[i] in tag_class):
                result[int(self.typ[i])].append([tmp[int(self.pos[i]):(int(self.pos[i]) + int(self.dur[i]))] for tmp in self.signal_list])
        return result
    
    def eeg_save(self, eeg_dataset):
        np.save('/home/dhz/bci-data/data/' + self.filename + '.npy', eeg_dataset)
        
def signal_overview(signals, names):
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
        plt.title(names[i])
        plt.plot(signals[i])
    plt.show()
    
def load_eeg(filename):
    return np.load('/home/dhz/bci-data/data/' + filename + '.npy', allow_pickle = True).item()

def data_normalization(eeg_data):
    labels = []
    data = []
    for l in [769, 770, 771, 772]:
        index = 0
        for i in range(len(eeg_data[l])):
            l_temp = [0., 0., 0., 0.]
            l_temp[index] = 1.
            labels.append(l_temp)
            eeg_data[l][i] = [np.pad(x, (0, 512 - len(x)), 'constant') for x in eeg_data[l][i]]
            data.append([x.reshape((len(x), 1)) for x in eeg_data[l][i]])
        index = index + 1
            
    return torch.tensor(labels), torch.tensor(data)