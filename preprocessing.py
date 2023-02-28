import os
import numpy as np
import pandas as pd
import mne
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA

data_root_dir = '/home/dhz/bci-data/ds003478-download/'

class Preprocessing(object):
    def __init__(self, num):
        self.num = num
        sub_name = "sub-{0:0=3d}".format(self.num)
        self.file_dir = data_root_dir + sub_name + '/eeg/' + sub_name + '_task-Rest_run-01_eeg.set'
        
    def getData(self):
        raw = mne.io.read_raw_eeglab(self.file_dir)
        raw.set_channel_types({'VEOG': 'eog'})
        raw.pick(picks='all', exclude=['HEOG', 'EKG'])
        
        events = mne.events_from_annotations(raw)
        
        ind = np.where(events[0][:, 2] == events[1]['3'])[0].min()
        start_t = events[0][ind, 0]
        ind = np.where(events[0][:, 2] == events[1]['3'])[0].max()
        end_t = events[0][ind, 0]
        
        raw_cropped = raw.copy()
        raw_cropped.crop(tmin=start_t / 500, tmax=end_t / 500)
        raw_cropped.set_montage('standard_1020', on_missing='warn')
        
        raw_ref = raw_cropped.copy()
        raw_ref.load_data()
        raw_ref.set_eeg_reference(ref_channels=['M1', 'M2'])
        
        raw_filter = raw_ref.copy()
        raw_filter.filter(l_freq=1, h_freq=None)
        
        ica = mne.preprocessing.ICA(n_components=13, method='fastica', max_iter='auto', random_state=97)
        ica.fit(raw_filter)
        ica.exclude = [0, 3, 7]
        raw_ica = raw_filter.copy()
        raw_ica.load_data()
        ica.apply(raw_ica)
        
        raw_downsample = raw_ica.copy().resample(sfreq=50).crop(tmin=0, tmax=16 * 64 / 50 - 1./ 50)
        picks = mne.pick_types(raw_downsample.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        
        X = raw_downsample.get_data(picks=picks)
        return X