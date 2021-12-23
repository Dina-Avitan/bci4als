import pickle
import mne
import os
import pickle
from bci4als.eeg import EEG
from bci4als.ml_model import MLModel
from bci4als.experiments.offline import OfflineExperiment
from mne.channels import make_standard_montage


class Visualization:
    def __init__(self, epochs,info,raw_data):
        # convert data to mne.Epochs
        self.epochs = epochs
        self.info = info
        self.raw_data = raw_data
    def plot_raw_data(self):
        print('plot_raw_data')
        # mne.viz.plot_raw(self.raw_data)
        self.epochs.plot()
        self.epochs.plot_psd(fmin=5, fmax = 60, picks='eeg')
        montage = make_standard_montage('standard_1020')
        epochs.set_montage(montage)
        # Apply band-pass filter
        epochs.filter(7., 30., fir_design='firwin', skip_by_annotation='edge', verbose=False)
        self.epochs.plot(n_epochs=10)
        self.epochs.plot_psd(fmin=5, fmax=60, picks='eeg')

fpath1 = 'C:\\Users\\pc\\Desktop\\bci4als\\recordings\\epochs.pickle'
fpath2 = 'C:\\Users\\pc\\Desktop\\bci4als\\recordings\\info.pickle'
fpath3 = 'C:\\Users\\pc\\Desktop\\bci4als\\recordings\\raw_data.pickle'
epochs = pickle.load(open(fpath1, 'rb'))
info = pickle.load(open(fpath2, 'rb'))
raw_data = pickle.load(open(fpath3, 'rb'))
a = Visualization(epochs,info,raw_data)
a.plot_raw_data()

    # # set montage
    # montage = make_standard_montage('standard_1020')
    # epochs.set_montage(montage)
    #
    # # Apply band-pass filter
    # epochs.filter(7., 30., fir_design='firwin', skip_by_annotation='edge', verbose=False)
# if d == 1:
#     D = 1
# #def visualization(raw_data):
