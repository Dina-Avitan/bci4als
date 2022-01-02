import pickle
import mne
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bci4als.eeg import EEG
from bci4als.ml_model import MLModel
from bci4als.experiments.offline import OfflineExperiment
from mne.channels import make_standard_montage
import scipy
from scipy import signal

def plot_raw_elec(trials, elec_name='all',range_time = 'all'):
    '''
    Args:
        trials: the trials file in the subject directory
        elec_name: (list) list with the names of electrodes (str) you want to plot.
        range_time: (num) the num of trials form the beginning you want to plot
    Returns: plot of the voltage over time of the electrodes
    '''
    if elec_name == 'all':
        elec_name = list(trials[0].columns)
    if range_time == 'all':
        range_time = len(trials)
    all_sig = [trials[i][elec_name] for i in range(range_time)]
    all_sig = np.concatenate(all_sig)
    plt.plot(all_sig)
    plt.legend(elec_name)
    plt.suptitle('EEG elec over time', fontsize=16)
    plt.show()

def plot_psd_classes(raw_model, classes = [0,1,2],elec = 0,show_std = False,fmin = 3, fmax =70):
    colors = ['blue','darkred','green']
    std_colors = ['lightsteelblue','salmon','palegreen']
    class_name = ['Right','Left','Idle']
    sr = raw_model.epochs.info['sfreq']
    for i_cls in classes:
        indices = [i for i in range(len(raw_model.labels)) if raw_model.labels[i] == i_cls]
        data = raw_model.epochs.get_data(item=indices)
        f, Pxx = signal.welch(data, sr,window=str(sr),noverlap= 0.5*sr)
        mean = np.ndarray.mean(Pxx,axis=0)
        plt.plot(f,mean[elec],color=colors[i_cls], label = class_name[i_cls])
        if show_std == True:
            std = np.ndarray.std(Pxx, axis=0)
            res1 = mean[elec] - std[elec]
            res2 = mean[elec] + std[elec]
            plt.plot(f,res1,color=std_colors[i_cls])
            plt.plot(f, res2, color=std_colors[i_cls])
            plt.fill_between(f,mean[elec], res1,color=std_colors[i_cls])
            plt.fill_between(f, mean[elec], res2, color=std_colors[i_cls])
    title_cls = '-'.join(class_name[i] for i in classes)
    plt.title('Elec: '+f'{raw_model.epochs.ch_names[elec]}'+', Classes:  '+title_cls)
    plt.legend()
    plt.xlim(fmin, fmax)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD- semilogy Scale')#[V**2/Hz]?
    plt.show()


fpath1 = 'C:\\Users\\pc\\Desktop\\bci4als\\recordings\\noam\\2\\trials.pickle'
fpath2 = 'C:\\Users\\pc\\Desktop\\bci4als\\recordings\\noam\\2\\raw_model.pickle'
fpath3 = 'C:\\Users\\pc\\Desktop\\bci4als\\recordings\\train_model.pickle'
trials = pickle.load(open(fpath1, 'rb'))
raw_model = pickle.load(open(fpath2, 'rb'))
# raw_data = pickle.load(open(fpath3, 'rb'))
#plot_raw_elec(trials)
plot_psd_classes(raw_model, classes = [0,1],fmin =3,fmax = 70)
plot_psd_classes(raw_model, classes = [0],fmin =3,fmax = 70,show_std=True)
plot_psd_classes(raw_model, classes = [1],fmin =3,fmax = 70,show_std=True)
plot_psd_classes(raw_model, classes = [0,1,2],fmin =3,fmax = 70,show_std=True)
    # # set montage
    # montage = make_standard_montage('standard_1020')
    # epochs.set_montage(montage)
    #
    # # Apply band-pass filter
    # epochs.filter(7., 30., fir_design='firwin', skip_by_annotation='edge', verbose=False)
# if d == 1:
#     D = 1
# #def visualization(raw_data):
