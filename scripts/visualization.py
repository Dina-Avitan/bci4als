# import pickle
import mne
# import os
import pickle
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# from bci4als.eeg import EEG
# from bci4als.ml_model import MLModel
# from bci4als.experiments.offline import OfflineExperiment
#from mne.channels import make_standard_montage
import scipy
from scipy import signal
# import matplotlib as mpl
# from matplotlib import cm
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

def plot_psd_classes(raw_model, classes = [0,1,2] ,elec = 0,show_std = False,fmin = 3, fmax = 70):
    colors = ['blue','darkred','green']
    std_colors = ['lightsteelblue','salmon','palegreen']
    class_name = ['Right','Left','Idle']
    sr = raw_model.epochs.info['sfreq']
    for i_cls in classes:
        indices = [i for i in range(len(raw_model.labels)) if raw_model.labels[i] == i_cls]
        data = raw_model.epochs.get_data(item=indices)
        f, Pxx = signal.welch(data, sr,window=str(sr),noverlap= 0.5*sr)
        mean = np.ndarray.mean(Pxx, axis=0)
        plt.plot(f,mean[elec],color=colors[i_cls], label = class_name[i_cls])
        if show_std == True:
            std = np.ndarray.std(Pxx, axis=0)
            res1 = mean[elec] - std[elec]/2
            res2 = mean[elec] + std[elec]/2
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



def create_spectrogram(raw_model,elec=0, nwindow=100, noverlap=10, nperseg=50,nfft = 125):
    sr = raw_model.epochs.info['sfreq']
    elec = (raw_model.epochs.ch_names[elec],elec)
    spec_dict ={}
    for i_spec in range(4):
        if i_spec < 3:
            indices = [i for i in range(len(raw_model.labels)) if raw_model.labels[i] == i_spec]
            data = raw_model.epochs.get_data(item=indices)
            f,t,Sxx = scipy.signal.spectrogram(data,  sr, window=str(nwindow), noverlap=noverlap, nperseg=nperseg,nfft=nfft)
            spec_dict[str(i_spec)] = np.ndarray.mean(Sxx, axis=0)
        else:
            indices_right = [i for i in range(len(raw_model.labels)) if raw_model.labels[i] == 0]
            indices_left = [i for i in range(len(raw_model.labels)) if raw_model.labels[i] == 1]
            data_right = raw_model.epochs.get_data(item=indices_right)
            data_left = raw_model.epochs.get_data(item=indices_left)
            f, t, Sxx_rigt = scipy.signal.spectrogram(data_right, sr, window=str(nwindow), noverlap=noverlap, nperseg=nperseg,nfft=nfft)
            f, t, Sxx_left = scipy.signal.spectrogram(data_left,  sr, window=str(nwindow), noverlap=noverlap, nperseg=nperseg,nfft=nfft)
            mean_right = np.ndarray.mean(Sxx_rigt, axis=0)
            mean_left = np.ndarray.mean(Sxx_left, axis=0)
            spec_dict[str(i_spec)] = mean_right-mean_left
        spec_dict['t'] = t
        spec_dict['f'] = f
    plot_spectrogram(spec_dict,elec)

def plot_spectrogram(spec_dict,elec):
    class_name = ['Right', 'Left', 'Idle','Right-Left diff']
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    # add a big axes, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.title('Elec: ' + f'{elec[0]}', fontsize=20,x = 0.4 , y = 1)
    for i, ax in enumerate(axs.flat):
        im = ax.pcolormesh(spec_dict['t'],spec_dict['f'] , spec_dict[str(i)][elec[1]], shading='auto',cmap = 'jet')
        ax.set_title(class_name[i])
    plt.setp(axs[-1, :], xlabel='Time [sec]')
    plt.setp(axs[:, 0], ylabel='Frequency [Hz]')
    plt.colorbar(im, ax=axs.ravel().tolist())
    plt.show()


fpath1 = 'C:\\Users\\pc\\Desktop\\bci4als\\recordings\\noam\\5\\trials.pickle'
fpath2 = 'C:\\Users\\pc\\Desktop\\bci4als\\recordings\\noam\\5\\raw_model.pickle'
fpath3 = 'C:\\Users\\pc\\Desktop\\bci4als\\recordings\\noam\\5\\trained_model.pickle'
trials = pickle.load(open(fpath1, 'rb'))
raw_model = pickle.load(open(fpath2, 'rb'))
traind_model = pickle.load(open(fpath3, 'rb'))
create_spectrogram(raw_model,elec=1)
plot_raw_elec(trials,range_time = 1)
plot_psd_classes(raw_model, classes = [0,1,2] ,elec = 0,show_std = False,fmin = 1, fmax = 70)
# plot_psd_classes(raw_model, classes = [0,1,2] ,elec = 1,show_std = False,fmin = 1, fmax = 70)
# #raw_model.plot(scalings="auto", clipping=None)
#trials[1].shape[1]
# for i in range(len(trials)):
#     # sum_col = trials[i].sum(axis=0)
#     # sum_col[sum_col == 0].index.tolist()
#     std_col = trials[i].std(axis=0)
#     to_remove = std_col[std_col == 0].index.tolist()

# to_remove = []
# for i in range(len(features_mat)):
#     nan_col = np.isnan(raw_model.trials[0]).sum(axis = 0)
#     add_remove = np.where(np.in1d(nan_col,not 0))
#     to_rermove.append(add_remove)
#     func = lambda x: x>2
#     z_mat = scipy.stats.zscore(a, axis=0, ddof=0, nan_policy='omit')
#     Z_bool = func(z_mat).sum(axis=0)
#     add_remove = np.where(np.in1d(Z_bool,not 0))
# roy roy
