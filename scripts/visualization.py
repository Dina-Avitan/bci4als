
import mne
from mne.preprocessing import ICA
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from bci4als import ml_model
from sklearn.decomposition import FastICA
import scipy
from scipy import signal
import scipy
import scipy.io
from copy import copy


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

def plot_elec_model(model, elec_num='all',range_time = 'all'):
    '''
    Args:
        model: the modelqunfilterd_model file in the subject directory
        elec_num: (list) list with the indices of electrodes (str) you want to plot.
        range_time: (tuple) the range of trials you want to plot
    Returns: plot of the voltage over time of the electrodes
    '''
    trials = model.epochs.get_data()
    if range_time == 'all':
        end_time = len(trials)
        start_time = 0
    else:
        start_time = range_time[0]
        end_time= range_time[1]
    all_sig = [trials[i] for i in range(start_time, end_time)]
    all_sig = np.concatenate(all_sig,1)
    all_sig = all_sig.transpose()
    elec_name = model.epochs.ch_names
    if elec_num != 'all':
        all_sig = all_sig[:,tuple(elec_num)]
        elec_name = [elec_name[i] for i in elec_num]
    plt.plot(all_sig)
    plt.legend(elec_name)
    plt.suptitle('EEG elec over time', fontsize=16)
    plt.show()
def plot_elec_model_ica(model, elec_num='all',range_time = 'all'):
    '''
    Args:
        model: the modelqunfilterd_model file in the subject directory
        elec_num: (list) list with the indices of electrodes (str) you want to plot.
        range_time: (tuple) the range of trials you want to plot
    Returns: plot of the voltage over time of the electrodes
    '''
    #trials = model.epochs.get_data()
    trials = ICA(data2)
    all_sig = ICA(data2)
    if range_time != 'all':
        all_sig = all_sig[range_time[0]:range_time[1]]
    elec_name = model.epochs.ch_names
    if elec_num != 'all':
        all_sig = all_sig[:,tuple(elec_num)]
        elec_name = [elec_name[i] for i in elec_num]
    plt.plot(all_sig)
    plt.legend(elec_name)
    plt.suptitle('EEG elec over time-ICA', fontsize=16)
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
    plt.ylabel('PSD')#[V**2/Hz]?
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
        im = ax.pcolormesh(spec_dict['t'],spec_dict['f'] , spec_dict[str(i)][elec[1]], shading='gouraud',cmap = 'jet')
        ax.set_title(class_name[i])
    plt.setp(axs[-1, :], xlabel='Time [sec]')
    plt.setp(axs[:, 0], ylabel='Frequency [Hz]')
    plt.colorbar(im, ax=axs.ravel().tolist())
    #plt.show()

def ICA_noam(unfiltered_model):
    # ica = ICA(n_components=15, method='fastica', max_iter="auto").fit(epochs)
    trials = unfiltered_model.epochs.get_data()
    all_sig = [trials[i] for i in range(len(trials))]
    all_sig = np.concatenate(all_sig, 1)
    #all_sig = all_sig[:,:20000]
    all_sig = all_sig.transpose()
    ica_data = np.zeros(all_sig.shape)
    transformer = FastICA(n_components=20, random_state=0)
    X_transformed = transformer.fit_transform(all_sig)
    for j in range(X_transformed.shape[1]):
        plt.plot(X_transformed[:,j])
        plt.show()
    X_transformed[:,6] = np.zeros(X_transformed.shape)[:,1]
    X_transformed[:,7] = np.zeros(X_transformed.shape)[:,1]
    ica_data = transformer.inverse_transform(X_transformed, copy=True)
    return ica_data

def ICA_check(unfiltered_model):
    """
    This function is for visualization the ICA process and for choosing coordinates to exclude
    Args:
        unfiltered_model: A model, before ICA transform
    for GUI: run this lines in the console:
             %matplotlib qt
             %gui qt
    """
    data = unfiltered_model.epochs
    epochs = data.copy()
    ica = ICA(n_components=10, max_iter='auto', random_state=0)
    ica.fit(epochs)
    ica.plot_sources(epochs,start=0, stop=6, show_scrollbars=False,title='ICA components')
    ica.plot_components(title='ICA components-topoplot')
    to_exclude = input("\nEnter a list of the numbers of the components to exclude: ")
    to_exclude = to_exclude.strip(']')
    to_exclude = [int(i) for i in to_exclude.strip('[').split(',')]
    if to_exclude:
        ica.exclude = to_exclude
    ica.apply(epochs)
    data.plot(scalings=10,title='Before ICA')
    epochs.plot(scalings=10, title='After ICA')
    # before = epochs_to_raw(data)
    # after=epochs_to_raw(epochs)
    # before.plot(scalings=10)
    # after.plot(scalings=10)

def ICA_perform(model,to_exclude):
    """
    Args:
        model: the model before ICA transform
        to_exclude: (list) list of the coordinates numbers to exclude

    Returns: epochs array after ICA transform
    """
    epochs = model.epochs
    ica = ICA(n_components=10, max_iter='auto', random_state=97)
    ica.fit(epochs)
    ica.exclude = to_exclude
    ica.apply(epochs)
    return epochs

def epochs_z_score(epochs):
    """
    this function is for normalize all the electrods in epochs array by Z_score
    Args:
        epochs: epochs array
    Returns: (ndarray) the data after normalizing all the electrods
    """
    data=epochs.get_data()
    for i in range(len(data)):
        scaler = StandardScaler()
        scaler.fit(data[i].transpose())
        array= scaler.transform(data[i].transpose())
        data[i]= array.transpose()
    return data

def get_feature_mat(model):
    # define parameters
    fs = 125
    bands = np.matrix('7 12; 12 15; 17 22; 25 30; 7 35; 30 35')
    # get data
    data = model.epochs.get_data()
    class_labels = model.labels
    feature_labels = []
    # get features
    # CSP
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)#, transform_into='average_power', cov_est='epoch')
    csp_features = Pipeline([('CSP', csp), ('LDA', lda)]).fit_transform(data, class_labels)
    [feature_labels.append(f'CSP_Component{i}') for i in range(csp_features.shape[1])]
    # Bandpower
    bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
    [feature_labels.append(f'BP_non_rel{np.ravel(i)}_{chan}') for i in bands for chan in model.epochs.ch_names]
    # relative bandpower
    bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
    [feature_labels.append(f'BP_non_rel{np.ravel(i)}_{chan}') for i in bands for chan in model.epochs.ch_names]
    # hjorthMobility
    hjorthMobility_features = ml_model.MLModel.hjorthMobility(data)
    [feature_labels.append(f'hjorthMobility_{chan}') for chan in model.epochs.ch_names]
    # LZC
    LZC_features = ml_model.MLModel.LZC(data)
    [feature_labels.append(f'LZC_{chan}') for chan in model.epochs.ch_names]
    # DFA
    DFA_features = ml_model.MLModel.DFA(data)
    [feature_labels.append(f'DFA_{chan}') for chan in model.epochs.ch_names]
    # get all of them in one matrix
    features_mat = np.concatenate((csp_features,bandpower_features_new, bandpower_features_rel,
                                   hjorthMobility_features,LZC_features,DFA_features), axis=1)
    scaler = StandardScaler()
    scaler.fit(features_mat)
    return features_mat, class_labels, feature_labels

def histo_histo(features_mat,class_labels, features_labels):
    right_indices = [i for i in range(len(class_labels)) if class_labels[i] == 0]
    left_indices = [i for i in range(len(class_labels)) if class_labels[i] == 1]
    idle_indices = [i for i in range(len(class_labels)) if class_labels[i] == 2]
    num_plot = 0
    for subplt in range(0,features_mat.shape[1],12):
        num_plot += 1
        fig, axs = plt.subplots(3, 4, figsize=(16, 10), facecolor='w', edgecolor='k')
        #fig.subplots_adjust(hspace=.5, wspace=.001) -0.65 -1.227
        axs = axs.ravel()
        for feature in range(subplt, min(subplt+12,features_mat.shape[1])):#features_mat.shape[1]):
            x = features_mat[:,feature]
            right = [x[i] for i in right_indices]
            left = [x[i] for i in left_indices]
            idle = [x[i] for i in idle_indices]
            axs[feature-subplt].hist(right,bins=30, alpha=0.4, label='right')
            axs[feature-subplt].hist(left,bins=30, alpha=0.4, label='left')
            axs[feature-subplt].hist(idle,bins=30, alpha=0.4, label='idle')
            axs[feature-subplt].set_title(features_labels[feature],size='large',fontweight='bold')
            axs[feature-subplt].legend(loc='upper right')
        fig.suptitle(f'Features histograms {num_plot}',size='xx-large',fontweight='bold')
        fig.text(0.04, 0.5, 'Probability', va='center', rotation='vertical',size='xx-large',fontweight='bold')
        plt.show()

def epochs_to_raw(epochs):
    trials = epochs.get_data()
    data = [trials[i] for i in range(len(trials))]
    data = np.concatenate(data, 1)
    info= mne.create_info(ch_names=epochs.ch_names,sfreq= 125 , ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    return raw

def ndarray_to_raw(data, ch_names):
    comb_data = [data[i] for i in range(len(data))]
    comb_data = np.concatenate(data, 1)
    info= mne.create_info(ch_names=ch_names,sfreq= 125 , ch_types='eeg')
    raw = mne.io.RawArray(comb_data, info)
    return raw

data2 = pd.read_pickle(r'C:\Users\pc\Desktop\bci4als\recordings\roy\3\unfiltered_model.pickle')
data3 = pd.read_pickle(r'C:\Users\pc\Desktop\bci4als\recordings\roy\10\trials.pickle')
raw_model = pd.read_pickle(r'C:\Users\pc\Desktop\bci4als\recordings\roy\10\raw_model.pickle')
#
#epochs_to_raw(data2.epochs)
"""
%matplotlib qt
%gui qt
mne.viz.set_browser_backend('qt')
"""

# plot_psd_classes(raw_model)
# features_mat, class_lables, features_lables = get_feature_mat(data2)
# histo_histo(features_mat, class_lables, features_lables)




#plot_raw_elec(data3, elec_name='all',range_time = 'all')
#ICA_noam(data2)
# plot_elec_model(data2, elec_num='all',range_time = 'all')
# plot_elec_model_ica(data2, elec_num='all',range_time = 'all')
# plot_elec_model(data2, elec_num='all',range_time = (0,3))
# plot_elec_model_ica(data2, elec_num='all',range_time = (0,2400))
# labels = data2.labels
# data = data2.epochs.get_data()
# perm_c3 = (0, 3, 5, 9, 7, 1, 4, 6, 8, 10)
# C3_not_laplacian = data[1][perm_c3[0]]
# plt.plot(C3_not_laplacian)
# #plt.show()
# plt.plot(data[1][perm_c3[5]])
# #plt.show()
# plt.plot(data[1])
# #plt.show()
# for trial in range(data.shape[0]):
#     # C3
#     data[trial][perm_c3[0]] = (data[trial][perm_c3[0]]-data[trial][perm_c3[0]].mean())-\
#                               (((data[trial][perm_c3[1]]-data[trial][perm_c3[1]].mean())
#                               + (data[trial][perm_c3[2]]-data[trial][perm_c3[2]].mean())
#                               + (data[trial][perm_c3[3]]-data[trial][perm_c3[3]].mean())
#                               + (data[trial][perm_c3[4]]-data[trial][perm_c3[4]].mean())) / 4)
#     # C4
#     data[trial][perm_c3[5]] = (data[trial][perm_c3[5]] - data[trial][perm_c3[5]].mean()) - \
#                               (((data[trial][perm_c3[6]] - data[trial][perm_c3[6]].mean())
#                               + (data[trial][perm_c3[7]]  -data[trial][perm_c3[7]].mean())
#                               + (data[trial][perm_c3[8]] - data[trial][perm_c3[8]].mean())
#                               + (data[trial][perm_c3[9]] - data[trial][perm_c3[9]].mean())) / 4)
#     new_data = np.delete(data[trial], [perm_c3[point] for point in [1, 2, 3, 4, 6, 7, 8, 9]], axis=0)
#     if trial == 0:
#         final_data = new_data[np.newaxis]
#     else:
#         final_data = np.vstack((final_data, new_data[np.newaxis]))
# one_tr = final_data[1,:,:]
# one_tr = one_tr.transpose()
# plt.plot(one_tr)
#plt.show()
#for trial in range(data.shape[0]):
# data=data2.epochs
# data.plot(picks=data.picks, scalings=None, n_epochs=1, n_channels=len(data.ch_names))#, title='noam', events=data.events, event_colors=None, order=None, show=True, block=False, decim='auto', noise_cov=None, butterfly=False)

