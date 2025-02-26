
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
import json
from sklearn.metrics import confusion_matrix
import seaborn

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

def plot_psd_classes(raw_model, classes = [0,1,2] ,elec = 0,show_std = False,fmin = 0, fmax = 70):
    """
    Plot the powerspectrum
    Args:
        raw_model: The data as model
        classes: The Classes you want to plot
        elec: The electrod index you want to plot
        show_std: (bool) if you want to show standard division
        fmin, fmax: The range

    """
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

def plot_psd_classes_trials(raw_model, epoch, classes = [0,1,2] ,elec = 0,show_std = False,fmin = 1, fmax = 70):
    """
        Plot the powerspectrum
        Args:
            raw_model: The data as model
            epoch: The data as epoch
            classes: The Classes you want to plot
            elec: The electrod index you want to plot
            show_std: (bool) if you want to show standard division
            fmin, fmax: The range

        """
    colors = ['blue','darkred','green']
    std_colors = ['lightsteelblue','salmon','palegreen']
    class_name = ['Right','Left','Idle']
    sr = raw_model.epochs.info['sfreq']
    for i_cls in classes:
        indices = [i for i in range(len(raw_model.labels)) if raw_model.labels[i] == i_cls]
        data = epoch.get_data(item=indices)
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

def create_spectrogram(raw_model,elec=0, nwindow=100, noverlap=10, nperseg=50,nfft = 125,scaling = 'spectrum'):
    """
    The function create spectrograms of the data
    Args:
        raw_model: The data as model
        elec: he electrod index you want to plot
        nwindow, noverlap, nperseg, nfft, scaling: hyperparameters- read in scipy.signal.spectrogram documentation

    """
    sr = raw_model.epochs.info['sfreq']
    elec = (raw_model.epochs.ch_names[elec],elec)
    spec_dict ={}
    for i_spec in range(4):
        if i_spec < 3:
            indices = [i for i in range(len(raw_model.labels)) if raw_model.labels[i] == i_spec]
            data = raw_model.epochs.get_data(item=indices)
            f,t,Sxx = scipy.signal.spectrogram(data,  sr, window=str(nwindow), noverlap=noverlap, nperseg=nperseg,nfft=nfft,scaling=scaling)
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
            spec_dict[str(i_spec)] = abs(mean_right-mean_left)
        spec_dict['t'] = t
        spec_dict['f'] = f
    plot_spectrogram(spec_dict,elec)

def create_spectrogram_raw(all_data,labels,elec=0, nwindow=100, noverlap=10, nperseg=50,nfft = 125,scaling = 'spectrum'):
    """
        The function create spectrograms of the data- from ndarray
        Args:
            raw_model: The data as ndarray
            labels: The labels
            elec: he electrod index you want to plot
            nwindow, noverlap, nperseg, nfft, scaling: hyperparameters- read in scipy.signal.spectrogram documentation

        """
    all_elec = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6'];
    sr = 125
    elec = (all_elec[elec],elec)
    spec_dict ={}
    for i_spec in range(4):
        if i_spec < 3:
            indices = [i for i in range(len(labels)) if labels[i] == i_spec]
            data = all_data[indices,:,:]
            f,t,Sxx = scipy.signal.spectrogram(data,  sr, window=str(nwindow), noverlap=noverlap, nperseg=nperseg,nfft=nfft,scaling=scaling)
            spec_dict[str(i_spec)] = np.ndarray.mean(Sxx, axis=0)
        else:
            indices_right = [i for i in range(len(labels)) if labels[i] == 0]
            indices_left = [i for i in range(len(labels)) if labels[i] == 1]
            data_right = all_data[indices_right,:,:]
            data_left = all_data[indices_left,:,:]
            f, t, Sxx_rigt = scipy.signal.spectrogram(data_right, sr, window=str(nwindow), noverlap=noverlap, nperseg=nperseg,nfft=nfft)
            f, t, Sxx_left = scipy.signal.spectrogram(data_left,  sr, window=str(nwindow), noverlap=noverlap, nperseg=nperseg,nfft=nfft)
            mean_right = np.ndarray.mean(Sxx_rigt, axis=0)
            mean_left = np.ndarray.mean(Sxx_left, axis=0)
            spec_dict[str(i_spec)] = abs(mean_right-mean_left)
        spec_dict['t'] = t
        spec_dict['f'] = f
    plot_spectrogram(spec_dict,elec)

def plot_spectrogram(spec_dict,elec):
    """
    Helper function of the "reate_spectrogram" function

    """
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
    plt.show()

def ICA_without_mne(unfiltered_model):
    """
    This function preforms ICA process without using mne.
    Args:
        unfiltered_model: A model, before ICA transform

    Returns: The data after ICA

    """
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
             mne.viz.set_browser_backend('qt')
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

def laplacian(model):
    """
    The function preforms laplacian
    Args:
        model: The data use to be as model structure

    Returns: The data after laplacian

    """
    perm_c3 = (0, 5, 3, 9, 7, 1, 4, 6, 8, 10)
    data = model.epochs.get_data()
    for trial in range(data.shape[0]):
        # C3
        data[trial][perm_c3[0]] -= (data[trial][perm_c3[1]] + data[trial][perm_c3[2]] + data[trial][perm_c3[3]] +
                              data[trial][perm_c3[4]]) / 2
        # C4
        data[trial][perm_c3[5]] -= (data[trial][perm_c3[6]] + data[trial][perm_c3[7]] + data[trial][perm_c3[8]] +
                              data[trial][perm_c3[9]]) / 2
        new_data = np.delete(data[trial], [perm_c3[point] for point in [1, 2, 3, 4, 6, 7, 8, 9]], axis=0)
        # new_data = data[trial]
        if trial == 0:
            final_data = new_data[np.newaxis]
        else:
            final_data = np.vstack((final_data, new_data[np.newaxis]))
    data = final_data
    return data

def get_feature_mat(model, do_laplacian = True):
    """
    Create features matrix with a lot of kinds of features
    Args:
        model: The data use to be as model structure
        do_laplacian: (bool) if you want to preform laplcian

    Returns:

    """
    # define parameters
    fs = 125
    bands = np.matrix('7 12; 12 15; 17 22; 25 30; 7 35; 30 35')
    # get data
    if do_laplacian:
        data = laplacian(model)
        chan_list = ["c3", "c4", "cz"]
    else:
        data = model.epochs.get_data()
        chan_list = model.epochs.ch_names
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
    [feature_labels.append(f'BP_non_rel{np.ravel(i)}_{chan}') for i in bands for chan in chan_list]
    # relative bandpower
    bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
    [feature_labels.append(f'BP_non_rel{np.ravel(i)}_{chan}') for i in bands for chan in chan_list]
    # hjorthMobility
    hjorthMobility_features = ml_model.MLModel.hjorthMobility(data)
    [feature_labels.append(f'hjorthMobility_{chan}') for chan in chan_list]
    # LZC
    LZC_features = ml_model.MLModel.LZC(data)
    [feature_labels.append(f'LZC_{chan}') for chan in chan_list]
    # DFA
    DFA_features = ml_model.MLModel.DFA(data)
    [feature_labels.append(f'DFA_{chan}') for chan in chan_list]
    # get all of them in one matrix
    features_mat = np.concatenate((csp_features,bandpower_features_new, bandpower_features_rel,
                                   hjorthMobility_features,LZC_features,DFA_features), axis=1)
    scaler = StandardScaler()
    scaler.fit(features_mat)
    return features_mat, class_labels, feature_labels

def histo_histo(features_mat,class_labels, features_labels):
    """
    This function creates histograms of the features To get a visual measure of the ability to separate the classes
    Args:
        features_mat: The features matrix
        class_labels: The class labels
        features_labels: The features names

    """
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
    """
    Convert epoch (mne structure) to raw (mne structure)
    good fot ICA

    """
    trials = epochs.get_data()
    data = [trials[i] for i in range(len(trials))]
    data = np.concatenate(data, 1)
    info= mne.create_info(ch_names=epochs.ch_names,sfreq= 125 , ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    return raw

def ndarray_to_raw(data, ch_names):
    """
     Convert epoch (mne structure) to raw (mne structure)
    good fot ICA

    Returns:raw (mne structure)

    """
    comb_data = [data[i] for i in range(len(data))]
    comb_data = np.concatenate(comb_data, 1)
    info= mne.create_info(ch_names=ch_names,sfreq= 125 , ch_types='eeg')
    raw = mne.io.RawArray(comb_data, info)
    return raw

def plot_online_results(path):
    """
    Thif function plots the results of the online session:
    Bar plot + Confusion matrix
    Args:
        path: The path of the data
    """
    with open(path) as f:
        data = json.load(f)
    #rep_on_class = len(data[0])
    num_of_trials_class = len(data)/3
    results_dict = {'0': 0, '1':0,'2':0}
    expected = []
    prediction = []
    for trial in data:
        rep_on_class = len(trial)
        for ind in trial:
            if ind[0]==ind[1]:
                results_dict[str(ind[0])] += 1/(rep_on_class*num_of_trials_class)
            expected.append(ind[0])
            prediction.append(ind[1])

    # the bar plot
    labels = ['Right', 'Left', 'Idle']
    classes = list(results_dict.keys())
    values = list(results_dict.values())
    plt.bar(classes,values,color = (0.5,0.1,0.5,0.6))
    plt.title('Online results - The prediction percentage for each class\n')
    plt.xlabel('Prediction percentage')
    plt.ylabel('Classes ')
    plt.xticks(classes,labels)
    plt.show()

    # the confusion matrix
    cm = confusion_matrix(expected,prediction)
    ax = seaborn.heatmap((cm*3)/np.sum(cm),fmt='.2%', annot=True, cmap='Blues')
    ax.set_title('Online results - confusion matrix\n')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')
    ## Ticket labels - List must be in alphabetical order7k
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()


