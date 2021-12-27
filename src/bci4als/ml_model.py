import os
import pickle
from typing import List
import mne
import pandas as pd
from bci4als.eeg import EEG
import numpy as np
from matplotlib.figure import Figure
from mne.channels import make_standard_montage
from mne.decoding import CSP
from nptyping import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
import scipy
from sklearn import svm
from sklearn.model_selection import cross_val_score
from abc import abstractstaticmethod

class MLModel:
    """
    A class used to wrap all the ML model train, partial train and predictions

    ...

    Attributes
    ----------
    trials : list
        a formatted string to print out what the animal says
    """

    def __init__(self, trials: List[pd.DataFrame], labels: List[int], channel_removed:List[str]):

        self.trials: List[NDArray] = [t.to_numpy().T for t in trials]
        self.labels: List[int] = labels
        self.channel_removed: List[str] = channel_removed
        self.debug = True
        self.clf = None
        self.features_mat = None
        self.epochs = None

    def offline_training(self, epochs, model_type: str = 'csp_lda'):

        if model_type.lower() == 'csp_lda':
            self._csp_lda(epochs)

        elif model_type.lower() == 'simple_svm':
            self._simple_svm(epochs)

        else:
            raise NotImplementedError(f'The model type `{model_type}` is not implemented yet')

    def epochs_extractor(self, eeg: EEG):
        # convert data to mne.Epochs
        ch_names = eeg.get_board_names()
        [ch_names.remove(bad_ch) for bad_ch in self.channel_removed if bad_ch in ch_names]
        ch_types = ['eeg'] * len(ch_names)
        sfreq: int = eeg.sfreq
        n_samples: int = min([t.shape[1] for t in self.trials])
        epochs_array: np.ndarray = np.stack([t[:, :n_samples] for t in self.trials])
        info = mne.create_info(ch_names, sfreq, ch_types)
        epochs = mne.EpochsArray(epochs_array, info)

        # set montage
        montage = make_standard_montage('standard_1020')
        epochs.set_montage(montage)

        # Apply band-pass filter
        epochs.filter(7., 30., fir_design='firwin', skip_by_annotation='edge', verbose=False)
        self.epochs = epochs

    def _simple_svm(self):
        # Extract spectral features
        data = self.epochs.get_data()
        bands = np.matrix('8 12; 16 22; 30 35')
        fs = self.epochs.info['sfreq']
        bandpower_features = self.extract_bandpower(data, bands, fs)
        self.features_mat = bandpower_features
        self.clf = svm.SVC(decision_function_shape='ovo')
        self.clf.fit(bandpower_features, self.labels)

    @staticmethod
    def extract_bandpower(data: NDArray, bands: np.matrix, fs: int):
        bp_mat_final = pd.DataFrame()
        for band in bands:
            bp_mat = np.zeros((data.shape[0], data.shape[1]))
            fmin = band.item(0)
            fmax = band.item(1)
            f, pxx = scipy.signal.periodogram(data, fs=fs)
            ind_min = scipy.argmax(f > fmin) - 1
            ind_max = scipy.argmax(f > fmax) - 1
            bp_func = lambda power_elec: scipy.trapz(power_elec[ind_min: ind_max], f[ind_min: ind_max])
            for trial in range(data.shape[0]):
                bp_per_elec_per_trial = []
                power = pxx[trial]
                for elec in range(data.shape[1]):
                    bp_per_elec_per_trial.append([bp_func(power[elec])])
                bp_mat[trial] = np.asarray(bp_per_elec_per_trial).T
            bp_concat = pd.DataFrame(bp_mat)
            bp_mat_final = pd.concat([bp_mat_final, bp_concat], axis=1)
        bp_mat_final.transpose().reset_index(drop=True).transpose()
        return bp_mat_final

    def _csp_lda(self):
        print('Training CSP & LDA model')

        # Assemble a classifier
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)

        # Use scikit-learn Pipeline
        self.clf = Pipeline([('CSP', csp), ('LDA', lda)])

        # fit transformer and classifier to data
        self.clf.fit(self.epochs.get_data(), self.labels)

    def online_predict(self, data: NDArray, eeg: EEG):
        # Prepare the data to MNE functions
        data = data.astype(np.float64)

        # Filter the data ( band-pass only)
        data = mne.filter.filter_data(data, l_freq=7, h_freq=30, sfreq=eeg.sfreq, verbose=False)

        # Predict
        prediction = self.clf.predict(data[np.newaxis])[0]

        return prediction

    def cross_val(self):
        self.clf = svm.SVC(kernel='linear', random_state=42)
        scores = cross_val_score(self.clf, self.features_mat, self.labels, cv=2)
        return scores

    def partial_fit(self, eeg, X: NDArray, y: int):

        # Append X to trials
        self.trials.append(X)

        # Append y to labels
        self.labels.append(y)

        # Fit with trials and labels
        self._csp_lda(eeg)

