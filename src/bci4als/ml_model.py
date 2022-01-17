import math
import os
import pickle
from typing import List
import mne
import numpy
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
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler

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
        self.clf = svm.SVC(decision_function_shape='ovo', kernel='linear')  # maybe make more dynamic to user
        self.features_mat = None
        self.epochs = None
        self.raw_trials = None
        self.select_features = None
        self.scaler = StandardScaler()

    def offline_training(self, model_type: str = 'csp_lda'):

        if model_type.lower() == 'csp_lda':
            self._csp_lda()

        elif model_type.lower() == 'simple_svm':
            self._simple_svm()

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
        epochs.filter(1., 40., fir_design='firwin', skip_by_annotation='edge', verbose=False)
        self.epochs = epochs

    def _simple_svm(self):
        # Extract spectral features
        data = self.epochs.get_data()
        bands = np.matrix('8 12; 16 22; 30 35')
        fs = self.epochs.info['sfreq']
        bandpower_features = self.bandpower(data, bands, fs, window_sec=0.9, relative=False)
        hjorth_complexity = self.hjorthMobility(data)
        self.features_mat = np.concatenate((hjorth_complexity, bandpower_features), axis=1)
        # Normalize
        self.scaler.fit(self.features_mat)
        self.scaler.transform(self.features_mat)
        # trial rejection
        self.features_mat = self.trials_rejection(self.features_mat)
        score, feature_num = self.cross_val()  # get best feature number
        # model creation for the online prediction
        self.select_features = SelectKBest(mutual_info_classif, k=feature_num).fit(self.features_mat, self.labels)
        # extract best features
        self.features_mat = self.select_features.transform(self.features_mat)
        # Prepare for online classification
        self.clf.fit(self.features_mat, self.labels)

    @staticmethod
    def trials_rejection(features_mat):
        to_remove = []
        nan_col = np.isnan(features_mat).sum(axis=0)  # remove features with None values
        add_remove = np.where(np.in1d(nan_col, not 0))[0].tolist()
        to_remove += add_remove

        func = lambda x: x > 2  # remove features with extreme values - 2 std over the mean
        Z_bool = func(features_mat).sum(axis=0)
        add_remove = np.where(np.in1d(Z_bool, not 0))[0].tolist()
        to_remove += add_remove
        np.delete(features_mat, to_remove, axis=1)
        return features_mat


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
        data = mne.filter.filter_data(data, l_freq=1, h_freq=40, sfreq=eeg.sfreq, verbose=False)
        # LaPlacian filter
        data, channels_removed = eeg.laplacian(data)
        # maybe make feature extraction static and avoid replicating this shit
        bands = np.matrix('8 12; 16 22; 30 35')
        fs = eeg.sfreq
        bandpower_features = self.bandpower(data[np.newaxis], bands, fs, window_sec=0.9, relative=False)
        hjorth_complexity = self.hjorthMobility(data[np.newaxis])
        # combine features
        features_mat_test = np.concatenate((hjorth_complexity, bandpower_features), axis=0)
        # Normalize
        features_mat_test = self.scaler.transform(features_mat_test[numpy.newaxis])
        # Trials rejection
        # features_mat_test = self.trials_rejection(features_mat_test)
        if self.clf is None:
            self.clf = svm.SVC(decision_function_shape='ovo', kernel='linear')  # maybe make more dynamic to user
            self.clf.fit(self.features_mat, self.labels)  # create new model (not necessary in new recordings)
        # select features on test set
        features_mat_test = self.select_features.transform(features_mat_test)
        # Predict
        prediction = self.clf.predict(features_mat_test)
        return prediction, features_mat_test

    def cross_val(self):
        max_score = 1
        for feat_num in range(1, int(math.sqrt(self.features_mat.shape[0]))):
            features_mat_selected = SelectKBest(mutual_info_classif, k=feat_num).fit_transform(self.features_mat, self.labels)
            scores_mix = cross_val_score(self.clf, features_mat_selected, self.labels, cv=8)
            if np.mean(scores_mix) * 100 > max_score:
                max_score = np.mean(scores_mix) * 100
                feat_num_max = feat_num
        return max_score, feat_num_max

    def partial_fit(self, X: NDArray, y: int, test_features):

        # Append X to trials
        self.trials.append(X)

        # Append y to labels
        self.labels.append(y)

        # update feature mat

        # append to feature mat

        # fit model again

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

    @staticmethod
    def bandpower(data, bands, sf, window_sec=None, relative=False):
        """Compute the average power of the signal x in a specific frequency band.

        Parameters
        ----------
        data : 3d-array
            Input signal in the time-domain. trial X electrode X sample
        sf : float
            Sampling frequency of the data.
        bands : ndarray
            Lower and upper frequencies of the band of interest.
        window_sec : float
            Length of each window in seconds.
            If None, window_sec = (1 / min(band)) * 2
        relative : boolean
            If True, return the relative power (= divided by the total power of the signal).
            If False (default), return the absolute power.

        Return
        ------
        bp : float
            Absolute or relative band power.
        """
        from scipy.signal import welch
        from scipy.integrate import simps
        freq_res = sf/data.shape[2]
        feature_mat = []
        for band in bands:
            band = np.ravel(band)
            low, high = band
            bp_per_elec = []
            bp_per_epoch = []
            # Define window length
            if window_sec is not None:
                nperseg = window_sec * sf
            else:
                nperseg = (2 / low) * sf
            # Compute the modified periodogram (Welch)
            freqs, psd = welch(data, sf, nperseg=nperseg)
            # Find closest indices of band in frequency vector
            idx_band = np.logical_and(freqs >= low, freqs <= high)

            for epoch_idx in range(data.shape[0]):
                for elec_idx in range(data.shape[1]):
                    # Integral approximation of the spectrum using Simpson's rule.
                    bp = simps(psd[epoch_idx][elec_idx][idx_band], dx=freq_res)
                    if relative:
                        bp /= simps(psd[epoch_idx][elec_idx], dx=freq_res)
                    bp_per_elec.append(bp)
                if epoch_idx == 0:
                    bp_per_epoch = bp_per_elec
                else:
                    bp_per_epoch = np.vstack((bp_per_epoch, bp_per_elec))
                bp_per_elec = []
            if all(band == np.ravel(bands[0])):
                feature_mat = bp_per_epoch
            else:
                if isinstance(feature_mat, list) or feature_mat.shape.__len__() == 1:  # for test
                    feature_mat = np.concatenate((feature_mat, bp_per_epoch), axis=0)
                else:
                    feature_mat = np.concatenate((feature_mat, bp_per_epoch), axis=1)
            bp_per_epoch = []

        return feature_mat

    @staticmethod
    def hjorthMobility(data):
        """
        Returns the Hjorth Mobility of the given data

        Parameters
        ----------
        data: array_like

        Returns
        -------
        float
            The resulting value
        """
        bp_per_elec = []
        bp_per_epoch = []
        for epoch_idx in range(data.shape[0]):
            for elec_idx in range(data.shape[1]):
                bp_per_elec.append(np.sqrt(np.var(np.gradient(data[epoch_idx][elec_idx])) / np.var(data[epoch_idx][elec_idx])))
            if epoch_idx == 0:
                bp_per_epoch = bp_per_elec
            else:
                bp_per_epoch = np.vstack((bp_per_epoch, bp_per_elec))
            bp_per_elec = []
        return bp_per_epoch
