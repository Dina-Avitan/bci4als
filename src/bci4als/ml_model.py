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

    def offline_training(self, eeg: EEG, model_type: str = 'csp_lda'):

        if model_type.lower() == 'csp_lda':
            self._csp_lda(eeg)

        elif model_type.lower() == 'simple_svm':
            self._simple_svm(eeg)

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
        return epochs

    def _simple_svm(self, eeg: EEG):
        epochs = self.epochs_extractor(eeg)

        #Extract spectral features
        data = epochs.get_data()
        bands = np.matrix('8 12; 16 22; 30 35')
        fs = epochs.info['sfreq']
        bandpower_features = self.extract_bandpower(data, bands, fs)
        return bandpower_features

    def extract_bandpower(self, data: NDArray,bands: np.matrix, fs: int):
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
            pd.concat([bp_mat_final, bp_concat], axis=1)
        pickle.dump(bp_mat_final, open(os.path.join('', 'features.pickle'), 'wb'))
        return bp_mat_final

    def _csp_lda(self, eeg: EEG):

        print('Training CSP & LDA model')

        # Extract epochs
        epochs = self.epochs_extractor(eeg)

        # Assemble a classifier
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)

        # Use scikit-learn Pipeline
        self.clf = Pipeline([('CSP', csp), ('LDA', lda)])

        # fit transformer and classifier to data
        self.clf.fit(epochs.get_data(), self.labels)

    def online_predict(self, data: NDArray, eeg: EEG):
        # Prepare the data to MNE functions
        data = data.astype(np.float64)

        # Filter the data ( band-pass only)
        data = mne.filter.filter_data(data, l_freq=8, h_freq=30, sfreq=eeg.sfreq, verbose=False)

        # Predict
        prediction = self.clf.predict(data[np.newaxis])[0]

        return prediction

    def partial_fit(self, eeg, X: NDArray, y: int):

        # Append X to trials
        self.trials.append(X)

        # Append y to labels
        self.labels.append(y)

        # Fit with trials and labels
        self._csp_lda(eeg)

