import copy
import math
import os
import pickle
from typing import List
import mne
import numpy
import pandas as pd
from mne.preprocessing import ICA
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from bci4als.eeg import EEG
import numpy as np
from matplotlib.figure import Figure
from mne.channels import make_standard_montage
from mne.decoding import CSP, UnsupervisedSpatialFilter
from nptyping import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
import scipy
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel, SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from numba import njit


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
        self.clf = RandomForestClassifier(random_state=0)  # maybe make more dynamic to user
        self.features_mat = None
        self.epochs = None
        self.raw_trials = None
        self.select_features = None
        self.scaler = StandardScaler()
        self.ica = ICA(n_components=11, max_iter='auto', random_state=97)
        self.csp_space = None


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
        print(eeg.sfreq)
        n_samples: int = min([t.shape[1] for t in self.trials])
        epochs_array: np.ndarray = np.stack([t[:, :n_samples] for t in self.trials])
        info = mne.create_info(ch_names, sfreq, ch_types)
        epochs = mne.EpochsArray(epochs_array, info)

        # set montage
        montage = make_standard_montage('standard_1020')
        epochs.set_montage(montage)

        # Apply band-pass filter
        epochs.filter(40., 1., fir_design='firwin', skip_by_annotation='edge', verbose=False)
        #Save epochs
        self.epochs = epochs
        # Prepare ICA
        self.ica.fit(epochs)
        self.ica.detect_artifacts(epochs)

    def _simple_svm(self):
        """
        This function will re-learn the model's feature mat and clf object which represents the model itself
        """

        # Extract spectral features
        data = copy.deepcopy(self.epochs)
        bands = np.matrix('7 12; 12 15; 17 22; 25 30; 7 35; 30 35')
        fs = self.epochs.info['sfreq']
        #Apply ICA
        data = self.ica.apply(data).get_data()
        # Laplacian
        data, _ = EEG.laplacian(data)
        # Get features
        bandpower_features = self.bandpower(data, bands, fs, window_sec=0.5, relative=False)
        bandpower_features_rel = self.bandpower(data, bands, fs, window_sec=0.5, relative=True)

        # Get CSP features
        csp = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False, transform_into='average_power',
                  cov_est='epoch')
        self.csp_space = Pipeline(
            [('asd', UnsupervisedSpatialFilter(PCA(3), average=True)), ('asdd', csp)]).fit(data, self.labels)

        csp_features = self.csp_space.transform(data)

        self.features_mat = np.concatenate((csp_features, bandpower_features_rel, bandpower_features), axis=1)
        # Normalize
        self.scaler.fit(self.features_mat)
        self.features_mat = self.scaler.transform(self.features_mat)
        # trial rejection
        self.features_mat, self.labels = self.trials_rejection(self.features_mat, self.labels)
        # pick classifier
        clf = svm.SVC(decision_function_shape='ovo', kernel='linear')
        rf_classifier = RandomForestClassifier(random_state=0)
        # Prepare Pipeline
        mi_select = SelectKBest(mutual_info_classif, k=int(math.sqrt(self.features_mat.shape[0])))
        model = SelectFromModel(LogisticRegression(C=1, penalty="l1", solver='liblinear', random_state=0))
        seq_select_clf = SequentialFeatureSelector(clf, n_features_to_select=int(math.sqrt(self.features_mat.shape[0])), n_jobs=1)
        # Select pipeline
        pipeline_SVM = Pipeline([('lasso', model), ('feat_selecting', seq_select_clf), ('SVM', clf)])
        pipeline_RF = Pipeline([('lasso', model), ('feat_selecting', mi_select), ('classify', rf_classifier)])
        # Initiate Pipeline for online classification
        self.clf = pipeline_RF.fit(self.features_mat, self.labels)

    @staticmethod
    def trials_rejection(feature_mat, labels):
        to_remove = []
        nan_col = np.isnan(feature_mat).sum(axis=1)  # remove features with None values
        add_remove = np.where(np.in1d(nan_col, not 0))[0].tolist()
        to_remove += add_remove

        func = lambda x: np.mean(np.abs(x),axis=0) > 1.8  # remove features with extreme values - 2 std over the mean
        Z_bool = func(feature_mat)
        add_remove = np.where(np.in1d(Z_bool, not 0))[0].tolist()
        to_remove += add_remove
        feature_mat = np.delete(feature_mat, to_remove, axis=0)
        labels = np.delete(labels, to_remove, axis=0)
        return feature_mat, labels

    def online_predict(self, data: NDArray, eeg: EEG):
        # Prepare parameters
        bands = np.matrix('7 12; 12 15; 17 22; 25 30; 7 35; 30 35')
        fs = eeg.sfreq
        # Get features
        csp_features = self.csp_space.transform(data[np.newaxis])[0]
        bandpower_features = self.bandpower(data[np.newaxis], bands, fs, window_sec=0.5, relative=False)
        bandpower_features_rel = self.bandpower(data[np.newaxis], bands, fs, window_sec=0.5, relative=True)
        # combine features
        features_mat_test = np.concatenate((csp_features, bandpower_features_rel, bandpower_features), axis=0)
        # Normalize
        features_mat_test = self.scaler.transform(features_mat_test[numpy.newaxis])
        # Predict
        prediction = self.clf.predict(features_mat_test)
        return prediction, features_mat_test

    def partial_fit(self, X, y, epochs, sfreq):
        # Append X to trials
        [self.trials.append(trial) for trial in X]
        # Append y to labels
        print(y)
        print(self.labels)
        print(type(y))
        for label in y:
            self.labels = np.append(self.labels, label)
        # update self.epochs
        ch_types = ['eeg'] * len(epochs.ch_names)
        info = mne.create_info(epochs.ch_names, sfreq, ch_types)
        temp_epoch = copy.deepcopy(self.epochs.get_data())
        for trial in X:
            temp_epoch = np.concatenate((temp_epoch, trial[np.newaxis]))
        self.epochs = mne.EpochsArray(temp_epoch, info)
        del temp_epoch
        # update feature mat and fit model
        self._simple_svm()

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

    @staticmethod
    # Lempel-Ziv Complexity
    def LZC(data, threshold=None):
        """
        Returns the Lempel-Ziv Complexity (LZ76) of the given data.

        Parameters
        ----------
        data: array_like
            The signal.
        theshold: numeric, optional
            A number use to binarize the signal. The values of the signal above
            threshold will be converted to 1 and the rest to 0. By default, the
            median of the data.

        References
        ----------
        .. [1] M. Aboy, R. Hornero, D. Abasolo and D. Alvarez, "Interpretation of
               the Lempel-Ziv Complexity Measure in the Context of Biomedical
               Signal Analysis," in IEEE Transactions on Biomedical Engineering,
               vol. 53, no.11, pp. 2282-2288, Nov. 2006.
        """

        lzc_per_elec = []
        lzc_per_epoch = []

        for epoch_idx in range(data.shape[0]):
            for elec_idx in range(data.shape[1]):
                if not threshold:
                    threshold = np.median(data[epoch_idx, elec_idx])
                n = len(data[epoch_idx, elec_idx])
                sequence = MLModel._binarize(data[epoch_idx, elec_idx], threshold)
                n_seq = len(sequence)
                complexity = 1
                q0 = 1
                qSize = 1
                sqi = 0
                where = 0
                while q0 + qSize < n_seq:
                    # If we are checking the end of the sequence we just need to look at
                    # the last element
                    if sqi != q0 - 1:
                        contained, where = MLModel._isSubsequenceContained(sequence[q0:q0 + qSize],
                                                                           sequence[sqi:q0 + qSize - 1])
                    else:
                        contained = sequence[q0 + qSize] == sequence[q0 + qSize - 1]

                    # If Q is contained in sq~, we increase the size of q
                    if contained:
                        qSize += 1
                        sqi = where
                    # If Q is not contained the complexity is increased by 1 and reset Q
                    else:
                        q0 += qSize
                        qSize = 1
                        complexity += 1
                        sqi = 0
                b = n / np.log2(n)
                lzc_per_elec.append(complexity / b)
            if epoch_idx == 0:
                lzc_per_epoch = lzc_per_elec
            else:
                lzc_per_epoch = np.vstack((lzc_per_epoch, lzc_per_elec))
            lzc_per_elec = []
        return lzc_per_epoch

    @staticmethod
    def _binarize(data, threshold):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        return np.array(data > threshold, np.uint8)

    @staticmethod
    @njit
    def _isSubsequenceContained(subSequence, sequence):  # pragma: no cover
        """
        Checks if the subSequence is into the sequence and returns a tuple that
        informs if the subsequence is into and where. Return examples: (True, 7),
        (False, -1).
        """
        n = len(sequence)
        m = len(subSequence)

        for i in range(n - m + 1):
            equal = True
            for j in range(m):
                equal = subSequence[j] == sequence[i + j]
                if not equal:
                    break

            if equal:
                return True, i

        return False, -1

    @staticmethod
    def DFA(data, fit_degree=1, min_window_size=4, max_window_size=None,
            fskip=1, max_n_windows_sizes=None):
        """
        Applies Detrended Fluctuation Analysis algorithm to the given data.

        Parameters
        ----------
        data: array_like
            The signal.
        fit_degree: int, optional
            Degree of the polynomial used to model de local trends. Default: 1.
        min_window_size: int, optional
            Size of the smallest window that will be used. Default: 4.
        max_window_size: int, optional
            Size of the biggest window that will be used. Default: signalSize//4
        fskip: float, optional
            Fraction of the window that will be skiped in each iteration for each
            window size. Default: 1
        max_n_windows_sizes: int, optional
            Maximum number of window sizes that will be used. The final number can
            be smaller once the repeated values are removed
            Default: log2(size)

        Returns
        -------
        float
            The resulting value
        """
        # Arguments handling
        dfa_per_elec = []
        dfa_per_epoch = []
        for epoch_idx in range(data.shape[0]):
            for elec_idx in range(data.shape[1]):
                size = len(data[epoch_idx, elec_idx])
                if not max_window_size:
                    max_window_size = size // 4

                # Detrended data
                Y = np.cumsum(data - np.mean(data[epoch_idx, elec_idx]))

                # Windows sizes
                if not max_n_windows_sizes:
                    max_n_windows_sizes = int(np.round(np.log2(size)))

                ns = np.unique(
                    np.geomspace(min_window_size, max_window_size, max_n_windows_sizes,
                                 dtype=int))

                # Fluctuations for each window size
                F = np.zeros(ns.size)

                # Loop for each window size
                for indexF, n in enumerate(ns):
                    itskip = max(int(fskip * n), 1)
                    nWindows = int(np.ceil((size - n + 1) / itskip))

                    # Aux x
                    x = np.arange(n)

                    y = np.array([Y[i * itskip:i * itskip + n] for i in range(0, nWindows)])
                    c = np.polynomial.polynomial.polyfit(x, y.T, fit_degree)
                    yn = np.polynomial.polynomial.polyval(x, c)

                    F[indexF] = np.mean(np.sqrt(np.sum((y - yn) ** 2, axis=1) / n))

                alpha = np.polyfit(np.log(ns), np.log(F), 1)[0]

                if np.isnan(alpha):  # pragma: no cover
                    alpha = 0
                dfa_per_elec.append(alpha)
            if epoch_idx == 0:
                dfa_per_epoch = dfa_per_elec
            else:
                dfa_per_epoch = np.vstack((dfa_per_epoch, dfa_per_elec))
            dfa_per_elec = []
        return dfa_per_epoch
