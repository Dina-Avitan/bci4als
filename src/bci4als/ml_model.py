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
    This is the essence of the ML side of the project. It contains all the features to be extracted
    and pipeline for the ML.
    The flow is like so:
    offline recording -> * you can add another pipeline but for now theres only one (rf_bandpower)* ->
    -> our pipeline (rf_bandpower).
    after that, you created a MLMODEL object that you can save and use within the online script.
    the important thing is self.clf field which contains the trained model that we will use for prediction
    in the online_predict method
    """

    def __init__(self, trials: List[pd.DataFrame], labels: List[int], channel_removed:List[str], reference_to_baseline=0):
        """
        Constructor for the MLModel class
        Args:
            trials [list]: The trials in a list to be converted into ndarray in the constructor.
            labels [list]: The labels for each trial.
            channel_removed [list[str]]: decides which channels to be removed. we did not use it because we did not like
                -> the idea of removing entire channels based on our analysis. Maybe someone else can find use to it.
            reference_to_baseline [int]: default is 0. how many seconds you wish to record as baseline reference.
                -> this will be used after feature extraction to enhance contrast between trials and baseline.
        """
        self.trials: List[NDArray] = [t.to_numpy().T for t in trials]
        self.labels: List[int] = labels
        self.channel_removed: List[str] = channel_removed
        self.debug = True
        self.features_mat = None
        self.epochs = None
        self.raw_trials = None
        self.select_features = None
        self.scaler = StandardScaler()
        self.ica = ICA(n_components=11, max_iter='auto', random_state=97)
        self.csp_space = []
        self.bands = []
        self.reference_to_baseline = reference_to_baseline  # 0=no reference. positive int for length of baseline recording

    def offline_training(self, model_type: str = 'rf_bandpower'):
        """
        This function decides the pipeline that the model will go through. we currently only use one. if you wish to
            -> experiment with your own pipeline, you can add it here.
        Args:
            model_type: the name of the pipeline you wish your model will go through
        """
        if model_type.lower() == 'rf_bandpower':
            self._rf_bandpower()

        else:
            raise NotImplementedError(f'The model type `{model_type}` is not implemented yet')

    def epochs_extractor(self, eeg: EEG):
        """
        This function is part of the pipeline. It uses the fields of the model to:
        remove bad channels -> create mne.epochs object (for later use) using the trials and save it in self.epochs ->
            -> filter the data -> fit ICA.
        *fitting ICA means that we prepare a object that contains weights and bad components and receives data to
        filter it using those weights and bad components. the idea is that we always save raw data in trials and
        filtered data in epochs. we DO NOT use ica on our saved data. only as part of feature extraction and prediction
        this enables us control over the data for analysis, so we can research on raw data.
        Args:
            eeg: EEG object. we use it to get channel names.
        """
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
        #Save epochs
        self.epochs = epochs
        # Prepare ICA
        self.ica.fit(epochs)
        self.ica.detect_artifacts(epochs)

    def _rf_bandpower(self):
        """
        This function is the entire pipeline. It has parameters and hyper-parameters you can play with. we suggest
        changing them only if you have good reason to believe you have better parameters. basically, it takes the epochs
        and applies ICA, does Laplacian spatial filter, extract features, standardize them and reject outliers. then,
        it saves inside self.clf the trained model over the current data, for online prediction.
        """

        # Set parameters
        data = copy.deepcopy(self.epochs)
        # self.bands = np.matrix('1 4; 7 12; 17 22; 25 40; 1 40')
        self.bands = np.matrix('7 12; 12 15; 17 22; 25 30; 7 35; 30 35') # new
        fs = self.epochs.info['sfreq']
        #Apply ICA
        data = self.ica.apply(data).get_data()
        # Laplacian
        data, _ = EEG.laplacian(data)
        # For early models that did not have this field
        try:
            self.reference_to_baseline
        except AttributeError as err:
            self.reference_to_baseline = 0
        # If reference to baseline is activated
        if self.reference_to_baseline:
            data, baseline = self.baseline_extractor(data=data, fs=fs, baseline_length=self.reference_to_baseline)
            # Get csp_space only for the data and extract features from the baseline
            csp = CSP(n_components=2, reg='ledoit_wolf', log=True, norm_trace=False, transform_into='average_power',
                      cov_est='epoch')
            self.csp_space = Pipeline(
                [('asd', UnsupervisedSpatialFilter(PCA(3), average=True)), ('asdd', csp)]).fit(data, self.labels)

            # Get baseline features
            baseline_csp = self.csp_space.transform(baseline)
            baseline_bp = self.bandpower(data, self.bands, fs, window_sec=0.5, relative=False)
            baseline_bp_rel = self.bandpower(data, self.bands, fs, window_sec=0.5, relative=True)
            baseline_features = np.concatenate((baseline_csp, baseline_bp, baseline_bp_rel), axis=1)
        else:
            csp = CSP(n_components=2, reg='ledoit_wolf', log=True, norm_trace=False, transform_into='average_power',
                      cov_est='epoch')
            self.csp_space = Pipeline(
                [('asd', UnsupervisedSpatialFilter(PCA(3), average=True)), ('asdd', csp)]).fit(data, self.labels)

        # Get CSP features
        csp_features = self.csp_space.transform(data)
        # Extract spectral features
        bandpower_features = self.bandpower(data, self.bands, fs, window_sec=0.5, relative=False)
        bandpower_features_rel = self.bandpower(data, self.bands, fs, window_sec=0.5, relative=True)

        self.features_mat = np.concatenate((csp_features, bandpower_features_rel, bandpower_features), axis=1)
        # Reference the features in respect to the baseline if baseline reference is activated
        self.features_mat = np.divide(self.features_mat, baseline_features) if self.reference_to_baseline else self.features_mat
        # Normalize
        self.scaler.fit(self.features_mat)
        self.features_mat = self.scaler.transform(self.features_mat)
        # trial rejection
        self.features_mat, self.labels, to_remove = self.trials_rejection(self.features_mat, self.labels)
        self.epochs.drop(to_remove)
        # pick classifier
        rf_classifier = RandomForestClassifier()
        # Select pipeline
        pipeline_RF = Pipeline([('classify', rf_classifier)])
        # Initiate Pipeline for online classification
        self.clf = pipeline_RF.fit(self.features_mat, self.labels)

    @staticmethod
    def trials_rejection(feature_mat, labels):
        """
        Our outlier rejection method. It might be basic but we found it was very viable. It remove features with nans,
        and features with above-threshold std.
        we found that usually, specific features tend to return nans, and not specific trials. keep in mind that
        specific feature correspond with specific channels so it makes sense.
        Args:
            feature_mat:
            labels:

        Returns: feature_mat, labels, to_remove. to_remove is a list with the index of the epochs to drop.
        """
        to_remove = []
        nan_col = np.isnan(feature_mat).sum(axis=1)  # remove features with None values
        add_remove = np.where(np.in1d(nan_col, not 0))[0].tolist()
        to_remove += add_remove

        func = lambda x: np.mean(np.abs(x),axis=1) > 1.2  # remove features with extreme values - x std over the mean
        Z_bool = func(feature_mat)
        add_remove = np.where(np.in1d(Z_bool, not 0))[0].tolist()
        to_remove += add_remove
        feature_mat = np.delete(feature_mat, to_remove, axis=0)
        labels = np.delete(labels, to_remove, axis=0)
        print(f"trial rejected: {to_remove}")
        return feature_mat, labels, to_remove

    def online_predict(self, data: NDArray, eeg: EEG, baseline=NDArray):
        """
        This function receives the data from the online script. It will extract the same features and predict the label.
        Args:
            data: [ndarray] : online data. single trial.
            eeg: EEG object.
            baseline: [int]. amount of seconds that are the baseline.
        """
        # Prepare parameters
        fs = eeg.sfreq
        # Get features
        csp_features = self.csp_space.transform(data)  # old
        #Spectral features
        bandpower_features = self.bandpower(data, self.bands, fs, window_sec=0.5, relative=False)
        bandpower_features_rel = self.bandpower(data, self.bands, fs, window_sec=0.5, relative=True)
        # Get baseline features
        if baseline.shape[-1]:
            baseline_csp = self.csp_space.transform(baseline)
            baseline_bp = self.bandpower(baseline, self.bands, fs, window_sec=0.5, relative=False)
            baseline_bp_rel = self.bandpower(baseline, self.bands, fs, window_sec=0.5, relative=True)
            baseline_features_mat = np.concatenate((np.squeeze(baseline_csp), baseline_bp, baseline_bp_rel), axis=0)

        # combine features
        features_mat_test = np.concatenate((np.squeeze(csp_features), bandpower_features, bandpower_features_rel), axis=0)
        # re-reference features in respect to baseline: (features)/(baseline_features) - element-wise
        features_mat_test = np.divide(features_mat_test, baseline_features_mat) if baseline.shape[-1] else features_mat_test
        # Normalize
        features_mat_test = self.scaler.transform(features_mat_test[numpy.newaxis])
        # Predict
        prediction = self.clf.predict(features_mat_test)
        return prediction

    def partial_fit(self, X, y, epochs, sfreq):
        """
        This function receives the data from the online recording and concatenates it to the existing data, in a
        stratified manner, to add the current data to the model. this is essentially known as co-adaptive learning.
        It will then fit the model with the new data.
        Args:
            X: [list[ndarray]] a list that contains 3 recordings from each trial type(left/right/etc.)
            y: [list[int]] a list that contains 3 labels from each trial
            epochs: the epochs of those trials.
            sfreq: the sample frequency rate.

        Returns:

        """
        # Append X to trials
        [self.trials.append(trial) for trial in X]
        # Append y to labels
        for label in y:
            self.labels = np.append(self.labels, label)

        # update self.epochs
        ch_types = ['eeg'] * len(epochs.ch_names)
        info = mne.create_info(epochs.ch_names, sfreq, ch_types)
        temp_epoch = copy.deepcopy(self.epochs.get_data())
        for trial in X:
            # band-aid for weird bug that sometimes causes the online recording to be slightly shorter than the offline model.
            while trial.shape[1] < temp_epoch.shape[2]:
                trial = np.concatenate((trial,np.reshape(np.mean(trial[:,-10:-1],axis=1),[-1,1])),axis=1)
                if trial.shape[1] == temp_epoch.shape[2]:
                    print("data recorded was too short. padding with mean to compensate")
            temp_epoch = np.concatenate((temp_epoch, trial[np.newaxis]))
        self.epochs = mne.EpochsArray(temp_epoch, info)
        del temp_epoch
        # update feature mat and fit model
        self._rf_bandpower()

    @staticmethod
    def baseline_extractor(data, fs, baseline_length):
        """
        This funciton extracts the baseline from the total recording
        Args:
            data [ndarray]: the whole data: baseline and then the recording together in a single ndarray.
            fs [int]: sample frequency rate.
            baseline_length [int]: the baseline length in seconds.

        Returns: the baseline and the trial recording seperated. both in ndarrays.
            data, baseline
        """
        baseline = data[:, :, :int(fs*baseline_length)]
        data = data[:, :, int(fs*baseline_length)::]
        return data, baseline

    # From here downwards are features we used

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
    # HjorthParameters
    def hjorthActivity(data):
        """
        Returns the Hjorth Activity of the given data

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
                bp_per_elec.append(np.var(data[epoch_idx][elec_idx]))
            if epoch_idx == 0:
                bp_per_epoch = bp_per_elec
            else:
                bp_per_epoch = np.vstack((bp_per_epoch, bp_per_elec))
            bp_per_elec = []
        return bp_per_epoch

    @staticmethod
    def hjorthComplexity(data):
        """
        Returns the Hjorth Complexity of the given data

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
                bp_per_elec.append(MLModel.hjorthMobility(np.gradient(data[epoch_idx][elec_idx])[np.newaxis][np.newaxis]
                                    )[0] / MLModel.hjorthMobility(data[epoch_idx][elec_idx][np.newaxis][np.newaxis])[0])
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
