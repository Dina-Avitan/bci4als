# This script is meant to load models and allow the user to change hyper-parameters
# so you could fine-tune the real offline_training class
import copy
import random
import os
import time

import lightgbm as lgb
import math
from tkinter import filedialog, Tk
import mne
import niapy.problems
import scipy.io
import seaborn
import sklearn.feature_selection
from matplotlib.colors import ListedColormap
import json
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, SelectFromModel
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from skfeature.function.similarity_based import fisher_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from mne.decoding import UnsupervisedSpatialFilter
from bci4als import ml_model, EEG
from sklearn import svm, manifold
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from mne.preprocessing import ICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization, DifferentialEvolution
from tqdm import tqdm

class SVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        accuracy = cross_val_score(lgb.LGBMClassifier(), self.X_train[:, selected], self.y_train, cv=2, n_jobs=-1).mean()
        score = 1 - accuracy
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)
#import eeglib.eeg #steal features
def load_eeg(path,data=None,labels=None, method='baseline'):
    """This function is very very messy. its sole purpose is to be messy. It is meant to try and play with the data
    and to test for different ideas. change it and use it as you wish."""
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
        ica = ICA(n_components=11, max_iter='auto', random_state=0)
        ica.fit(epochs)
        ica.plot_sources(epochs, start=15, stop=20, show_scrollbars=False, title='ICA components')
        ica.plot_components(title='ICA components-topoplot')
        to_exclude = input("\nEnter a list of the numbers of the components to exclude: ")
        to_exclude = to_exclude.strip(']')
        to_exclude = [int(i) for i in to_exclude.strip('[').split(',')]
        if to_exclude:
            ica.exclude = to_exclude
        ica.apply(epochs)
        data.plot(scalings=10, title='Before ICA')
        epochs.plot(scalings=10, title='After ICA')
        # before = epochs_to_raw(data)
        # after=epochs_to_raw(epochs)
        # before.plot(scalings=10)
        # after.plot(scalings=10)
    def ICA_perform(model):
        """
        Args:
            model: the model before ICA transform
            to_exclude: (list) list of the coordinates numbers to exclude

        Returns: epochs array after ICA transform
        """
        epochs = model.epochs
        ica = ICA(n_components=11, max_iter='auto', random_state=97)
        ica.fit(epochs)
        # ica.exclude = [0,1]
        ica.detect_artifacts(epochs)
        ica.apply(epochs)
        return epochs
    def trials_rejection(feature_mat, labels):
        """
        Trial rejection, he same as the function in the MLMODEL, but he parameters here might be different
        because we tried different std's.
        Args:
            feature_mat:
            labels:

        Returns:

        """
        to_remove = []
        features_remove = []
        add_remove = list(set([column[1] for column in np.argwhere(np.isnan(feature_mat))]))
        features_remove += add_remove

        func = lambda x: np.mean(np.abs(x),axis=1) > 1.2 # remove features with extreme values - 2 std over the mean
        Z_bool = func(feature_mat)
        add_remove = np.where(np.in1d(Z_bool, not 0))[0].tolist()
        to_remove += add_remove
        feature_mat = np.delete(feature_mat, to_remove, axis=0)
        feature_mat = np.delete(feature_mat, features_remove, axis=1)
        labels = np.delete(labels, to_remove, axis=0)
        print(f'trials removed: {to_remove}')
        return feature_mat, labels
    def extract_2_labels(data, labels, classes_to_extract):
        """
        This function extracts the classes you choose from the given data
        Args:
            data: ndarrray of the data
            labels: list of the labels
            classes_to_extract: list[int[]] : a list of integers which represents the calsses in the daa you want
            to extract

        Returns:
            data, labels (with only the desired classes)
        """
        data_2_extract = [data[ind] for ind, label in enumerate(labels) if labels[ind] in classes_to_extract]
        labels_2_extract = [label for ind, label in enumerate(labels) if labels[ind] in classes_to_extract]
        return np.reshape(data_2_extract,[len(data_2_extract)] + list(data_2_extract[0].shape)), np.ravel(labels_2_extract)
    def band_hunter(data, labels,fs):
        rf_classifier = RandomForestClassifier(random_state=0)
        epochs = 0
        max_pred = 0
        winning_band = []
        lower = 0.1
        upper = 40
        band_number = 5
        delta = 3
        while epochs <= 5:
            band_range = np.round(np.linspace(lower, upper, band_number),2)
            bands = np.matrix(';'.join([str(i) + ' ' + str(i + delta) for i in band_range]))
            pipeline_RF = Pipeline([('classify', rf_classifier)])
            # Get rest of features
            bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
            bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
            csp = CSP(n_components=3, reg='ledoit_wolf', log=True, norm_trace=False, transform_into='average_power',
                      cov_est='epoch')
            csp_features = Pipeline(
                [('asd', UnsupervisedSpatialFilter(PCA(3), average=True)), ('asdd', csp)]).fit_transform(data,
                                                                                                         labels)

            features = np.concatenate((csp_features,bandpower_features_rel,bandpower_features_new), axis=1)
            scaler = StandardScaler()
            scaler.fit(features)
            features = scaler.transform(features)
            # Trial rejection
            features, labels_curr = features, labels # trials_rejection(features, labels)
            scores_mix = np.mean(cross_val_score(pipeline_RF, features, labels_curr, cv=3, n_jobs=1)) * 100
            if max_pred < scores_mix:
                max_pred = scores_mix
                winning_band = bands
                band_number = band_number + 2
                delta = np.linspace(lower, upper, band_number)[1] - np.linspace(lower, upper, band_number)[0]
            else:
                lower = np.random.randint(0.1, 40)
                upper = 40 if lower+7 >= 40 else np.random.randint(lower+7, 40)
                delta = np.linspace(lower, upper, band_number)[1] - np.linspace(lower, upper, band_number)[0]
                band_number = band_number -1
            band_number = band_number if band_number > 3 else np.random.randint(3,20)
            delta = delta if delta > 2 else np.random.uniform(low=2.5,high=8,size=1)
            epochs += 1
        return max_pred, winning_band
    def band_hunter_swarm_algorithm(data,labels,fs):
        epochs = 50
        max_pred = 0
        winning_band = []
        band_number = random.randint(4, 10)
        lower = [np.random.randint(0.1, 40) for _ in range(band_number)]
        delta = np.multiply([np.abs(random.gauss(0.8, 0.1)) for _ in range(band_number)], [random.randint(3, 10) for _ in
                                                                       range(band_number)])
        bands_to_keep = []
        def get_band_from_string(str):
            marker = False
            ans = ''
            for letter in str:
                if marker and not letter == str[-1]:
                    ans += letter
                if letter == '[':
                    marker = True
            # ans = [float(to_integer) for to_integer in str.split(ans)] if ans else ''
            return ans
        def band_str_to_int(band_str):
            return [float(to_integer) for to_integer in str.split(band_str)] if band_str else ''
        for _ in tqdm(range(epochs)):
            curr_time = time.time()
            feature_labels = []
            counter = 0
            no_err = ''
            while not no_err:
                try:
                    bands_in_list = [[np.round(lower_sc,2), np.round(lower_sc + delta_sc,2)] for (lower_sc,delta_sc) in zip(lower,delta)]
                    # combine old bands with new bands
                    bands = bands_in_list + bands_to_keep
                    bands = np.matrix(bands)
                    # Get rest of features
                    bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
                    [feature_labels.append(f'BP_non_rel{np.ravel(i)}') for i in bands for _ in range(np.size(data,1))]# for chan in model.epochs.ch_names]
                    bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
                    [feature_labels.append(f'BP_rel{np.ravel(i)}') for i in bands for _ in range(np.size(data,1))]# for chan in model.epochs.ch_names]
                    no_err = 'all good'
                except IndexError:
                    counter += 1
                    delta += 0.5
                    print(counter)
                    print(f'error. trying to increase delta ({delta}). loop number {counter}')
            csp = CSP(n_components=2, reg='ledoit_wolf', log=True, norm_trace=False, transform_into='average_power',
                      cov_est='epoch')
            csp_features = Pipeline(
                [('asd', UnsupervisedSpatialFilter(PCA(3), average=True)), ('asdd', csp)]).fit_transform(data,
                                                                                                         labels)
            [feature_labels.append(f'CSP_Component{i}') for i in range(csp_features.shape[1])]
            features = np.concatenate((csp_features,bandpower_features_rel,bandpower_features_new), axis=1)
            problem = SVMFeatureSelection(features, labels)
            task = Task(problem, max_iters=5)
            algorithm = DifferentialEvolution()
            best_features, best_fitness = algorithm.run(task)
            features_to_keep = set([feature for ind, feature in enumerate(feature_labels) if (best_features > 0.95)[ind]])
            bands_to_keep_str = set([get_band_from_string(good_band) for good_band in features_to_keep if get_band_from_string(good_band)])
            bands_to_keep = [band_str_to_int(good_band) for good_band in bands_to_keep_str]
            if max_pred < best_fitness:
                max_pred = best_fitness
                winning_band = bands_to_keep
                band_number = band_number + 2
                delta = np.concatenate([delta*random.gauss(0.9, 0.1),
                                        np.ones(2) * np.abs(random.gauss(0, 0.7)) * [random.randint(4, 10) for _ in
                                                                               range(2)]])
            else:
                band_number = band_number - 1
                lower = [np.random.randint(0.1, 40) for _ in range(band_number)]
                delta = np.ones(band_number)*np.abs(random.gauss(0,0.7))*[random.randint(3,10) for _ in range(band_number)]
            if not band_number > 3:
                band_number = random.randint(4,10)
                lower = [np.random.randint(0.1, 40) for _ in range(band_number)]
                delta = np.ones(band_number)*np.abs(random.gauss(0,0.7))*[random.randint(3,10) for _ in range(band_number)]
            end_time = time.time()
            print(f'TIME: {end_time - curr_time}')
            with open('json_data.json', 'w') as outfile:
                outfile.write(f'{bands_to_keep}')
        return max_pred, winning_band
    def get_features(data, labels, fs):
        """
        This funciton returns feature matrix containing most of the features that we assumed that made sense.
        In the end of the day, it is slower and performs similarly to the minimalist version and requires feature
        selection to outperform the minimalist features.
        Args:
            data: ndarrray of the data
            labels: list of the labels
            fs: frequency rate

        Returns:
            Features matrix in NDARRAY
        """
        csp = CSP(n_components=2, reg='ledoit_wolf', log=True, norm_trace=False, transform_into='average_power',
                  cov_est='epoch')
        csp_features = Pipeline(
            [('asd', UnsupervisedSpatialFilter(PCA(3), average=True)), ('asdd', csp)]).fit_transform(data,
                                                                                                     labels)

        ## trying csp space
        bandpower_features_wtf = []
        csp = CSP(n_components=2, reg='ledoit_wolf', norm_trace=False, transform_into='csp_space',
                  cov_est='epoch')
        csp_pwer = CSP(n_components=2, reg='ledoit_wolf', norm_trace=False, transform_into='average_power',
                  cov_est='epoch')
        for i in bands:
            data_mu = mne.filter.filter_data(copy.copy(data), fs, i[0, 0], i[0, 1])
            csp_space = Pipeline(
                [('asd', UnsupervisedSpatialFilter(PCA(3), average=True)), ('asdd', csp)]).fit_transform(data_mu,
                                                                                                         labels)
            csp_feats = Pipeline(
            [('asd', UnsupervisedSpatialFilter(PCA(3), average=True)), ('asdd', csp_pwer)]).fit_transform(data_mu,
                                                                                                     labels)
            # Get hjorth parameters from every csp space across every band
            hjorthMobility_features = ml_model.MLModel.hjorthMobility(csp_space)
            hjorthMobility_features2 = ml_model.MLModel.hjorthActivity(csp_space)
            hjorthMobility_features3 = ml_model.MLModel.hjorthComplexity(csp_space)
            features = [hjorthMobility_features, hjorthMobility_features2, hjorthMobility_features3, csp_feats]
            if type(bandpower_features_wtf) is not list:
                features = [bandpower_features_wtf] + features
                bandpower_features_wtf = np.concatenate(features, axis=1)
            else:
                bandpower_features_wtf = np.concatenate(tuple(features), axis=1)

        # Get rest of features
        # Get bandpower from the data in original space
        bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
        bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
        csp_space = Pipeline(
            [('asd', UnsupervisedSpatialFilter(PCA(3), average=True)), ('asdd', csp)]).fit_transform(data,
                                                                                                     labels)
        # Get bandpower from the data in csp space
        bandpower_features_new_csp = ml_model.MLModel.bandpower(csp_space, bands, fs, window_sec=0.5, relative=False)
        bandpower_features_rel_csp = ml_model.MLModel.bandpower(csp_space, bands, fs, window_sec=0.5, relative=True)
        bandpower_features_wtf = np.concatenate(
            (csp_features, bandpower_features_wtf, bandpower_features_new, bandpower_features_rel,
             bandpower_features_new_csp, bandpower_features_rel_csp), axis=1)
        return bandpower_features_wtf
    def get_features_with_baseline(data,baseline,  labels, fs):
        """
        Extract the features in relation to the baseline. It essentially extracts the same features from both the
        data sets and then divides the trial data by the baseline data. The idea is that this will make the effect
        more significant. We did not have the chance to play with it too much but we found to improves the results
        slightly
        Args:
            data: ndarrray of the data
            baseline: ndarrray of the baseline
            labels: list of the labels
            fs: frequency rate

        Returns:

        """
        csp = CSP(n_components=3, reg='ledoit_wolf', log=True, norm_trace=False, transform_into='average_power',
                  cov_est='epoch')
        csp_features = Pipeline(
            [('asd', UnsupervisedSpatialFilter(PCA(3), average=True)), ('asdd', csp)]).fit_transform(data,
                                                                                                     labels)
        csp_features_base = Pipeline(
            [('asd', UnsupervisedSpatialFilter(PCA(3), average=True)), ('asdd', csp)]).fit_transform(baseline,
                                                                                                     labels)
        ## trying csp space
        data_features = []
        baseline_features = []
        csp = CSP(n_components=2, reg='ledoit_wolf', norm_trace=False, transform_into='csp_space',
                  cov_est='epoch')
        # for i in bands:
        #     data_mu = mne.filter.filter_data(copy.copy(data), fs, i[0, 0], i[0, 1])
        #     baseline_mu = mne.filter.filter_data(copy.copy(baseline), fs, i[0, 0], i[0, 1])
        #     csp_space = Pipeline(
        #         [('asd', UnsupervisedSpatialFilter(PCA(3), average=True)), ('asdd', csp)]).fit(data_mu,labels)
        #     data_and_baseline_in_csp_space = [csp_space.transform(data_mu),csp_space.transform(baseline_mu)]
        #     for ind, thing_in_csp_space in enumerate(data_and_baseline_in_csp_space):
        #         hjorthMobility_features = ml_model.MLModel.hjorthMobility(thing_in_csp_space)
        #         hjorthMobility_features2 = ml_model.MLModel.hjorthActivity(thing_in_csp_space)
        #         hjorthMobility_features3 = ml_model.MLModel.hjorthComplexity(thing_in_csp_space)
        #         bandpower_features_new = ml_model.MLModel.bandpower(thing_in_csp_space, np.matrix([1, 100]), fs, window_sec=0.5,
        #                                                             relative=False)
        #         bandpower_features_rel = ml_model.MLModel.bandpower(thing_in_csp_space, np.matrix([1, 100]), fs, window_sec=0.5,
        #                                                             relative=True)
        #         LZC_features = ml_model.MLModel.LZC(thing_in_csp_space)
        #         features = [hjorthMobility_features, hjorthMobility_features2]
        #         if ind == 0:
        #             if type(data_features) is not list:
        #                 features = [data_features] + features
        #                 data_features = np.concatenate(features, axis=1)
        #             else:
        #                 data_features = np.concatenate(tuple(features), axis=1)
        #         else:
        #             if type(baseline_features) is not list:
        #                 features = [baseline_features] + features
        #                 baseline_features = np.concatenate(features, axis=1)
        #             else:
        #                 baseline_features = np.concatenate(tuple(features), axis=1)
        # Get rest of features
        bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
        bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
        bandpower_features_new_base = ml_model.MLModel.bandpower(baseline, bands, fs, window_sec=0.5, relative=False)
        bandpower_features_rel_base = ml_model.MLModel.bandpower(baseline, bands, fs, window_sec=0.5, relative=True)
        # hjorthMobility_features = ml_model.MLModel.hjorthMobility(data)
        # hjorthMobility_features2 = ml_model.MLModel.hjorthActivity(data)
        # hjorthMobility_features3 = ml_model.MLModel.hjorthComplexity(data)

        # LZC_features = ml_model.MLModel.LZC(data)
        # DFA_features = ml_model.MLModel.DFA(data)
        data_features = np.concatenate(
            (csp_features, bandpower_features_new, bandpower_features_rel), axis=1)
        baseline_features = np.concatenate(
            (csp_features_base, bandpower_features_new_base, bandpower_features_rel_base), axis=1)
        return data_features, baseline_features
    def get_current_features(data, labels, fs):
        """
        This function extracts the minimalist version of our features. Power over csp broadband (after .5-40 filter)
        and bandpower over selected bands.
        Args:
            data: ndarrray of the data
            labels: list of the labels
            fs: frequency rate

        Returns:

        """
        csp = CSP(n_components=2, reg='ledoit_wolf', log=True, norm_trace=False, transform_into='average_power',
                  cov_est='epoch')
        csp_features = Pipeline(
            [('asd', UnsupervisedSpatialFilter(PCA(3), average=True)), ('asdd', csp)]).fit_transform(data,
                                                                                                     labels)
        # Get rest of features
        bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
        bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
        bandpower_features_wtf = np.concatenate(
            (csp_features,bandpower_features_new, bandpower_features_rel), axis=1)
        return bandpower_features_wtf


    pred = []
    fs = 125
    bands = np.matrix('7 12; 12 15; 17 22; 25 30; 7 35; 30 35')
    # bands = np.matrix([[i,i+2]for ind,i in enumerate(list(range(1,40))[:-2])])
    # bands = np.matrix([[4.0, 12.15], [32.0, 34.24], [23.0, 28.89], [37.0, 46.54], [4.0, 9.66], [29.0, 32.81], [39.0, 43.72], [39.0, 44.2], [33.0, 38.17], [33.0, 39.78], [4.0, 11.4], [31.0, 36.46], [31.0, 37.5], [11.0, 18.63], [1.0, 4.61], [33.0, 39.92], [37.0, 38.83], [4.0, 10.74], [32.0, 34.2], [23.0, 28.77], [37.0, 39.23], [33.0, 39.28], [38.0, 47.54], [0.0, 2.86], [21.0, 27.68]])
    # bands = '[28.34.64],[24.25.97],[21.23.85],[17.19.36],[38.40.05],[20.24.83],[34.39.69],[17.23.],[32.39.72],[38.42.76],[28.29.77],[38.38.97],[23.24.74],[10.12.92],[36.47.62],[22.33.14],[12.14.13],[14.17.42],[13.18.84],[26.27.9],[26.28.21],[26.28.13],[20.23.04],[35.39.26],[4.17.93],[25.27.21],[20.21.61],[8.10.85],[20.24.2],[10.11.28],[26.29.41],[12.17.79],[16.28.94],[1.2.77],[36.41.69],[1.6.69],[35.36.28],[37.39.35],[1.2.8],[3.4.98],[24.29.99],[13.17.35],[4.11.06],[23.30.76],[17.21.35],[8.11.88],[0.5.81],[24.28.92],[24.25.8],[37.40.54],[1.2.69],[24.28.7],[14.19.58],[24.29.88],[10.11.3],[21.22.24],[11.12.05],[15.17.21],[32.33.9],[35.36.12],[20.28.23],[3.6.49],[36.48.13],[3.6.6],[24.31.52],[17.21.83],[35.37.85],[25.26.05],[10.18.54],[28.30.95],[24.27.88],[35.36.97],[5.8.86],[25.29.7],[13.19.97],[37.40.88],[28.29.43],[0.6.06],[17.21.92],[38.39.7],[20.21.69],[33.39.81],[26.28.27],[7.11.86],[16.17.54],[1.2.7],[6.7.96],[32.33.99],[16.17.77],[32.34.21],[3.4.9],[20.22.99],[13.22.73],[35.38.8],[6.17.76],[34.41.06],[3.13.35],[5.7.43],[32.35.26],[16.17.83],[12.14.56],[14.17.7],[8.10.92],[0.5.88],[34.42.76],[8.15.78],[3.7.18],[6.9.88],[23.28.39],[33.44.14],[37.38.05],[16.25.41],[16.19.88],[37.38.45],[12.15.6],[29.35.76],[13.14.45],[13.17.27],[10.11.79],[2.4.13],[3.6.64],[23.28.29],[5.7.78],[1.9.69],[5.7.59],[3.5.61],[22.25.76],[28.29.54],[6.9.41],[6.11.99],[29.39.58],[27.36.49],[23.25.13],[3.5.56],[15.17.11],[39.49.35],[25.27.11],[32.41.4],[11.13.75],[26.31.29],[1.3.02],[1.6.84],[9.12.44],[8.10.64],[23.32.06],[33.34.09],[9.12.53],[25.27.4],[17.21.27],[20.25.39],[9.11.13],[14.17.48],[37.38.96],[10.14.86],[6.11.57],[16.17.48],[25.28.59],[12.14.61],[39.40.43],[10.12.09],[31.33.12],[25.27.46],[36.37.45],[39.44.84],[3.5.12],[38.42.23],[35.38.7],[28.29.24],[37.38.26],[14.18.8],[6.8.99],[9.12.14],[17.20.8],[5.7.32],[12.19.72],[19.20.61],[37.38.8],[8.11.59],[20.22.02],[28.29.83],[33.35.12],[32.33.47],[28.29.48],[33.35.13],[5.8.59],[4.5.43],[10.11.8],[22.28.64],[34.45.65],[9.11.64],[16.17.8],[17.23.97],[7.9.64],[20.24.88],[21.25.83],[21.30.73],[27.30.44],[5.9.74],[32.34.27],[26.27.99],[38.42.19],[12.23.14],[2.11.4],[5.10.84],[13.19.],[4.6.17],[27.30.86],[27.31.76],[4.6.99],[19.20.98],[7.9.99],[20.21.77],[21.27.64],[19.21.35],[15.17.4],[29.33.86],[3.9.81],[19.22.53],[25.27.53],[37.38.1],[37.39.98],[29.34.99],[15.17.53],[12.17.88],[12.16.18],[38.39.57]'
    #
    # # Ofir's data
    # EEG = scipy.io.loadmat(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\scripts\EEG.mat')
    # trainingVec = scipy.io.loadmat(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\scripts\trainingVec.mat')
    # data = EEG['EEG']
    # labels = np.ravel(trainingVec['trainingVec'].T)
    #  # data should be trails X electrodes X samples.
    # data = np.transpose(data, (2, 0, 1))
    #
    # final_data = []
    #
    # for trial in range(data.shape[0]):
    #     # C4
    #     data[trial][8] -= (data[trial][2] + data[trial][14] + data[trial][7] +
    #                           data[trial][9]) / 4
    #
    #     # C4
    #     data[trial][4] -= (data[trial][5] + data[trial][3] + data[trial][0] +
    #                           data[trial][10]) / 4
    #     new_data = np.delete(data[trial], [2, 14, 7, 9, 5, 3, 0, 10], axis=0)
    #     if trial == 0:
    #         final_data = new_data[np.newaxis]
    #     else:
    #         final_data = np.vstack((final_data, new_data[np.newaxis]))
    # data = final_data

    ## if the input is data:
    if data is None and labels is None:
        # Our data
        data2 = pd.read_pickle(path)
        #
        labels = data2.labels

        # # # Choose clean data or not
        # data = data2.epochs.get_data()
        data = ICA_perform(data2).get_data() # ICA
    # data = epochs_z_score(data)  # z score?
    # SPATIAL FILTERS LETS GO
    #Laplacian
    data, _ = EEG.laplacian(data)
    # data, labels = extract_2_labels(data, labels, [0,1,2])
    # pred, bands = band_hunter_swarm_algorithm(data, labels,fs)
    # bands = np.matrix(bands)

    # Initiate classifiers
    clf = svm.SVC(decision_function_shape='ovo', kernel='linear',tol=1e-4)
    rf_classifier = RandomForestClassifier(random_state=0)
    mlp_classifier = MLPClassifier(solver='adam',hidden_layer_sizes=[80,50,20,3,20,50,80],max_iter=400, random_state=0)
    xgb_classifier = XGBClassifier()
    ada_classifier = lgb.LGBMClassifier()
    # which features
    if method == 'current_features':
        bandpower_features_wtf = get_current_features(data=data, labels=labels, fs=fs)
    elif method == 'wacky_features':
        bandpower_features_wtf = get_features(data=data, labels=labels, fs=fs)
    else:
        data, baseline = baseline_extractor(data, fs)
        bandpower_features_wtf = get_features(data=data, labels=labels, fs=fs)
        baseline_features = get_features(data=baseline, labels=labels, fs=fs)
        bandpower_features_wtf = np.divide(bandpower_features_wtf, baseline_features)

    # # # feature selection
    # problem = SVMFeatureSelection(bandpower_features_wtf, labels)
    # task = Task(problem, max_iters=50)
    # algorithm = ParticleSwarmOptimization()
    # best_features, best_fitness = algorithm.run(task)
    # best_features_index = [ind for ind, feature in enumerate(best_features) if (feature > 0.5)]
    # bandpower_features_wtf = bandpower_features_wtf.T[best_features > 0.5].T

    # normalization
    scaler = StandardScaler()
    scaler.fit(bandpower_features_wtf)
    bandpower_features_wtf = scaler.transform(bandpower_features_wtf)

    # Trial rejection
    bandpower_features_wtf, labels = trials_rejection(bandpower_features_wtf, labels)

    # seperate the data before feature selection
    indices = np.arange(bandpower_features_wtf.shape[0])
    X_train, X_test, y_train, y_test, train_ind, test_ind = train_test_split(bandpower_features_wtf,
                                labels,indices, random_state=0)

    # Define selection algorithms
    rf_select = SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=800,random_state=0))
    mi_select = SelectKBest(mutual_info_classif, k=int(math.sqrt(data.shape[0])))
    # fisher_select = bandpower_features_wtf[:, fisher_score.fisher_score(bandpower_features_wtf,
    #                                                                     labels)[0:int(math.sqrt(data.shape[0]))]]

    # Define Pipelines
    model = SelectFromModel(LogisticRegression(C=1, penalty="l1", solver='liblinear', random_state=0))
    # define seq selections
    seq_select_clf = SequentialFeatureSelector(clf, n_features_to_select=int(math.sqrt(X_train.shape[0]))//2, n_jobs=1)
    seq_select_RF = SequentialFeatureSelector(rf_classifier, n_features_to_select=int(math.sqrt(X_train.shape[0])), n_jobs=1)
    seq_select_MLP = SequentialFeatureSelector(mlp_classifier, n_features_to_select=int(math.sqrt(X_train.shape[0])),  n_jobs=1)
    seq_select_XGB = SequentialFeatureSelector(xgb_classifier, n_features_to_select=int(math.sqrt(X_train.shape[0])), n_jobs=1)
    seq_select_ADA = SequentialFeatureSelector(ada_classifier, n_features_to_select=int(math.sqrt(X_train.shape[0])), n_jobs=1)

    pipeline_SVM = Pipeline([('lasso', model),('mi_select', mi_select), ('feat_selecting', seq_select_clf), ('SVM', clf)])
    pipeline_RF = Pipeline([('classify', rf_classifier)])
    pipeline_MLP = Pipeline([('lasso', model),('feat_selecting', mi_select), ('classify', mlp_classifier)])
    pipeline_XGB = Pipeline([('classify', xgb_classifier)])
    pipeline_ADA = Pipeline([('classify', ada_classifier)])
    # get scores with CV for each pipeline
    scores_mix = cross_val_score(pipeline_SVM, bandpower_features_wtf, labels, cv=5, n_jobs=1)
    scores_mix_pred = cross_val_predict(pipeline_SVM, bandpower_features_wtf, labels, cv=5, n_jobs=1)
    scores_mix2 = cross_val_score(pipeline_RF, bandpower_features_wtf, labels, cv=5, n_jobs=1)
    scores_mix_pred2 = cross_val_predict(pipeline_RF, bandpower_features_wtf, labels, cv=5, n_jobs=1)
    scores_mix3 = cross_val_score(pipeline_MLP, bandpower_features_wtf, labels, cv=5, n_jobs=1)
    scores_mix_pred3 = cross_val_predict(pipeline_MLP, bandpower_features_wtf, labels, cv=5, n_jobs=1)
    scores_mix4 = cross_val_score(pipeline_XGB, bandpower_features_wtf, labels, cv=5, n_jobs=1)
    scores_mix_pred4 = cross_val_predict(pipeline_XGB, bandpower_features_wtf, labels, cv=5, n_jobs=1)
    scores_mix5 = cross_val_score(pipeline_ADA, bandpower_features_wtf, labels, cv=5, n_jobs=1)
    scores_mix_pred5 = cross_val_predict(pipeline_ADA, bandpower_features_wtf, labels, cv=5, n_jobs=1)
    values = [scores_mix, scores_mix2,scores_mix3, scores_mix4,scores_mix5]
    names = ['SVM','RandomForest','NN','XGBC','ADA Boost']
    plt.bar(names, np.mean(values, axis=1))
    plt.suptitle(f'Classifiers prediction rate for method: {method}')
    plt.show()


    #print scores
    (print(f"SVM rate is: {np.mean(scores_mix)*100}%"))
    (print(f"RandomForest rate is: {np.mean(scores_mix2)*100}%"))
    (print(f"MLP rate is: {np.mean(scores_mix3)*100}%"))
    (print(f"XGBC rate is: {np.mean(scores_mix4)*100}%"))
    (print(f"ADA rate is: {np.mean(scores_mix5)*100}%"))
    plt.clf()
    fig = plt.figure(figsize=(8, 6), dpi=80)
    fontsize = 11

    # fit pipelines for the confusion matrix and get matrices
    mat1 = ConfusionMatrixDisplay.from_predictions(labels,scores_mix_pred,
                            normalize='all')
    ax1 = fig.add_subplot(231)
    mat1.plot(ax=ax1,cmap=plt.cm.Blues)
    ax1.set_title('SVM', fontsize=fontsize, fontweight='bold')


    mat2 = ConfusionMatrixDisplay.from_predictions(labels,scores_mix_pred2,
                            normalize='all')
    ax2 = fig.add_subplot(232)
    mat2.plot(ax =ax2,cmap=plt.cm.Blues)
    ax2.set_title('RF', fontsize=fontsize, fontweight='bold')

    mat3 = ConfusionMatrixDisplay.from_predictions(labels,scores_mix_pred3,
                            normalize='all')
    ax3 = fig.add_subplot(233)
    mat3.plot(ax=ax3,cmap=plt.cm.Blues)
    ax3.set_title('MLP', fontsize=fontsize, fontweight='bold')

    mat4 = ConfusionMatrixDisplay.from_predictions(labels,scores_mix_pred4,
                            normalize='all')
    ax4 = fig.add_subplot(234)
    mat4.plot(ax=ax4,cmap=plt.cm.Blues)
    ax4.set_title('XGB', fontsize=fontsize, fontweight='bold')

    mat5 = ConfusionMatrixDisplay.from_predictions(labels,scores_mix_pred5,
                            normalize='all')
    ax5 = fig.add_subplot(235)
    mat5.plot(ax=ax5,cmap=plt.cm.Blues)
    ax5.set_title('ADA', size='large', fontweight='bold')

    # another properties
    fig.suptitle(f'Confusion matrices', fontsize=20, fontweight='bold')
    fig.tight_layout(pad=2.0)
    textstr = '\n'.join(('0 - Right','1 - Left','2 - Idle'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.77, 0.29, textstr, fontsize=11, verticalalignment='top', bbox=props)
    plt.show()
    # print(bands)
    print(bandpower_features_wtf.shape)
    return np.mean(values, axis=1), bands

def get_tsne_feature_mat(model):
    """This function gets a model file and returns an dimension reduced(using tsne) features matrix and labels vector"""
    def ICA_perform(model):
        """
        Args:
            model: the model before ICA transform
            to_exclude: (list) list of the coordinates numbers to exclude

        Returns: epochs array after ICA transform
        """
        epochs = model.epochs
        ica = ICA(n_components=11, max_iter='auto', random_state=97)
        ica.fit(epochs)
        # ica.exclude = [0,1]
        ica.detect_artifacts(epochs)
        ica.apply(epochs)
        return epochs
    def trials_rejection(feature_mat, labels):
        to_remove = []
        nan_col = np.isnan(feature_mat).sum(axis=1)  # remove features with None values
        add_remove = np.where(np.in1d(nan_col, not 0))[0].tolist()
        to_remove += add_remove

        func = lambda x: np.mean(np.abs(x),axis=0) > 1.5  # remove features with extreme values - 2 std over the mean
        Z_bool = func(feature_mat)
        add_remove = np.where(np.in1d(Z_bool, not 0))[0].tolist()
        to_remove += add_remove
        feature_mat = np.delete(feature_mat, to_remove, axis=0)
        labels = np.delete(labels, to_remove, axis=0)
        print(f'trials rejected: {to_remove}')
        return feature_mat, labels
    # define parameters
    fs = 125
    bands = np.matrix('7 12; 12 15; 17 22; 25 30; 7 35; 30 35')
    # get data
    class_labels = model.labels
    feature_labels = []
    # get features
    data = ICA_perform(model).get_data()  # ICA
    #Laplacian
    data, _ = EEG.laplacian(data)
    csp = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False, transform_into='average_power', cov_est='epoch')
    csp_features = Pipeline([('asd',UnsupervisedSpatialFilter(PCA(3), average=False)),('asdd',csp)]).fit_transform(data,class_labels)
    [feature_labels.append(f'CSP_Component{i}') for i in range(csp_features.shape[1])]
    # Bandpower
    bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
    [feature_labels.append(f'BP_non_rel{np.ravel(i)}_{chan}') for i in bands for chan in model.epochs.ch_names]
    # relative bandpower
    bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
    [feature_labels.append(f'BP_non_rel{np.ravel(i)}_{chan}') for i in bands for chan in model.epochs.ch_names]
    # get all of them in one matrix
    features_mat = np.concatenate((csp_features,bandpower_features_new, bandpower_features_rel), axis=1)
    scaler = StandardScaler()
    features_mat = scaler.fit_transform(features_mat)
    #Trial rejection
    features_mat, labels = trials_rejection(features_mat, class_labels)
    # Define selection algorithms
    rf_select = SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=800,random_state=0))
    mi_select = SelectKBest(mutual_info_classif, k=2)
    # fisher_select = bandpower_features_wtf[:, fisher_score.fisher_score(bandpower_features_wtf,
    #                                                                     labels)[0:int(math.sqrt(data.shape[0]))]]

    # Define Pipelines
    model = SelectFromModel(LogisticRegression(C=1, penalty="l1", solver='liblinear', random_state=0))
    features_mat = model.fit_transform(features_mat, class_labels)
    tsne = manifold.TSNE(
        n_components=2,
        init="random",
        random_state=0,
        perplexity=35,
        learning_rate="auto",
        n_iter=3000,
    )
    features_mat = tsne.fit_transform(features_mat, class_labels)
    # features_mat = mi_select.fit_transform(features_mat, class_labels)
    class_labels = labels
    return features_mat, class_labels, feature_labels
def plot_SVM(feature_mat,labels):
    """This function gets a features_mat and labels (it takes only the first 2 features for visualization) and
    visualize SVM's performance on the dimension reduces(preferably) data"""
    h = .02  # step size in the mesh
    C = 1.0  # SVM regularization parameter
    feature_mat = feature_mat[:,:2]
    clf = svm.SVC(decision_function_shape='ovo', kernel='linear').fit(feature_mat,labels)
    x_min, x_max = feature_mat[:, 0].min() - 1, feature_mat[:, 0].max() + 1
    y_min, y_max = feature_mat[:, 1].min() - 1, feature_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # Plot also the training points
    plt.scatter(feature_mat[:, 0], feature_mat[:, 1], c=labels, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('SVC with linear kernel')
    plt.show()
def plot_calssifiers(datasets):
    """This function gets a data set and plots the performance of different classifiers over that dataset."""
    h = 0.02  # step size in the mesh

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
        "XGBC",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(decision_function_shape='ovo', kernel='linear', tol=1e-4),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(random_state=0),
        OneVsRestClassifier(
            MLPClassifier(solver='adam', alpha=1e-6, hidden_layer_sizes=[80] * 5, max_iter=400, random_state=0)),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        OneVsRestClassifier(XGBClassifier()),

    ]
    figure = plt.figure(figsize=(27, 9))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        asd = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm,alpha=0.8, edgecolors="k")
        # Plot the testing points
        ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm,alpha=0.8, edgecolors="y"
        )
        ax.add_artist(ax.legend(*asd.legend_elements(),
                            loc="upper left", title="Classes"))
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                # Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

            # Plot the training points
            ax.scatter(
                X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm,alpha=0.8, edgecolors="k"
            )
            # Plot the testing points
            ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                cmap=cm,
                alpha=0.8,
                edgecolors="y",
            )

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(
                xx.max() - 0.3,
                yy.min() + 0.3,
                ("%.2f" % score).lstrip("0"),
                size=15,
                horizontalalignment="right",
            )
            i += 1
    plt.tight_layout()
    plt.savefig("High resoltion.png", dpi=300)
    plt.show()

def plot_online_results(path):
    """This function plots the online results of an online experiment using the path to the results file (json file)"""
    with open(path) as f:
        data = json.load(f)
    rep_on_class = len(data[0])
    num_of_trials_class = len(data)/3
    results_dict = {}
    for trial in data:
        results_dict[str(trial[0][0])] = 0
    expected = []
    prediction = []
    for trial in data:
        for ind in trial:
            if ind[0]==ind[1]:
                results_dict[str(ind[0])] += 1/(rep_on_class*num_of_trials_class)
            expected.append(ind[0])
            prediction.append(ind[1])

    # the bar plot
    all_labels = {'0':'Right','1':'Left','2':'Idle','3':'Tongue','4':'Hands'}
    items = list(results_dict.items())
    items = sorted(items)
    classes = [classes[0] for classes in items]
    values = [values[1] for values in items]
    labels = [all_labels[i] for i in classes]
    plt.bar(classes,values,color = (0.5,0.1,0.5,0.6))
    plt.title('Online results - The prediction percentage for each class\n')
    plt.xlabel('Prediction percentage')
    plt.ylabel('Classes ')
    plt.xticks(classes,labels)
    plt.show()

    # the confusion matrix
    cm = confusion_matrix(expected,prediction)
    ax = seaborn.heatmap(cm/np.sum(cm),fmt='.2%', annot=True, cmap='Blues')
    ax.set_title('Online results - confusion matrix\n')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()

def over_time_pred(recording_paths):
    """This function visualize the prediction rate of a trial. The motivation was to check to facilitation in the
    motor imagery paradigm as it was built when this function was written. f.e. it was meant to check if the prediciton
    in the first mini-trial inside a certain class trial had better results than the third mini-trial."""
    tot_class = 0
    success_per_trial = [0] * 5
    for path_results in recording_paths:
        with open(path_results) as f:
            data = json.load(f)
        for trial in data:
            for classification_ind, classification in enumerate(trial):
                if classification[0] == classification[1]:
                    success_per_trial[classification_ind] += 1
                    tot_class +=1
    names = [f'trial {ind + 1}' for ind in range(len(data[0]))]
    plt.figure(figsize=(9, 3))
    plt.bar(names, np.array(success_per_trial)/tot_class)
    plt.suptitle('Over-time online learning trial success rate out of total classification attempts')
    plt.show()

def concat_all_recordings(target_dir):
    """
    This function concatenates all the recordings in a given directory.
    Args:
        target_dir: string : path to the directory

    Returns:
        data, labels : of all the recordings in that directory
    """
    strip_func = lambda x: os.path.dirname(x)
    curr_dir = os.getcwd()
    recording_dir = strip_func(curr_dir) + r'\recordings' + target_dir
    files_in_recording_dir = os.listdir(recording_dir)
    all_trials = []
    all_labels = []
    for date_name in files_in_recording_dir:
        no_model_found = False
        temp_path = os.path.join(recording_dir, date_name)
        while True:
            try:
                temp_model = pd.read_pickle(os.path.join(temp_path, 'pre_laplacian.pickle'))
                break
            except FileNotFoundError:
                try:
                    temp_model = pd.read_pickle(os.path.join(temp_path,'trained_model.pickle'))
                    break
                except FileNotFoundError:
                    try:
                        temp_model = pd.read_pickle(os.path.join(temp_path, 'model.pickle'))
                        break
                    except FileNotFoundError:
                        no_model_found = True
                        break
        if no_model_found:
            continue
        temp_trials = temp_model.epochs.get_data()
        temp_labels = temp_model.labels
        try:
            all_trials = np.concatenate((all_trials,temp_trials),axis=0) if type(all_trials) == np.ndarray else temp_trials
        except ValueError:
            if np.size(all_trials,2) < np.size(temp_trials,2):
                temp_trials = temp_trials[:, :, 0:np.size(all_trials, 2)]
            else:
                all_trials = all_trials[:,:,0:np.size(temp_trials,2)]
            all_trials = np.concatenate((all_trials,temp_trials),axis=0) if type(all_trials) == np.ndarray else temp_trials
        all_labels = np.concatenate((all_labels,temp_labels),axis=0) if type(all_trials) == np.ndarray else temp_labels
    return all_trials, all_labels

def baseline_extractor(data, fs, baseline_length=1):
    """
    Extracts the baseline from the data.
    Args:
        data: data in ndarray
        fs: frequency rate

    Returns:
        data ,baseline: ndarrays.
    """
    baseline = data[:, :, :fs*baseline_length]
    data = data[:, :, fs*baseline_length::]
    return data, baseline

def extract_2_labels(data, labels, classes_to_extract):
        """
        This function extracts the classes you choose from the given data
        Args:
            data: ndarrray of the data
            labels: list of the labels
            classes_to_extract: list[int[]] : a list of integers which represents the calsses in the daa you want
            to extract

        Returns:
            data, labels (with only the desired classes)
        """
        data_2_extract = [data[ind] for ind, label in enumerate(labels) if labels[ind] in classes_to_extract]
        labels_2_extract = [label for ind, label in enumerate(labels) if labels[ind] in classes_to_extract]
        return np.reshape(data_2_extract,[len(data_2_extract)] + list(data_2_extract[0].shape)), np.ravel(labels_2_extract)

def save_as_mat(path, data, labels):
    """
    This function saves input data and labels in input path as .mat file. We used it to visualize data in matlab, s
    we did not found an equivalent in python the the spectograms in matlab.
    Args:
        path:
        data:
        labels:

    Returns:

    """
    labels = {"labels": labels, "label": "experiment"}
    data = {"data": data, "label": "experiment"}
    scipy.io.savemat(path + r"\data.mat", labels)
    scipy.io.savemat(path + r"\data.mat", data)

if __name__ == '__main__':
    path = r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\avi_right_left_idle\Online_05_07_22-18_16_32'
    trials, labels = concat_all_recordings(r'\all_recording_avi')
    # temp_results, temp_bands = load_eeg(path + '/model.pickle', method='wacky_features_asd')
    temp_results, temp_bands = load_eeg(path + '/trained_model.pickle', data=trials, labels=labels, method='wacky_features')
    # import pandas as pd
    # model1 = pd.read_pickle(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\roy/89/trained_model.pickle')
    # model2 = pd.read_pickle(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\roy/22/unfiltered_model.pickle')
    # model3 = pd.read_pickle(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\roy/57/trained_model.pickle')
    # datasets = [get_feature_mat(model1)[0:2],get_feature_mat(model2)[0:2],get_feature_mat(model3)[0:2]]
    # method_dict = {'current_features':0}#, 'wacky_features_with_baseline':2}
    # results = []
    # bands = []
    # for method in method_dict:
    #     temp_results, temp_bands = load_eeg(path+'/trained_model.pickle', data=trials,labels=labels, method=method)
    #     results.append(temp_results)
    #     bands.append(temp_bands)
    # plot_calssifiers(datasets)
    # import matplotlib.pyplot as plt
    # plot_online_results(path+'/results.json')
    # over_time_pred([fr'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\roy\{rec}\results.json'
    #                for rec in [88]])
