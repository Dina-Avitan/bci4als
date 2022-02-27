# This script is meant to load models and allow the user to change hyper-parameters
# so you could fine-tune the real offline_training class
import copy
import math
from tkinter import filedialog, Tk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import FastICA
import scipy
import scipy.io
from bci4als import ml_model
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from itertools import permutations
from mne import filter

def playground():
    # load eeg data
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    raw_model = pd.read_pickle(fr'{file_path}')
    raw_model.offline_training(model_type='simple_svm')
    scores = raw_model.cross_val()
    (print(f"Prediction rate is: {scores[0]}%"))


def load_eeg():
    fs = 125
    bands = np.matrix('7 12; 12 15; 17 22; 25 30; 7 35; 30 35')
    max_score = 1
    clf = svm.SVC(decision_function_shape='ovo', kernel='linear')
    # clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=50, random_state=0, max_iter=400)

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

    # Our data
    # data1 = pd.read_pickle(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\noam\2\raw_model.pickle')
    data2 = pd.read_pickle(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\roy\2\unfiltered_model.pickle')
    # data = np.concatenate((data1.epochs.get_data()[:, :, :550], data2.epochs.get_data()[:, :, :550]), axis=0)
    # labels = np.concatenate((data1.labels,data2.labels), axis=0)
    #
    labels = data2.labels
    data = data2.epochs.get_data()
    # # ICA which does not work
    # for d in range(len(data)):
    #     if d == 0:
    #         data_ica = sklearn.decomposition.FastICA().fit_transform(data[d, :, :].T).T[np.newaxis]
    #     else:
    #         data_ica = np.vstack((data_ica, sklearn.decomposition.FastICA(n_components=data.shape[0],max_iter=400).fit_transform(data[d,:, :].T).
    #                           T[np.newaxis]))
    # data = data_ica

    # # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)#, transform_into='average_power', cov_est='epoch')
    csp_features = Pipeline([('CSP', csp), ('LDA', lda)]).fit_transform(data, labels)

    for feat_num in [9]:#range(1, int(math.sqrt(data.shape[0]))):
        bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
        bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
        bandpower_features_old = ml_model.MLModel.hjorthMobility(data)
        bandpower_features_wtf = np.concatenate((bandpower_features_new, bandpower_features_rel), axis=1)
        scaler = StandardScaler()
        scaler.fit(bandpower_features_wtf)
        bandpower_features_wtf = scaler.transform(bandpower_features_wtf)
        # bandpower_features_wtf = SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=50, max_features=9)).fit_transform(bandpower_features_wtf, labels)
        bandpower_features_wtf = SelectKBest(mutual_info_classif, k=feat_num).fit_transform(bandpower_features_wtf, labels)
        print(bandpower_features_wtf.shape)
        scores_mix = cross_val_score(clf, bandpower_features_wtf, labels, cv=8)
        (print(f"Prediction rate is: {np.mean(scores_mix)*100}%"))

        if np.mean(scores_mix)*100 > max_score:
            max_score = np.mean(scores_mix)*100
            feat_num_max = feat_num
    print(max_score, feat_num_max)

def permutation_func():
    fs = 125
    bands = np.matrix('7 12; 12 15; 17 22; 25 30; 7 35; 30 35')
    max_score = 1
    clf = svm.SVC(decision_function_shape='ovo', kernel='linear')
    #data2 = pd.read_pickle(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\roy\2\unfiltered_model.pickle')
    data2 = pd.read_pickle(r'C:\Users\pc\Desktop\bci4als\recordings\roy\2\unfiltered_model.pickle')
    labels = data2.labels
    # data2.epochs.filter(1., 40., fir_design='firwin', skip_by_annotation='edge', verbose=False)
    #data = data2.epochs.get_data()
    data = ICA(data2)
    # perm_c3 = (0, 5, 3, 9, 7, 1, 4, 6, 8, 10)
    # perm_c3 = (0, 3, 5, 9, 7, 1, 4, 6, 8, 10)
    # perm_c3 = (0, 1, 2, 3, 5, 6, 4, 7, 8, 9, 10)
    #
    # for trial in range(data.shape[0]):
    #     # C3
    #     data[trial][perm_c3[0]] -= (data[trial][perm_c3[1]] + data[trial][perm_c3[2]] + data[trial][perm_c3[3]] +
    #                           data[trial][perm_c3[4]]) / 2
    #     # data[trial][perm_c3[0]] = (data[trial][perm_c3[0]] - data[trial][perm_c3[0]].mean()) - \
    #     #                           (((data[trial][perm_c3[1]] - data[trial][perm_c3[1]].mean())
    #     #                             + (data[trial][perm_c3[2]] - data[trial][perm_c3[2]].mean())
    #     #                             + (data[trial][perm_c3[3]] - data[trial][perm_c3[3]].mean())
    #     #                             + (data[trial][perm_c3[4]] - data[trial][perm_c3[4]].mean())) / 4)
    #     # C4
    #     data[trial][perm_c3[5]] -= (data[trial][perm_c3[6]] + data[trial][perm_c3[7]] + data[trial][perm_c3[8]] +
    #                           data[trial][perm_c3[9]]) / 2
    #     # data[trial][perm_c3[5]] = (data[trial][perm_c3[5]] - data[trial][perm_c3[5]].mean()) - \
    #     #                           (((data[trial][perm_c3[6]] - data[trial][perm_c3[6]].mean())
    #     #                             + (data[trial][perm_c3[7]] - data[trial][perm_c3[7]].mean())
    #     #                             + (data[trial][perm_c3[8]] - data[trial][perm_c3[8]].mean())
    #     #                             + (data[trial][perm_c3[9]] - data[trial][perm_c3[9]].mean())) / 4)
    #     new_data = np.delete(data[trial], [perm_c3[point] for point in [1, 2, 3, 4, 6, 7, 8, 9]], axis=0)
    #     # new_data = data[trial]
    #     if trial == 0:
    #         final_data = new_data[np.newaxis]
    #     else:
    #         final_data = np.vstack((final_data, new_data[np.newaxis]))
    # data = final_data
    # # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)#, transform_into='average_power', cov_est='epoch')
    csp_features = Pipeline([('CSP', csp), ('LDA', lda)]).fit_transform(data, labels)
    bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
    bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
    # bandpower_features_old = ml_model.MLModel.hjorthMobility(data)
    bandpower_features_wtf = np.concatenate((bandpower_features_new, bandpower_features_rel), axis=1)
    scaler = StandardScaler()
    scaler.fit(bandpower_features_wtf)
    bandpower_features_wtf = scaler.transform(bandpower_features_wtf)
    bandpower_features_wtf = SelectKBest(mutual_info_classif, k=8).fit_transform(bandpower_features_wtf, labels)
    scores_mix = cross_val_score(clf, bandpower_features_wtf, labels, cv=8)
    print(np.mean(scores_mix)*100)
    if np.mean(scores_mix)*100 > max_score:
        max_score = np.mean(scores_mix)*100
        feat_num_max = 9
        print(f"Prediction rate is: {np.mean(scores_mix) * 100}%")
        print(max_score, feat_num_max)

def ICA(ufiltered_model):
    # ica = ICA(n_components=15, method='fastica', max_iter="auto").fit(epochs)
    data = ufiltered_model.epochs.get_data()
    ica_data = np.zeros(data.shape)
    for i in range(len(data)):
        transformer = FastICA(n_components=5, random_state=0)
        X_transformed = transformer.fit_transform(data[i].transpose())
        s = transformer.inverse_transform(X_transformed, copy=True)
        ica_data[i]= s.transpose()
    return ica_data

if __name__ == '__main__':
    # playground()
    # load_eeg()
    permutation_func()
