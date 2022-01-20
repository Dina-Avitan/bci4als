# This script is meant to load models and allow the user to change hyper-parameters
# so you could fine-tune the real offline_training class
import copy
import math
from tkinter import filedialog, Tk

import eeglib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
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
    # clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=10, random_state=1, max_iter=700)
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=800)
    final_data = []
    # permutations for laplace
    # perm_ofir
    # perm_c3 = (8, 2, 14, 7, 9, 4, 5, 3, 0, 10)
    # perm our weird combo
    # perm_c3 = (0, 1, 2, 3, 7, 9, 4, 5, 6, 8, 10) # 63%
    #more
    # perm_c3 = (0, 1, 2, 3, 5, 6, 4, 7, 8, 9, 10) # 61
    # perm_c3 = (0, 1, 2, 3, 5, 6, 4, 7, 9, 8, 10) # 61
    # perm_c3 = (0, 1, 2, 3, 4, 6, 5, 9, 7, 8, 10)
    # no bp
    perm_c3 = (9, 10, 3, 4, 1, 2, 0, 6, 7, 8, 5)

    # # Ofir's data
    # EEG = scipy.io.loadmat(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\scripts\EEG.mat')
    # trainingVec = scipy.io.loadmat(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\scripts\trainingVec.mat')
    # data = EEG['EEG']
    # labels = np.ravel(trainingVec['trainingVec'].T)
    #  # data should be trails X electrodes X samples.
    # data = np.transpose(data, (2, 0, 1))

    # Our data
    data2 = pd.read_pickle(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\roy\2\unfiltered_model.pickle')
    labels = data2.labels
    data = data2.epochs.get_data()

    for trial in range(data.shape[0]):
        # C3
        data[trial][perm_c3[0]] -= (data[trial][perm_c3[1]] + data[trial][perm_c3[2]] + data[trial][perm_c3[3]] +
                                    data[trial][perm_c3[4]]) / 4

        # C4
        data[trial][perm_c3[5]] -= (data[trial][perm_c3[6]] + data[trial][perm_c3[7]] + data[trial][perm_c3[8]] +
                                    data[trial][perm_c3[9]]) / 4

        new_data = np.delete(data[trial], [perm_c3[point] for point in [1, 2, 3, 4, 6, 7, 8, 9]], axis=0)
        if trial == 0:
            final_data = new_data[np.newaxis]
        else:
            final_data = np.vstack((final_data, new_data[np.newaxis]))
    data = final_data

    # # get csp features
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=False, norm_trace=False, transform_into='average_power', cov_est='epoch')
    csp_features = Pipeline([('CSP', csp), ('LDA', lda)]).fit_transform(data, labels)
    for feat_num in range(1, int(math.sqrt(data.shape[0]))):
        bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
        bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
        bandpower_features_old = ml_model.MLModel.hjorthMobility(data)
        # lzc_features = ml_model.MLModel.LZC(data)
        # dfa_features = ml_model.MLModel.DFA(data)

        # add as much features as you like
        bandpower_features_wtf = np.concatenate((bandpower_features_new, bandpower_features_old, csp_features,
                                                 bandpower_features_rel), axis=1)
        scaler = StandardScaler()
        scaler.fit(bandpower_features_wtf)
        bandpower_features_wtf = scaler.transform(bandpower_features_wtf)
        # bandpower_features_wtf = SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=80)).fit_transform(bandpower_features_wtf, labels)
        bandpower_features_wtf = SelectKBest(mutual_info_classif, k=feat_num).fit_transform(bandpower_features_wtf, labels)
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
    # clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=200, random_state=0, max_iter=800)
    clf = svm.SVC(decision_function_shape='ovo', kernel='linear')
    data2 = pd.read_pickle(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\roy\2\unfiltered_model.pickle')
    labels = data2.labels
    combinations = list(permutations(range(11)))
    counter = 0
    add = 2
    unique_check = 0
    while counter < len(combinations):
        perm_c3 = combinations[counter]
        data = copy.deepcopy(data2.epochs.get_data())
        final_data = []
        # perm_c3 = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 7)
        # perm_c3 = (0, 5, 3, 9, 7, 1, 4, 6, 8, 10)
        # perm_c3 = (0, 1, 2, 3, 7, 9, 4, 5, 6, 8, 10)
        # perm_c3 = [8, 9, 6, 5, 3, 10, 2, 7, 4, 1, 0]
        new_data = []
        # perm_c3 = list(reversed(perm_c3))
        for trial in range(data.shape[0]):
            # C3
            data[trial][perm_c3[0]] -= (data[trial][perm_c3[1]] + data[trial][perm_c3[2]] + data[trial][perm_c3[3]] +
                                  data[trial][perm_c3[4]]) / 4

            # C4
            data[trial][perm_c3[5]] -= (data[trial][perm_c3[6]] + data[trial][perm_c3[7]] + data[trial][perm_c3[8]] +
                                  data[trial][perm_c3[9]]) / 4
            new_data = np.delete(data[trial], [perm_c3[point] for point in [1, 2, 3, 4, 6, 7, 8, 9]], axis=0)
            if trial == 0:
                final_data = new_data[np.newaxis]
            else:
                final_data = np.vstack((final_data, new_data[np.newaxis]))
        data = final_data
        bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
        bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
        bandpower_features_old = ml_model.MLModel.hjorthMobility(data)
        bandpower_features_wtf = np.concatenate((bandpower_features_new, bandpower_features_old, bandpower_features_rel
                                                 ), axis=1)
        scaler = StandardScaler()
        scaler.fit(bandpower_features_wtf)
        scaler.transform(bandpower_features_wtf)
        bandpower_features_wtf = SelectKBest(mutual_info_classif, k=9).fit_transform(bandpower_features_wtf, labels)
        scores_mix = cross_val_score(clf, bandpower_features_wtf, labels, cv=8)
        if counter // 10000 >= 1 and counter // 10000 != unique_check:
            print(counter)
            unique_check = counter // 10000
        if np.mean(scores_mix)*100 > max_score:
            max_score = np.mean(scores_mix)*100
            feat_num_max = 9
            print(f"Prediction rate is: {np.mean(scores_mix) * 100}%")
            print(max_score, feat_num_max, perm_c3)
        elif (max_score - np.mean(scores_mix)*100) > 3:
            if add < 200:
                add = add**2
            else:
                add *= 2
            counter += add
        elif (max_score - np.mean(scores_mix)*100) <= 3:
            add = 2
            counter += add
        counter += 1

def sort_perm():
    combinations = list(permutations(range(11)))[0:1000]
    ans = sorted(combinations, key=dunno, reverse=True)
    print(ans[0:10])

def dunno(perm_c3):
    fs = 125
    bands = np.matrix('7 12; 12 15; 17 22; 25 30; 7 35; 30 35')
    clf = svm.SVC(decision_function_shape='ovo', kernel='linear')
    data2 = pd.read_pickle(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\roy\2\unfiltered_model.pickle')
    labels = data2.labels
    data = copy.deepcopy(data2.epochs.get_data())
    final_data = []
    for trial in range(data.shape[0]):
        # C3
        data[trial][perm_c3[0]] -= (data[trial][perm_c3[1]] + data[trial][perm_c3[2]] + data[trial][perm_c3[3]] +
                                    data[trial][perm_c3[4]]) / 4

        # C4
        data[trial][perm_c3[5]] -= (data[trial][perm_c3[6]] + data[trial][perm_c3[7]] + data[trial][perm_c3[8]] +
                                    data[trial][perm_c3[9]]) / 4
        new_data = np.delete(data[trial], [perm_c3[point] for point in [1, 2, 3, 4, 6, 7, 8, 9]], axis=0)
        if trial == 0:
            final_data = new_data[np.newaxis]
        else:
            final_data = np.vstack((final_data, new_data[np.newaxis]))
    data = final_data
    # bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
    # bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
    bandpower_features_old = ml_model.MLModel.hjorthMobility(data)
    lzc_features = ml_model.MLModel.LZC(data)
    dfa_features = ml_model.MLModel.DFA(data)

    bandpower_features_wtf = np.concatenate((lzc_features, dfa_features, bandpower_features_old), axis=1)
    scaler = StandardScaler()
    scaler.fit(bandpower_features_wtf)
    scaler.transform(bandpower_features_wtf)
    bandpower_features_wtf = SelectKBest(mutual_info_classif, k=8).fit_transform(bandpower_features_wtf, labels)
    scores_mix = cross_val_score(clf, bandpower_features_wtf, labels, cv=8)
    return np.mean(scores_mix) * 100

if __name__ == '__main__':
    # playground()
    # load_eeg()
    permutation_func()
    # sort_perm()
