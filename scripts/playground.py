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
    perm_c3 = (0, 1, 2, 3, 7, 9, 4, 5, 6, 8, 10)
    # perm_c3 = [8, 9, 6, 5, 3, 10, 2, 7, 4, 1, 0]

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
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=200, random_state=0, max_iter=800)
    # clf = svm.SVC(decision_function_shape='ovo', kernel='linear')
    data2 = pd.read_pickle(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\noam\8\unfiltered_model.pickle')
    labels = data2.labels
    combinations = list(permutations(range(11)))
    counter = 1
    for perm_c3 in combinations[500000:]:
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
        bandpower_features_old = ml_model.MLModel.hjorthMobility(data)
        bandpower_features_wtf = np.concatenate((bandpower_features_new, bandpower_features_old), axis=1)
        scaler = StandardScaler()
        scaler.fit(bandpower_features_wtf)
        scaler.transform(bandpower_features_wtf)
        bandpower_features_wtf = SelectKBest(mutual_info_classif, k=15).fit_transform(bandpower_features_wtf, labels)
        scores_mix = cross_val_score(clf, bandpower_features_wtf, labels, cv=8)
        print(np.mean(scores_mix)*100)
        if counter % 100000 == 0:
            print(counter)
        if np.mean(scores_mix)*100 > max_score:
            max_score = np.mean(scores_mix)*100
            feat_num_max = 9
            print(f"Prediction rate is: {np.mean(scores_mix) * 100}%")
            print(max_score, feat_num_max, perm_c3)

def visualize_svm(X, y):
    X = SelectKBest(chi2, k=2).fit_transform(X, y)
    h = .02  # step size in the mesh
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    lin_svc = svm.LinearSVC(C=C).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']

    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()

if __name__ == '__main__':
    # playground()
    load_eeg()
    # permutation_func()
