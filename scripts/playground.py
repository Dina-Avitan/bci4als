# This script is meant to load models and allow the user to change hyper-parameters
# so you could fine-tune the real offline_training class
import math
from tkinter import filedialog, Tk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import scipy
import scipy.io
from bci4als import ml_model
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def playground():
    # load eeg data
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    raw_model = pd.read_pickle(fr'{file_path}')
    raw_model.offline_training(model_type='simple_svm')
    scores = raw_model.cross_val()
    (print(f"Prediction rate is: {scores}%"))


def load_eeg():
    fs = 500
    bands = np.matrix('7 12; 12 15; 17 22; 25 30; 7 35; 30 35')
    max_score = 1
    clf = svm.SVC(decision_function_shape='ovo', kernel='linear')

    # Ofir's data
    EEG = scipy.io.loadmat(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\scripts\EEG.mat')
    trainingVec = scipy.io.loadmat(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\scripts\trainingVec.mat')
    data = EEG['EEG']
    labels = np.ravel(trainingVec['trainingVec'].T)
     # data should be trails X electrodes X samples.
    data = np.transpose(data, (2, 0, 1))

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=False, norm_trace=False, transform_into='average_power', cov_est='epoch')
    csp_features = Pipeline([('CSP', csp), ('LDA', lda)]).fit_transform(data, labels)

    final_data = []
    for trial in range(data.shape[0]):
        # C4
        data[trial][8] -= (data[trial][2] + data[trial][14] + data[trial][7] +
                              data[trial][9]) / 4

        # C4
        data[trial][4] -= (data[trial][5] + data[trial][3] + data[trial][0] +
                              data[trial][10]) / 4
        new_data = np.delete(data[trial], [2, 14, 7, 9, 5, 3, 0, 10], axis=0)
        if trial == 0:
            final_data = new_data[np.newaxis]
        else:
            final_data = np.vstack((final_data, new_data[np.newaxis]))
    data = final_data

    # Our data
    # data1 = pd.read_pickle(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\noam\2\raw_model.pickle')
    # data2 = pd.read_pickle(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\noam\3\raw_model.pickle')
    # data = np.concatenate((data1.epochs.get_data()[:, :, :550], data2.epochs.get_data()[:, :, :550]), axis=0)
    # labels = np.concatenate((data1.labels,data2.labels), axis=0)

    for feat_num in range(1, int(math.sqrt(data.shape[0]))):
        bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
        # bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.9, relative=True)
        bandpower_features_old = ml_model.MLModel.hjorthMobility(data)

        bandpower_features_wtf = np.concatenate((bandpower_features_new, bandpower_features_old, csp_features), axis=1)
        bandpower_features_wtf = scipy.stats.zscore(bandpower_features_wtf)
        bandpower_features_wtf = SelectKBest(mutual_info_classif, k=feat_num).fit_transform(bandpower_features_wtf, labels)
        scores_mix = cross_val_score(clf, bandpower_features_wtf, labels, cv=8)
        (print(f"Prediction rate is: {np.mean(scores_mix)*100}%"))

        if np.mean(scores_mix)*100 > max_score:
            max_score = np.mean(scores_mix)*100
            feat_num_max = feat_num
    print(max_score, feat_num_max)


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
    playground()
    #load_eeg()
