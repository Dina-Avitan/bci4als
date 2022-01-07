# This script is meant to load models and allow the user to change hyper-parameters
# so you could fine-tune the real offline_training class
from tkinter import filedialog, Tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.feature_selection import SelectKBest,chi2
import scipy
import scipy.io
from bci4als import ml_model


def playground():
    # load eeg data
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    raw_model = pd.read_pickle(fr'{file_path}')
    raw_model.offline_training(model_type='simple_svm')
    scores = raw_model.cross_val()
    (print(f"Prediction rate is: {np.mean(scores)*100}%"))
    raw_model.features_mat = SelectKBest(chi2, k=8).fit_transform(raw_model.features_mat, raw_model.labels)
    scores = raw_model.cross_val()
    (print(f"Prediction rate is: {np.mean(scores)*100}%"))
    raw_model.features_mat = scipy.stats.zscore(raw_model.features_mat)
    scores = raw_model.cross_val()
    (print(f"Prediction rate is: {np.mean(scores)*100}%"))
    visualize_svm(raw_model.features_mat, raw_model.labels)


def load_eeg():
    EEG = scipy.io.loadmat(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\scripts\EEG.mat')
    trainingVec = scipy.io.loadmat(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\scripts\trainingVec.mat')
    data = EEG['EEG']
    labels = trainingVec['trainingVec']
    bands = np.matrix('8 12; 16 22; 30 35')
    fs = 500
    data = np.moveaxis(data, [0, 1, 2], [2, 1, 0])
    bandpower_features = ml_model.MLModel.extract_bandpower(data, bands, fs)

    # features_mat = bandpower_features

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
