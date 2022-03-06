# This script is meant to load models and allow the user to change hyper-parameters
# so you could fine-tune the real offline_training class
import copy
import math
from tkinter import filedialog, Tk
import scipy.io
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, SelectFromModel
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from skfeature.function.similarity_based import fisher_score
from sklearn.tree import DecisionTreeClassifier
from mne.decoding import UnsupervisedSpatialFilter
from bci4als import ml_model, EEG
from sklearn import svm
from sklearn.model_selection import cross_val_score,  train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from mne.preprocessing import ICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

    fs = 125
    bands = np.matrix('7 12; 12 15; 17 22; 25 30; 7 35; 30 35')
    # bands = np.matrix('1 4; 7 12; 12 15; 17 22; 1 40; 25 40')
    clf = svm.SVC(decision_function_shape='ovo', kernel='linear',tol=1e-4)
    # clf = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto', tol=1e-6)
    # clf = LinearDiscriminantAnalysis(tol=1e-8)
    # clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=50, random_state=0, max_iter=400)

    # # Ofir's data
    # EEG = scipy.io.loadmat(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\scripts\EEG.mat')
    # trainingVec = scipy.io.loadmat(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\scripts\trainingVec.mat')
    # data = EEG['EEG']
    # labels = np.ravel(trainingVec['trainingVec'].T)
    #  # data should be trails X electrodes X samples.
    # data = np.transpose(data, (2, 0, 1))

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
    data2 = pd.read_pickle(r'../recordings/roy/22/raw_model.pickle')
    #
    labels = data2.labels

    # Choose clean data or not
    data = data2.epochs.get_data()
    # data = ICA_perform(data2).get_data()  # ICA
    # data = epochs_z_score(data)  # z score?

    #Laplacian
    data, _ = EEG.laplacian(data)
    # Initiate classifiers
    rf_classifier = RandomForestClassifier(random_state=0)
    mlp_classifier = OneVsRestClassifier(MLPClassifier(solver='adam', alpha=1e-6,hidden_layer_sizes=[80]*5,max_iter=400, random_state=0))
    xgb_classifier = OneVsRestClassifier(XGBClassifier())
    ada_classifier = AdaBoostClassifier(random_state=0)

    # # Get CSP features
    csp = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False, transform_into='average_power', cov_est='epoch')
    csp_features = Pipeline([('asd',UnsupervisedSpatialFilter(PCA(11), average=True)),('asdd',csp)]).fit_transform(data, labels)
    # Get rest of features
    bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
    bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
    # hjorthMobility_features = ml_model.MLModel.hjorthMobility(data)
    # LZC_features = ml_model.MLModel.LZC(data)
    # DFA_features = ml_model.MLModel.DFA(data)
    bandpower_features_wtf = np.concatenate((csp_features, bandpower_features_new, bandpower_features_rel), axis=1)
    scaler = StandardScaler()
    scaler.fit(bandpower_features_wtf)
    bandpower_features_wtf = scaler.transform(bandpower_features_wtf)

    # Trial rejection
    bandpower_features_wtf, labels = trials_rejection(bandpower_features_wtf, labels)

    # seperate the data before feature selection
    indices = np.arange(bandpower_features_wtf.shape[0])
    X_train, X_test, y_train, y_test, train_ind, test_ind = train_test_split(bandpower_features_wtf, labels,indices, random_state=0)

    # Define selection algorithms
    rf_select = SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=800,random_state=0))
    mi_select = SelectKBest(mutual_info_classif, k=int(math.sqrt(data.shape[0])))
    # fisher_select = bandpower_features_wtf[:, fisher_score.fisher_score(bandpower_features_wtf,
    #                                                                     labels)[0:int(math.sqrt(data.shape[0]))]]

    # Define Pipelines
    model = SelectFromModel(LogisticRegression(C=1, penalty="l1", solver='liblinear', random_state=0))
    # define seq selections
    seq_select_clf = SequentialFeatureSelector(clf, n_features_to_select=int(math.sqrt(X_train.shape[0])), n_jobs=1)
    seq_select_RF = SequentialFeatureSelector(rf_classifier, n_features_to_select=int(math.sqrt(X_train.shape[0])), n_jobs=1)
    seq_select_MLP = SequentialFeatureSelector(mlp_classifier, n_features_to_select=int(math.sqrt(X_train.shape[0])),  n_jobs=1)
    seq_select_XGB = SequentialFeatureSelector(xgb_classifier, n_features_to_select=int(math.sqrt(X_train.shape[0])), n_jobs=1)
    seq_select_ADA = SequentialFeatureSelector(ada_classifier, n_features_to_select=int(math.sqrt(X_train.shape[0])), n_jobs=1)

    pipeline_SVM = Pipeline([('lasso', model), ('feat_selecting', seq_select_clf), ('SVM', clf)])
    pipeline_RF = Pipeline([('lasso', model),('feat_selecting', mi_select), ('classify', rf_classifier)])
    pipeline_MLP = Pipeline([('lasso', model),('feat_selecting', mi_select), ('classify', mlp_classifier)])
    pipeline_XGB = Pipeline([('lasso', model),('feat_selecting', mi_select), ('classify', xgb_classifier)])
    pipeline_ADA = Pipeline([('feat_selecting', mi_select),('classify', ada_classifier)])
    # get scores with CV for each pipeline
    scores_mix = cross_val_score(pipeline_SVM, bandpower_features_wtf, labels, cv=5, n_jobs=1)
    scores_mix2 = cross_val_score(pipeline_RF, bandpower_features_wtf, labels, cv=5, n_jobs=1)
    scores_mix3 = cross_val_score(pipeline_MLP, bandpower_features_wtf, labels, cv=5, n_jobs=1)
    scores_mix4 = cross_val_score(pipeline_XGB, bandpower_features_wtf, labels, cv=5, n_jobs=1)
    scores_mix5 = cross_val_score(pipeline_ADA, bandpower_features_wtf, labels, cv=5, n_jobs=1)
    print(scores_mix2)

    #print scores
    (print(f"SVM rate is: {np.mean(scores_mix)*100}%"))
    (print(f"RandomForest rate is: {np.mean(scores_mix2)*100}%"))
    (print(f"MLP rate is: {np.mean(scores_mix3)*100}%"))
    (print(f"XGBC rate is: {np.mean(scores_mix4)*100}%"))
    (print(f"ADA rate is: {np.mean(scores_mix5)*100}%"))

    # fit pipelines for the confusion matrix and get matrices
    pipeline_SVM.fit(bandpower_features_wtf[train_ind, :], np.array(labels)[train_ind])
    ConfusionMatrixDisplay.from_estimator(pipeline_SVM, bandpower_features_wtf[test_ind, :],
                                          np.array(labels)[test_ind])
    plt.show()

    pipeline_RF.fit(bandpower_features_wtf[train_ind, :], np.array(labels)[train_ind])
    print(pipeline_RF.predict(bandpower_features_wtf[test_ind, :]))
    print(np.sum(pipeline_RF.predict(bandpower_features_wtf[test_ind, :])==np.array(labels)[test_ind]))
    ConfusionMatrixDisplay.from_estimator(pipeline_RF, bandpower_features_wtf[test_ind, :],
                                          np.array(labels)[test_ind])
    plt.show()

    pipeline_MLP.fit(bandpower_features_wtf[train_ind, :], np.array(labels)[train_ind])
    ConfusionMatrixDisplay.from_estimator(pipeline_MLP, bandpower_features_wtf[test_ind, :],
                                          np.array(labels)[test_ind])
    plt.show()

    pipeline_XGB.fit(bandpower_features_wtf[train_ind, :], np.array(labels)[train_ind])
    ConfusionMatrixDisplay.from_estimator(pipeline_XGB, bandpower_features_wtf[test_ind, :],
                                          np.array(labels)[test_ind])
    plt.show()

    pipeline_ADA.fit(bandpower_features_wtf[train_ind, :], np.array(labels)[train_ind])
    ConfusionMatrixDisplay.from_estimator(pipeline_ADA,bandpower_features_wtf[test_ind, :],
                                          np.array(labels)[test_ind])
    plt.show()


def get_feature_mat(model):
    # define parameters
    fs = 125
    bands = np.matrix('7 12; 12 15; 17 22; 25 30; 7 35; 30 35')
    # get data
    data = model.epochs.get_data()
    class_labels = model.labels
    feature_labels = []
    # get features
    # CSP
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=3, reg='ledoit_wolf', log=True, norm_trace=False)#, transform_into='average_power', cov_est='epoch')
    csp_features = Pipeline([('CSP', csp), ('LDA', lda)]).fit_transform(data, class_labels)
    [feature_labels.append(f'CSP_Component{i}') for i in range(csp_features.shape[1])]
    # Bandpower
    bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
    [feature_labels.append(f'BP_non_rel{np.ravel(i)}_{chan}') for i in bands for chan in model.epochs.ch_names]
    # relative bandpower
    bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
    [feature_labels.append(f'BP_non_rel{np.ravel(i)}_{chan}') for i in bands for chan in model.epochs.ch_names]
    # hjorthMobility
    hjorthMobility_features = ml_model.MLModel.hjorthMobility(data)
    [feature_labels.append(f'hjorthMobility_{chan}') for chan in model.epochs.ch_names]
    # LZC
    LZC_features = ml_model.MLModel.LZC(data)
    [feature_labels.append(f'LZC_{chan}') for chan in model.epochs.ch_names]
    # DFA
    DFA_features = ml_model.MLModel.DFA(data)
    [feature_labels.append(f'DFA_{chan}') for chan in model.epochs.ch_names]
    # get all of them in one matrix
    features_mat = np.concatenate((csp_features,bandpower_features_new, bandpower_features_rel,
                                   hjorthMobility_features,LZC_features,DFA_features), axis=1)
    scaler = StandardScaler()
    scaler.fit(features_mat)
    return features_mat, class_labels, feature_labels
def plot_SVM(feature_mat,labels):
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


def epochs_z_score(epochs):
    """
    this function is for normalize all the electrods in epochs array by Z_score
    Args:
        epochs: epochs array
    Returns: (ndarray) the data after normalizing all the electrods
    """
    data=epochs.get_data()
    for i in range(len(data)):
        scaler = StandardScaler()
        scaler.fit(data[i].transpose())
        array= scaler.transform(data[i].transpose())
        data[i]= array.transpose()
    return data


if __name__ == '__main__':
    model = pd.read_pickle(r'../recordings/noam/13/unfiltered_model.pickle')
    # playground()
    load_eeg()
    # permutation_func()
    # a, b, c = get_feature_mat(model)
    # plot_SVM(a, b)
