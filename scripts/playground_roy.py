# This script is meant to load models and allow the user to change hyper-parameters
# so you could fine-tune the real offline_training class
import copy
import math
from tkinter import filedialog, Tk
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, SelectFromModel
import scipy
import scipy.io
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from skfeature.function.similarity_based import fisher_score
from bci4als import ml_model
from sklearn import svm
from sklearn.model_selection import cross_val_score, ShuffleSplit, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


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

    # Our data
    data2 = pd.read_pickle(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\roy\20\trained_model.pickle')
    #

    labels = data2.labels
    data = data2.epochs.get_data()

    rf_classifier = RandomForestClassifier(random_state=0)
    mlp_classifier = OneVsRestClassifier(MLPClassifier(solver='adam',hidden_layer_sizes=[80]*5))
    xgb_classifier = OneVsRestClassifier(XGBClassifier())
    ada_classifier = KNeighborsClassifier()

    # # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=3, reg='ledoit_wolf', log=True, norm_trace=False, transform_into='average_power', cov_est='epoch')
    csp_features = Pipeline([('CSP', csp), ('LDA', lda)]).fit_transform(data, labels)
    bandpower_features_new = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=False)
    bandpower_features_rel = ml_model.MLModel.bandpower(data, bands, fs, window_sec=0.5, relative=True)
    hjorthMobility_features = ml_model.MLModel.hjorthMobility(data)
    LZC_features = ml_model.MLModel.LZC(data)
    DFA_features = ml_model.MLModel.DFA(data)
    bandpower_features_wtf = np.concatenate((csp_features,bandpower_features_new, bandpower_features_rel,
                 hjorthMobility_features,LZC_features,DFA_features), axis=1)
    # scaler = StandardScaler()
    # scaler.fit(bandpower_features_wtf)
    # bandpower_features_wtf = scaler.transform(bandpower_features_wtf)
    for feat_num in range(1, int(math.sqrt(data.shape[0]))):
        # bandpower_features_selected = SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=80)).fit_transform(bandpower_features_wtf, labels)
        # bandpower_features_selected = SelectKBest(mutual_info_classif, k=feat_num).fit_transform(bandpower_features_wtf, labels)
        # bandpower_features_selected = bandpower_features_wtf[:,fisher_score.fisher_score(bandpower_features_wtf,labels)[0:feat_num ]]
        #lasso
        # logistic = LogisticRegression(C=1, penalty="l1", solver='liblinear', random_state=0).fit(bandpower_features_wtf,labels)
        # model = SelectFromModel(logistic, prefit=True)
        # bandpower_features_selected = model.transform(bandpower_features_wtf)
        # bandpower_features_selected = bandpower_features_selected[:,np.round(np.var(bandpower_features_selected,axis=0))!=0]
        # different lasso
        X_train, X_test, y_train, y_test = train_test_split(bandpower_features_wtf, labels, random_state = 0)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso())
        ])
        search = GridSearchCV(pipeline,
                              {'model__alpha': np.arange(0.1, 10, 0.1)},
                              cv=5, scoring="neg_mean_squared_error", verbose=3
                              )
        search.fit(X_train, y_train)
        search.best_params_
        coefficients = search.best_estimator_.named_steps['model'].coef_
        importance = np.abs(coefficients)
        X_train = X_train[:, importance != 0]
        X_test = X_test[:, importance != 0]
        # bandpower_features_selected = SelectKBest(mutual_info_classif, k=int(math.sqrt(data.shape[0]))).fit_transform(bandpower_features_selected, labels)
        # scaler = StandardScaler()
        # scaler.fit(bandpower_features_selected)
        # bandpower_features_selected = scaler.transform(bandpower_features_selected)
        # print(bandpower_features_selected.shape)
        # scores_mix = cross_val_score(clf, bandpower_features_selected, labels, cv=5, n_jobs=1)
        # scores_mix2 = cross_val_score(rf_classifier, bandpower_features_selected, labels, cv=5, n_jobs=1)
        # scores_mix3 = cross_val_score(mlp_classifier, bandpower_features_selected, labels, cv=5, n_jobs=1)
        # scores_mix4 = cross_val_score(xgb_classifier, bandpower_features_selected, labels, cv=5, n_jobs=1)
        # scores_mix5 = cross_val_score(ada_classifier, bandpower_features_selected, labels, cv=5, n_jobs=1)

        clf.fit(X_train, y_train)
        rf_classifier.fit(X_train, y_train)
        mlp_classifier.fit(X_train, y_train)
        xgb_classifier.fit(X_train, y_train)
        ada_classifier.fit(X_train, y_train)
        ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, normalize='true')
        plt.show()
        ConfusionMatrixDisplay.from_estimator(rf_classifier, X_test, y_test, normalize='true')
        plt.show()
        ConfusionMatrixDisplay.from_estimator(mlp_classifier, X_test, y_test, normalize='true')
        plt.show()
        ConfusionMatrixDisplay.from_estimator(xgb_classifier, X_test, y_test, normalize='true')
        plt.show()
        ConfusionMatrixDisplay.from_estimator(ada_classifier, X_test, y_test, normalize='true')
        plt.show()

        # (print(f"SVM rate is: {np.mean(scores_mix)*100}%"))
        # (print(f"RandomForest rate is: {np.mean(scores_mix2)*100}%"))
        # (print(f"MLP rate is: {np.mean(scores_mix3)*100}%"))
        # (print(f"XGBC rate is: {np.mean(scores_mix4)*100}%"))
        # (print(f"ADA rate is: {np.mean(scores_mix5)*100}%"))

        # if np.mean(scores_mix)*100 > max_score:
        #     max_score = np.mean(scores_mix)*100
        #     feat_num_max = feat_num
    # print(max_score, feat_num_max)
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
def permutation_func():
    fs = 125
    bands = np.matrix('7 12; 12 15; 17 22; 25 30; 7 35; 30 35')
    max_score = 1
    clf = svm.SVC(decision_function_shape='ovo', kernel='linear')
    data2 = pd.read_pickle(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\roy\4\unfiltered_model.pickle')
    # data2 = pd.read_pickle(r'C:\Users\pc\Desktop\bci4als\recordings\roy\2\unfiltered_model.pickle')
    labels = data2.labels
    # data2.epochs.filter(1., 40., fir_design='firwin', skip_by_annotation='edge', verbose=False)
    data = data2.epochs.get_data()
    # perm_c3 = (0, 5, 3, 9, 7, 1, 4, 6, 8, 10)
    # perm_c3 = (0, 3, 5, 9, 7, 1, 4, 6, 8, 10)
    perm_c3 = (0, 1, 2, 3, 5, 6, 4, 7, 8, 9, 10)

    for trial in range(data.shape[0]):
        # C3
        data[trial][perm_c3[0]] -= (data[trial][perm_c3[1]] + data[trial][perm_c3[2]] + data[trial][perm_c3[3]] +
                              data[trial][perm_c3[4]]) / 4
        # data[trial][perm_c3[0]] = (data[trial][perm_c3[0]] - data[trial][perm_c3[0]].mean()) - \
        #                           (((data[trial][perm_c3[1]] - data[trial][perm_c3[1]].mean())
        #                             + (data[trial][perm_c3[2]] - data[trial][perm_c3[2]].mean())
        #                             + (data[trial][perm_c3[3]] - data[trial][perm_c3[3]].mean())
        #                             + (data[trial][perm_c3[4]] - data[trial][perm_c3[4]].mean())) / 4)
        # C4
        data[trial][perm_c3[5]] -= (data[trial][perm_c3[6]] + data[trial][perm_c3[7]] + data[trial][perm_c3[8]] +
                              data[trial][perm_c3[9]]) / 4
        # data[trial][perm_c3[5]] = (data[trial][perm_c3[5]] - data[trial][perm_c3[5]].mean()) - \
        #                           (((data[trial][perm_c3[6]] - data[trial][perm_c3[6]].mean())
        #                             + (data[trial][perm_c3[7]] - data[trial][perm_c3[7]].mean())
        #                             + (data[trial][perm_c3[8]] - data[trial][perm_c3[8]].mean())
        #                             + (data[trial][perm_c3[9]] - data[trial][perm_c3[9]].mean())) / 4)
        new_data = np.delete(data[trial], [perm_c3[point] for point in [1, 2, 3, 4, 6, 7, 8, 9]], axis=0)
        # new_data = data[trial]
        if trial == 0:
            final_data = new_data[np.newaxis]
        else:
            final_data = np.vstack((final_data, new_data[np.newaxis]))
    data = final_data
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
    bandpower_features_wtf = SelectKBest(mutual_info_classif, k=9).fit_transform(bandpower_features_wtf, labels)
    scores_mix = cross_val_score(clf, bandpower_features_wtf, labels, cv=8)
    print(np.mean(scores_mix)*100)
    if np.mean(scores_mix)*100 > max_score:
        max_score = np.mean(scores_mix)*100
        feat_num_max = 9
        print(f"Prediction rate is: {np.mean(scores_mix) * 100}%")
        print(max_score, feat_num_max)
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

def ICA(ufiltered_model):
    epochs = ufiltered_model.epochs
    ica = ICA(n_components=15, method='fastica', max_iter="auto").fit(epochs)

if __name__ == '__main__':
    model = pd.read_pickle(r'C:\Users\User\Desktop\ALS_BCI\team13\bci4als-master\bci4als\recordings\roy\3\unfiltered_model.pickle')
    # playground()
    load_eeg()
    # permutation_func()
    # a, b, c = get_feature_mat(model)
    # plot_SVM(a, b)
