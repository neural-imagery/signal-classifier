import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from scipy.stats import linregress
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)
from sklearn.model_selection import (StratifiedKFold, GroupKFold, GridSearchCV,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

from src import featurize

OUTER_K = 5
INNER_K = 3

# Standard machine learning parameters
MAX_ITER = 250000  # for support vector classifier
C_LIST = [1e-3, 1e-2, 1e-1, 1e0]
N_NEIGHBORS_LIST = list(range(1, 10))

def machine_learn(nirs, labels, groups, model, normalize=False,
                  random_state=None, output_folder='./outputs'):
    """
    Perform nested k-fold cross-validation for standard machine learning models
    producing metrics and confusion matrices. The models include linear
    discriminant analysis (LDA), support vector classifier (SVC) with grid
    search for the regularization parameter (inner cross-validation), and
    k-nearest neighbors (kNN) with grid search for the number of neighbors
    (inner cross-validation).

    Parameters
    ----------
    nirs : array of shape (n_epochs, n_features)
        Processed NIRS data.

    labels : array of integers
        List of labels.

    groups : array of integers | None
        List of subject ID matching the epochs to perfrom a group k-fold
        cross-validation. If ``None``, performs a stratified k-fold
        cross-validation instead.

    model : string
        Standard machine learning to use. Either ``'lda'`` for a linear
        discriminant analysis, ``'svc'`` for a linear support vector
        classifier or ``'knn'`` for a k-nearest neighbors classifier.

    normalize : boolean
        Whether to normalize data before feeding to the model with min-max
        scaling based on the train set for each iteration of the outer
        cross-validation. Defaults to ``False`` for no normalization.

    random_state : integer | None
        Controls the shuffling applied to data. Pass an integer for
        reproducible output across multiple function calls. Defaults to
        ``None`` for not setting the seed.

    output_folder : string
        Path to the directory into which the figures will be saved. Defaults to
        ``'./outputs'``.

    Returns
    -------
    clf : Sklearn Classifier
        Sklearn object for classifying data.

    accuracies : list of floats
        List of accuracies on the test sets (one for each iteration of the
        outer cross-validation).

    all_hps : list of floats | list of None
        List of regularization parameters for the SVC or a list of None for the
        LDA (one for each iteration of the outer cross-validation).

    additional_metrics : list of tuples
        List of tuples of metrics composed of (precision, recall, F1 score) on
        the outer cross-validation (one tuple for each iteration of the outer
        cross-validation). This uses the ``precision_recall_fscore_support``
        function from scikit-learn with ``average='micro'``, ``y_true`` and
        ``y_pred`` being the true and the predictions on the specific iteration
        of the outer cross-validation.
    """
    
    print(f'Machine learning: {model}')

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # K-fold cross-validator
    if groups is None:
        out_kf = StratifiedKFold(n_splits=OUTER_K, shuffle=True, random_state=random_state)
        in_kf = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=random_state)
    else:
        out_kf = GroupKFold(n_splits=OUTER_K)
        in_kf = GroupKFold(n_splits=INNER_K)
    all_y_true = []
    all_y_pred = []
    accuracies = []
    additional_metrics = []
    all_hps = []
    classifiers = []
    out_split = out_kf.split(nirs, labels, groups)
    for k, out_idx in enumerate(out_split):
        print("out_idx", out_idx)
        print(f'\tFOLD #{k+1}')
        nirs_train, nirs_test = nirs[out_idx[0]], nirs[out_idx[1]]
        labels_train, labels_test = labels[out_idx[0]], labels[out_idx[1]]

        if groups is None:
            groups_train = None
            nirs_train, labels_train = shuffle(
                nirs_train, labels_train, random_state=random_state)
        else:
            groups_train = groups[out_idx[0]]
            nirs_train, labels_train, groups_train = shuffle(
                nirs_train, labels_train, groups_train,
                random_state=random_state)

        all_y_true += labels_test.tolist()

        # Min-max scaling
        if normalize:
            maxs = nirs_train.max(axis=0)[np.newaxis, :]
            mins = nirs_train.min(axis=0)[np.newaxis, :]
            nirs_train = (nirs_train - mins) / (maxs - mins)
            nirs_test = (nirs_test - mins) / (maxs - mins)

        in_split = in_kf.split(nirs_train, labels_train, groups_train)

        # LDA
        if model == 'lda':
            clf = LinearDiscriminantAnalysis()
            clf.fit(nirs_train, labels_train)
            y_pred = clf.predict(nirs_test).tolist()
            all_hps.append(None)

        # SVC
        elif model == 'svc':
            parameters = {'C': C_LIST}
            svc = LinearSVC(max_iter=MAX_ITER)
            clf = GridSearchCV(svc, parameters, scoring='accuracy',
                               cv=in_split)
            clf.fit(nirs_train, labels_train)
            y_pred = clf.predict(nirs_test).tolist()
            all_hps.append(clf.best_params_['C'])

        # kNN
        elif model == 'knn':
            parameters = {'n_neighbors': N_NEIGHBORS_LIST}
            knn = KNeighborsClassifier()
            clf = GridSearchCV(knn, parameters, scoring='accuracy',
                               cv=in_split)
            clf.fit(nirs_train, labels_train)
            y_pred = clf.predict(nirs_test).tolist()
            all_hps.append(clf.best_params_['n_neighbors'])

        # ann
        elif model == "ann":
            clf = MLPClassifier(solver='adam', hidden_layer_sizes=(32,64,1200,64,32))
            clf.fit(nirs_train, labels_train)
            y_pred = clf.predict(nirs_test).tolist()
            all_hps.append(None)

        # Metrics
        classifiers.append(clf)
        acc = accuracy_score(labels_test, y_pred)
        print("acc", acc, "test labels", labels_test, "predictions", y_pred)
        accuracies.append(acc)
        prfs = precision_recall_fscore_support(labels_test, y_pred,
                                               average='micro')
        additional_metrics.append(prfs[:-1])
        all_y_pred += y_pred

    # Figures
    cm = confusion_matrix(all_y_true, all_y_pred, normalize='true')
    sns.heatmap(cm, annot=True, cmap='crest', vmin=0.1, vmax=0.8)
    plt.xlabel('Predicted')
    plt.ylabel('Ground truth')
    plt.savefig(f'{output_folder}/confusion_matrix.png')
    plt.close()

    return classifiers, accuracies, all_hps, additional_metrics
