import time
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from preprocessing.feature_extraction import build_text_feature_matrix

try:
    from imblearn.over_sampling import SMOTE, ADASYN
except Exception:
    SMOTE = None
    ADASYN = None

'''Loading dataset'''
path0 = r'~/experiment/data/extracted_features/SO_w2v_200_non_violation.csv'       # negative data
path1 = r'~/experiment/data/extracted_features/SO_w2v_200_violation.csv'           # positive data
# path0 = r'~/experiment/data/extracted_features/FastText_200_non_violation.csv'    # negative data
# path1 = r'~/experiment/data/extracted_features/FastText_200_violation.csv'        # positive data
# path0 = r'~/experiment/data/extracted_features/GloVe_200_non_violation.csv'      # negative data
# path1 = r'~/experiment/data/extracted_features/GloVe_200_violation.csv'          # positive data
# path0 = r'~/experiment/data/extracted_features/FastText_100_non_violation.csv'    # negative data
# path1 = r'~/experiment/data/extracted_features/FastText_100_violation.csv'        # positive data
# path0 = r'~/experiment/data/extracted_features/FastText_300_non_violation.csv'    # negative data
# path1 = r'~/experiment/data/extracted_features/FastText_300_violation.csv'        # positive data

label0_path = r'~/experiment/data/Randomly_selected_comments.xlsx'
label1_path = r'~/experiment/data/Violation symptoms.xlsx'
label0 = pd.read_excel(label0_path, sheet_name='Comments', na_values='n/a')     # label as '0'
label1 = pd.read_excel(label1_path, sheet_name='combination', na_values='n/a')  # label as '1'

percentage = 1/5    # test set: 1/5，training set: 4/5
seed = 5            # int or None
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)  # 10-fold cross validation

def load_balanced_train_data(
    label_path0=label0_path,
    label_path1=label1_path,
    n_per_class=599,
    random_seed=seed,
):
    label_path0 = label_path0 or label0_path
    label_path1 = label_path1 or label1_path
    label0_df = pd.read_excel(os.path.expanduser(label_path0), sheet_name='Comments', na_values='n/a')
    label1_df = pd.read_excel(os.path.expanduser(label_path1), sheet_name='combination', na_values='n/a')
    non_violation = label0_df[['Comment', 'Label']].sample(n=n_per_class, random_state=random_seed)
    violation = label1_df[['Comment', 'Label']].sample(n=n_per_class, random_state=random_seed)
    combined = pd.concat([non_violation, violation], ignore_index=True)
    return combined['Comment'].tolist(), combined['Label'].tolist()


def load_test_data(
    test_path="~/experiment/data/Test set.xlsx",
    sheet_name="Test set",
):
    test_df = pd.read_excel(os.path.expanduser(test_path), sheet_name=sheet_name, na_values='n/a')
    return test_df['Comment'].tolist(), test_df['Label'].tolist()

'''Train classifiers: Build ML classifiers with best parameters'''

def SVM(X, Y, class_weight=None, param_grid=None, cv=10):
    clf = svm.SVC(kernel='rbf', class_weight=class_weight)  # , probability=True)     # Default: kernel='rbf', probability=False
    if param_grid is None:
        param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}
    grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X, Y)
    best_parameters = grid_search.best_estimator_.get_params()
    # print(best_parameters)
    clf = svm.SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True, class_weight=class_weight)
    clf.fit(X, Y)
    print('[SVM] best_parameters', best_parameters)
    return clf

def NB(X, Y, alpha_grid=None, cv=10):
    # clf = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
    clf = BernoulliNB()
    if alpha_grid is None:
        alpha_grid = np.logspace(-2, 1, 10)
    param_grid = {'alpha': alpha_grid}
    grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=cv)
    grid_search.fit(X, Y)
    best_parameters = grid_search.best_estimator_.get_params()
    clf = BernoulliNB(alpha=best_parameters['alpha'], binarize=0.0, fit_prior=True, class_prior=None)
    clf.fit(X, Y)
    print('[NB] best_parameters:', best_parameters)
    return clf

def LR(X, Y, class_weight=None, param_grid=None, cv=10):
    clf = LogisticRegression(class_weight=class_weight)
    # clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)
    if param_grid is None:
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
                      'max_iter': [500, 1000],
                      'solver': ['liblinear']}
    grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=cv, scoring='accuracy')
    grid_search.fit(X, Y)
    best_parameters = grid_search.best_estimator_.get_params()
    clf = LogisticRegression(random_state=seed, C=best_parameters['C'], max_iter=best_parameters['max_iter'], class_weight=class_weight, solver=best_parameters['solver'])
    clf.fit(X, Y)
    print('[LR] best_parameters:', best_parameters)
    return clf


def KNN(X, Y, param_grid=None, cv=10):
    clf = KNeighborsClassifier()
    if param_grid is None:
        param_grid = {'n_neighbors': [i for i in range(1, 31)],
                      'weights': ['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=cv, scoring='accuracy')
    grid_search.fit(X, Y)
    best_parameters = grid_search.best_estimator_.get_params()
    # print(best_parameters)
    clf = KNeighborsClassifier(n_neighbors=best_parameters['n_neighbors'], weights=best_parameters['weights'], algorithm=best_parameters['algorithm'])
    clf.fit(X, Y)
    print('[KNN] best_parameters:', best_parameters)
    return clf


def _sanitize_features(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -100.0, 100.0)
    return X


def _apply_oversampling(X, y, method, random_state):
    if method is None:
        return X, y
    if method == "smote":
        if SMOTE is None:
            raise ImportError("imblearn is required for SMOTE. Install imbalanced-learn.")
        sampler = SMOTE(random_state=random_state)
    elif method == "adasyn":
        if ADASYN is None:
            raise ImportError("imblearn is required for ADASYN. Install imbalanced-learn.")
        sampler = ADASYN(random_state=random_state)
    else:
        raise ValueError(f"Unknown oversample method: {method}")
    return sampler.fit_resample(X, y)

def DT(X, Y, class_weight=None, param_grid=None, cv=10):
    clf = DecisionTreeClassifier(class_weight=class_weight)
    if param_grid is None:
        param_grid = {'criterion': ['gini', 'entropy'],
                      'max_depth': [i for i in range(2, 15, 1)],
                      'min_samples_leaf': [i for i in range(1, 10, 2)],
                      'min_samples_split': [i for i in range(2, 10, 1)]}
    grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=cv, scoring='accuracy')
    grid_search.fit(X, Y)
    best_parameters = grid_search.best_estimator_.get_params()
    # print(best_parameters)
    clf = DecisionTreeClassifier(random_state=seed, criterion=best_parameters['criterion'],
                                 max_depth=best_parameters['max_depth'],
                                 min_samples_leaf=best_parameters['min_samples_leaf'],
                                 min_samples_split=best_parameters['min_samples_split'],
                                 class_weight=class_weight)
    clf.fit(X, Y)
    print('[DT] best_parameters', best_parameters)
    return clf


def cost_sensitive_nb_predict(proba, cost_matrix):
    expected_cost = proba @ cost_matrix
    return expected_cost.argmin(axis=1)


def cost_sensitive_dt_predict(model, X, threshold=0.5, positive_label=1):
    proba = model.predict_proba(X)[:, positive_label]
    return (proba >= threshold).astype(int), proba


def cost_sensitive_knn_predict(model, X, class_weights, threshold=0.3, positive_label=1):
    distances, indices = model.kneighbors(X, return_distance=True)
    neighbor_labels = model._y[indices]
    weighted_votes = np.zeros((X.shape[0], 2))
    for i, labels in enumerate(neighbor_labels):
        for lbl in labels:
            weighted_votes[i, lbl] += class_weights.get(lbl, 1.0)
    positive_scores = weighted_votes[:, positive_label] / weighted_votes.sum(axis=1)
    return (positive_scores >= threshold).astype(int), positive_scores

def run_ml_experiment(
    model_name,
    text_features=False,
    embeding_model="FastText",
    add_stats=True,
    class_weight=None,
    cost_matrix=None,
    dt_threshold=0.5,
    knn_threshold=0.3,
    knn_class_weights=None,
    decision_threshold=0.5,
    oversample_method=None,
    cv=10,
    param_grid=None,
    label_path0=label0_path,
    label_path1=label1_path,
    random_seed=seed,
    n_per_class=599,
    test_path="~/experiment/data/Test set.xlsx",
    test_sheet="combination",
    cv_folds=10,
):
    train_texts, train_labels = load_balanced_train_data(
        label_path0=label_path0,
        label_path1=label_path1,
        n_per_class=n_per_class,
        random_seed=random_seed,
    )
    test_texts, test_labels = load_test_data(test_path=test_path, sheet_name=test_sheet)

    if text_features:
        X_train_val = build_text_feature_matrix(train_texts, embeding_model, add_stats=add_stats)
        X_test = build_text_feature_matrix(test_texts, embeding_model, add_stats=add_stats)
    else:
        raise ValueError("text_features must be True for balanced training in this workflow.")

    X_train_val = _sanitize_features(X_train_val)
    X_test = _sanitize_features(X_test)

    y_train_val = np.array(train_labels)
    y_test = np.array(test_labels)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    fold_metrics = []
    scaler = StandardScaler()

    for train_idx, val_idx in skf.split(X_train_val, y_train_val):
        X_train = scaler.fit_transform(X_train_val[train_idx])
        X_val = scaler.transform(X_train_val[val_idx])
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_train = np.clip(X_train, -5.0, 5.0)
        X_val = np.clip(X_val, -5.0, 5.0)
        y_train = y_train_val[train_idx]
        y_val = y_train_val[val_idx]
        X_train, y_train = _apply_oversampling(
            X_train, y_train, oversample_method, random_state=random_seed
        )

        if model_name == "SVM":
            base = SVM(X_train, y_train, class_weight=class_weight, param_grid=param_grid, cv=cv)
            clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            clf.fit(X_train, y_train)
            val_proba = clf.predict_proba(X_val)[:, 1]
            val_pred = (val_proba >= decision_threshold).astype(int)
        elif model_name == "LR":
            clf = LR(X_train, y_train, class_weight=class_weight, param_grid=param_grid, cv=cv)
            val_proba = clf.predict_proba(X_val)[:, 1]
            val_pred = (val_proba >= decision_threshold).astype(int)
        elif model_name == "NB":
            clf = NB(X_train, y_train, alpha_grid=param_grid, cv=cv)
            val_proba = clf.predict_proba(X_val)[:, 1]
            if cost_matrix is None:
                val_pred = np.argmax(clf.predict_proba(X_val), axis=1)
            else:
                val_pred = cost_sensitive_nb_predict(clf.predict_proba(X_val), cost_matrix)
        elif model_name == "DT":
            clf = DT(X_train, y_train, class_weight=class_weight, param_grid=param_grid, cv=cv)
            val_pred, val_proba = cost_sensitive_dt_predict(clf, X_val, threshold=dt_threshold)
        elif model_name == "KNN":
            clf = KNN(X_train, y_train, param_grid=param_grid, cv=cv)
            if knn_class_weights is None:
                knn_class_weights = {0: 1.0, 1: 1.0}
            val_pred, val_proba = cost_sensitive_knn_predict(
                clf, X_val, knn_class_weights, threshold=knn_threshold
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        fold_metrics.append(
            {
                "precision": metrics.precision_score(y_val, val_pred, zero_division=0),
                "recall": metrics.recall_score(y_val, val_pred, zero_division=0),
                "f1": metrics.f1_score(y_val, val_pred, zero_division=0),
                "accuracy": metrics.accuracy_score(y_val, val_pred),
                "macro_f1": metrics.f1_score(y_val, val_pred, average='macro', zero_division=0),
                "auc_pr": average_precision_score(y_val, val_proba),
            }
        )

    X_train_val_std = scaler.fit_transform(X_train_val)
    X_test_std = scaler.transform(X_test)
    X_train_val_std = np.nan_to_num(X_train_val_std, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_std = np.nan_to_num(X_test_std, nan=0.0, posinf=0.0, neginf=0.0)
    X_train_val_std = np.clip(X_train_val_std, -5.0, 5.0)
    X_test_std = np.clip(X_test_std, -5.0, 5.0)
    y_train_val_res = y_train_val
    X_train_val_res = X_train_val_std
    X_train_val_res, y_train_val_res = _apply_oversampling(
        X_train_val_res, y_train_val_res, oversample_method, random_state=random_seed
    )

    if model_name == "SVM":
        base = SVM(X_train_val_res, y_train_val_res, class_weight=class_weight, param_grid=param_grid, cv=cv)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        clf.fit(X_train_val_res, y_train_val_res)
        test_proba = clf.predict_proba(X_test_std)[:, 1]
        test_pred = (test_proba >= decision_threshold).astype(int)
    elif model_name == "LR":
        clf = LR(X_train_val_res, y_train_val_res, class_weight=class_weight, param_grid=param_grid, cv=cv)
        test_proba = clf.predict_proba(X_test_std)[:, 1]
        test_pred = (test_proba >= decision_threshold).astype(int)
    elif model_name == "NB":
        clf = NB(X_train_val_res, y_train_val_res, alpha_grid=param_grid, cv=cv)
        test_proba = clf.predict_proba(X_test_std)[:, 1]
        if cost_matrix is None:
            test_pred = np.argmax(clf.predict_proba(X_test_std), axis=1)
        else:
            test_pred = cost_sensitive_nb_predict(clf.predict_proba(X_test_std), cost_matrix)
    elif model_name == "DT":
        clf = DT(X_train_val_res, y_train_val_res, class_weight=class_weight, param_grid=param_grid, cv=cv)
        test_pred, test_proba = cost_sensitive_dt_predict(clf, X_test_std, threshold=dt_threshold)
    elif model_name == "KNN":
        clf = KNN(X_train_val_res, y_train_val_res, param_grid=param_grid, cv=cv)
        if knn_class_weights is None:
            knn_class_weights = {0: 1.0, 1: 1.0}
        test_pred, test_proba = cost_sensitive_knn_predict(
            clf, X_test_std, knn_class_weights, threshold=knn_threshold
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    test_metrics = {
        "precision": metrics.precision_score(y_test, test_pred, zero_division=0),
        "recall": metrics.recall_score(y_test, test_pred, zero_division=0),
        "f1": metrics.f1_score(y_test, test_pred, zero_division=0),
        "accuracy": metrics.accuracy_score(y_test, test_pred),
        "macro_f1": metrics.f1_score(y_test, test_pred, average='macro', zero_division=0),
        "auc_pr": average_precision_score(y_test, test_proba),
    }

    return {
        "fold_metrics": fold_metrics,
        "test_metrics": test_metrics,
    }


if __name__ == '__main__':
    metrics_out = run_ml_experiment("KNN")
    print(metrics_out)
