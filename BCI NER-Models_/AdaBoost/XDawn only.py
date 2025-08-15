import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import timeit
import pickle
import sys
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    accuracy_score
)
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import optuna
from optuna import Trial
from optuna.samplers import TPESampler

# Load data
X_test = np.load('/content/drive/MyDrive/2021_VIIT08_P300/Dataset/bci-ner/XDAWN/X_test.npy')
X_train = np.load('/content/drive/MyDrive/2021_VIIT08_P300/Dataset/bci-ner/XDAWN/X_train.npy')
y_train = np.load('/content/drive/MyDrive/2021_VIIT08_P300/Dataset/bci-ner/Y_train.npy')
y_test = np.reshape(
    pd.read_csv('/content/drive/MyDrive/2021_VIIT08_P300/Dataset/bci-ner/true_labels.csv', header=None).values,
    3400
)

print("Shapes:", X_test.shape, X_train.shape, y_train.shape, y_test.shape)

# Create and train AdaBoost Classifier
clf = AdaBoostClassifier(random_state=96)
clf.fit(X_train, y_train)

# Predictions and evaluation
y_preds = clf.predict(X_test)
cls_report = classification_report(y_test, y_preds)
print(cls_report)
print("------------Confusion Matrix---------------")
print(confusion_matrix(y_test, y_preds))

# Optuna optimization
def objective(trial: Trial, X, y) -> float:
    param = {
        "n_estimators": trial.suggest_int('n_estimators', 0, 100),
        "learning_rate": trial.suggest_loguniform('learning_rate', 0.005, 0.5),
    }
    model = AdaBoostClassifier(**param)
    model.fit(X_train, y_train)
    return cross_val_score(model, X_test, y_test).mean()

study = optuna.create_study(direction='maximize', sampler=TPESampler())
study.optimize(lambda trial: objective(trial, X_train, y_train), timeout=600)

print("Best Trial:")
print(study.best_trial)
print("Best Params:", study.best_params)
print("Best Value:", study.best_value)
