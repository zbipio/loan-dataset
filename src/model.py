import numpy as np

from typing import Tuple

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from logger import (
    log_and_stop,
    LOAN_LOGGER,
)


def run_cv(model, X, y, folds=3, cv_type=StratifiedKFold, success_metric=roc_auc_score) -> Tuple:
    """
    Run the specified cross validation on the given model using the given X, y.
    Returns a tuple where:
     - the first item is the mean CV score 
     - the second item is the std of the CV scores
    """
    try:
        cv = cv_type(n_splits=folds, shuffle=True)

        scores = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)

            score = success_metric(y_test, y_pred[:, 1])
            scores.append(score)

        score_mean, score_std = np.mean(scores), np.std(scores)

        LOAN_LOGGER.info('CV on model completed')
        return score_mean, score_std
    except:
        message = 'CV on model NOT completed'
        log_and_stop(LOAN_LOGGER, message)


def train_and_test_model(model, X_train, y_train, X_test, y_test, success_metric=roc_auc_score):
    """
    Train the given model on the given training set and then test it on the given test set.
    Returns a tuple where:
     - the first item is the score achieved on the test set 
     - the second item are the predicted probabilities
    """
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        score = success_metric(y_test, y_pred[:, 1])

        LOAN_LOGGER.info('Model trained on train set and tested on test set')
        return score, y_pred
    except:
        message = 'Model NOT trained on train set and tested on test set'
        log_and_stop(LOAN_LOGGER, message)
