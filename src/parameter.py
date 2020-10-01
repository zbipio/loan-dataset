from functools import partial

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from logger import (
    log_and_stop,
    LOAN_LOGGER,
)


AVAILABLE_MODELS = {
    'LGBM': LGBMClassifier,
    'XGB': XGBClassifier,
    'DecisionTree': DecisionTreeClassifier
}


def _run_boosting_hyperopt(selected_model, X_train, y_train, X_test, y_test, max_evals):
    """
    Run Hyperopt for the LGBM or XGB models. The models are trained and tested on the given X and y.
    The score metric is ROC_AUC.
    tpe.suggest has been modified, so that only the first 3 tries are random, instead of the default 20.
    The function returns a tuple where:
        - the first item is a dictionary returned by the fmin function
        - the second item is the trials variable used in hyperopt
    """
    def objective(space):
        model_params = {
            'colsample_bytree': space['colsample_bytree'],
            'learning_rate': space['learning_rate'],
            'max_depth': int(space['max_depth']),
            'min_child_weight': int(space['min_child_weight']),
            'n_estimators': int(space['n_estimators']),
            'reg_alpha': space['reg_alpha'],
            'reg_lambda': space['reg_lambda'],
            'subsample': space['subsample'],
            'num_leaves': 20,
            'random_state': 2020,
            'importance_type': 'gain',
            'n_jobs': -1
        }

        model = selected_model(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        score = -roc_auc_score(y_test, y_pred[:, 1])

        return {'loss': score, 'status': STATUS_OK}
    try:
        space = {
            'max_depth': hp.quniform('ho_max_depth', 5, 20, 1),
            'colsample_bytree': hp.uniform('ho_colsample_bytree', 0.8, 1.),
            'learning_rate': hp.uniform('ho_learning_rate', 0.05, 0.2),
            'subsample': hp.uniform('ho_subsample', 0.7, 1.),
            'min_child_weight': hp.quniform('ho_min_child_weight', 1, 10, 1),
            'reg_alpha': hp.loguniform('ho_reg_alpha', 0., 1.),
            'reg_lambda': hp.uniform('ho_reg_lambda', 0.7, 1.),
            'n_estimators': hp.quniform('ho_n_estimators', 50, 500, 5)
        }

        trials = Trials()
        best_params = fmin(fn=objective,
                           space=space,
                           algo=partial(tpe.suggest, n_startup_jobs=3),
                           max_evals=max_evals,
                           trials=trials)

        LOAN_LOGGER.info('Boosting Hyperopt finished successfully')
        return best_params, trials
    except:
        message = 'Boosting Hyperopt NOT finished successfully'
        log_and_stop(LOAN_LOGGER, message)


def _run_tree_hyperopt(selected_model, X_train, y_train, X_test, y_test, max_evals):
    """
    Run Hyperopt for the DecisionTree model. The model is trained and tested on the given X and y.
    The score metric is ROC_AUC.
    tpe.suggest has been modified, so that only the first 3 tries are random, instead of the default 20.
    The function returns a tuple where:
        - the first item is a dictionary returned by the fmin function
        - the second item is the trials variable used in hyperopt
    """
    def objective(space):
        model_params = {
            'max_depth': int(space['max_depth']),
            'min_samples_split': int(space['min_samples_split']),
            'min_samples_leaf': int(space['min_samples_leaf']),
            'random_state': 2020
        }

        model = selected_model(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        score = -roc_auc_score(y_test, y_pred[:, 1])

        return {'loss': score, 'status': STATUS_OK}
    try:
        space = {
            'max_depth': hp.quniform('ho_max_depth', 5, 20, 1),
            'min_samples_split': hp.quniform('ho_min_samples_split', 2, 10, 1),
            'min_samples_leaf': hp.quniform('ho_min_samples_leaf', 1, 10, 1),
        }

        trials = Trials()
        best_params = fmin(fn=objective,
                           space=space,
                           algo=partial(tpe.suggest, n_startup_jobs=3),
                           max_evals=max_evals,
                           trials=trials)

        LOAN_LOGGER.info('Tree Hyperopt finished successfully')
        return best_params, trials
    except:
        message = 'Tree Hyperopt NOT finished successfully'
        log_and_stop(LOAN_LOGGER, message)


def get_model_params(selected_model, X_train, y_train, X_test, y_test, hyperopt=False, max_evals=10):
    """
    Retrieve parameters for the selected model, either by using the hyperopt algorithm or loading a given set of parameters.
    Hyperopt accepts only the XGB, LGBM or DecisionTree model.
    The given set of parameters was found using hyperopt.
    The function returns a tuple where:
        - the first item is a dictionary of model parameters ready to be used in a model
        - the second item is either the trials variable if hyperopt was done, or the string `No hyperopt done` otherwise
    """
    try:
        global AVAILABLE_MODELS
        assert (str(selected_model) in AVAILABLE_MODELS.keys()), 'Allowed models are {}'.format(AVAILABLE_MODELS.keys())

        if selected_model in ('LGBM', 'XGB'):
            model_type = 'Boosting'
        else:
            model_type = 'DecisionTree'

        LOAN_LOGGER.info('Correct model chosen for parameter retrieval')
    except:
        message = 'Correct model NOT chosen for parameter retrieval'
        log_and_stop(LOAN_LOGGER, message)
    else:
        if hyperopt:
            if model_type == 'Boosting':
                best_params, trials = _run_boosting_hyperopt(AVAILABLE_MODELS[selected_model], X_train, y_train, X_test, y_test, max_evals)
                params_dict = {
                    'colsample_bytree': best_params['ho_colsample_bytree'],
                    'learning_rate': best_params['ho_learning_rate'],
                    'max_depth': int(best_params['ho_max_depth']),
                    'min_child_weight': int(best_params['ho_min_child_weight']),
                    'n_estimators': int(best_params['ho_n_estimators']),
                    'reg_alpha': best_params['ho_reg_alpha'],
                    'reg_lambda': best_params['ho_reg_lambda'],
                    'subsample': best_params['ho_subsample'],
                    'num_leaves': 20,
                    'random_state': 2020,
                    'importance_type': 'gain',
                    'n_jobs': -1
                }
            else:
                best_params, trials = _run_tree_hyperopt(AVAILABLE_MODELS[selected_model], X_train, y_train, X_test, y_test, max_evals)
                params_dict = {
                    'max_depth': int(best_params['ho_max_depth']),
                    'min_samples_leaf': int(best_params['ho_min_samples_leaf']),
                    'min_samples_split': int(best_params['ho_min_samples_split']),
                    'random_state': 2020
                }

            return params_dict, trials
        else:
            params_dict = {
                'Boosting': {
                    'colsample_bytree': 0.8899759555042142,
                    'learning_rate': 0.09532621848124778,
                    'max_depth': 11,
                    'min_child_weight': 4,
                    'n_estimators': 215,
                    'reg_alpha': 2.016992556501955,
                    'reg_lambda': 0.7643883757438669,
                    'subsample': 0.7651869713043127,
                    'num_leaves': 20,
                    'random_state': 2020,
                    'importance_type': 'gain',
                    'n_jobs': -1
                },

                'DecisionTree': {
                    'max_depth': 11,
                    'min_samples_leaf': 8,
                    'min_samples_split': 10,
                    'random_state': 2020
                },
            }

            return params_dict[model_type], 'No hyperopt done'
