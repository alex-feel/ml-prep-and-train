import argparse
import json
import os
import warnings
import warnings as warn
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from keras.layers import Dense
from keras.models import Sequential
from lightgbm import LGBMClassifier
from lime import lime_tabular
from scikeras.wrappers import KerasClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import preprocess_data as dp

# Don't change this constant
# If the model does not support the `predict_proba` method, then when training it to create a stack model, you need
# to use CalibratedClassifierCV, for other models, as a rule, on the contrary, using CalibratedClassifierCV
# can worsen the results
MODEL_FEATURES_SUPPORT = {
    'GaussianNB': {
        'predict_proba': True,
        'class_weighting': False,
        'feature_importance': False
    },
    'KNeighborsClassifier': {
        'predict_proba': True,
        'class_weighting': False,
        'feature_importance': False
    },
    'LogisticRegression': {
        'predict_proba': True,
        'class_weighting': True,
        'feature_importance': True
    },
    'LinearDiscriminantAnalysis': {
        'predict_proba': True,
        'class_weighting': False,
        'feature_importance': False
    },
    'QuadraticDiscriminantAnalysis': {
        'predict_proba': True,
        'class_weighting': False,
        'feature_importance': False
    },
    'SVC': {
        'predict_proba': True,  # If you set probability=True
        'class_weighting': True,
        'feature_importance': False
    },
    'RandomForestClassifier': {
        'predict_proba': True,
        'class_weighting': True,
        'feature_importance': True
    },
    'XGBClassifier': {
        'predict_proba': True,
        'class_weighting': True,
        'feature_importance': True
    },
    'LGBMClassifier': {
        'predict_proba': True,
        'class_weighting': True,
        'feature_importance': True
    },
    'GradientBoostingClassifier': {
        'predict_proba': True,
        'class_weighting': False,
        'feature_importance': True
    },
    'CatBoostClassifier': {
        'predict_proba': True,
        'class_weighting': True,
        'feature_importance': True
    },
    'AdaBoostClassifier': {
        'predict_proba': True,
        'class_weighting': False,
        'feature_importance': True
    },
    'ExtraTreesClassifier': {
        'predict_proba': True,
        'class_weighting': True,
        'feature_importance': True
    },
    'MLPClassifier': {
        'predict_proba': True,
        'class_weighting': False,
        'feature_importance': False
    },
    'RidgeClassifier': {
        'predict_proba': False,
        'class_weighting': True,
        'feature_importance': False
    },
    'Perceptron': {
        'predict_proba': False,
        'class_weighting': True,
        'feature_importance': False
    },
    'PassiveAggressiveClassifier': {
        'predict_proba': False,
        'class_weighting': True,
        'feature_importance': False
    },
    'BaggingClassifier': {
        'predict_proba': True,
        # It supports class weighting if the base estimator supports class_weight
        'class_weighting': True,
        # It supports feature importance for each base estimator if the base estimator supports feature importance
        'feature_importance': False
    }
}


# Defining the hyperparameter search space
def create_param_grids(x_train: np.ndarray, config: Dict) -> Dict:
    """
    Creates parameter grids for hyperparameter search based on the provided training data and configuration.
    The grids are constructed for different base model names listed in the configuration, including parameters
    and ranges for classifiers such as GaussianNB, KNeighborsClassifier, LogisticRegression, and others.

    Args:
        x_train: The input training data as a numpy array.
        config: Configuration dictionary containing the base model names and number of iterations for the search.

    Returns:
        A dictionary containing the parameter grids for each base model name specified in the configuration.
    """
    base_model_names = config['training']['base_model_names']
    num_iter = config['training']['num_iter']

    param_grids = {}

    if 'GaussianNB' in base_model_names:
        # Generating enough values for GaussianNB uniformly distributed on a logarithmic scale
        num_values = int(num_iter * 3.5)
        var_smoothing_values = np.logspace(-15, 0, num=num_values)
        param_grids['GaussianNB'] = [
            {
                'var_smoothing': var_smoothing_values
            }
        ]

    if 'KNeighborsClassifier' in base_model_names:
        # Calculating of the inverse covariance matrix VI for KNeighborsClassifier
        cov_matrix = np.cov(x_train.T)
        vi = np.linalg.inv(cov_matrix)
        param_grids['KNeighborsClassifier'] = [
            {
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
                'n_neighbors': list(range(1, 11)),
                'p': [1, 2, 3, 4],
                'weights': ['uniform', 'distance']
            },
            {
                'algorithm': ['brute'],
                'metric': ['mahalanobis'],
                'metric_params': [{'VI': vi}],
                'n_neighbors': list(range(1, 11)),
                'weights': ['uniform', 'distance']
            }
        ]

    if 'LogisticRegression' in base_model_names:
        param_grids['LogisticRegression'] = [
            {
                'penalty': ['l1'],
                'C': np.logspace(-20, 10, 100),
                'solver': ['liblinear', 'saga'],
                'max_iter': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 10000],
            },
            {
                'penalty': ['l2'],
                'C': np.logspace(-20, 10, 100),
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 10000],
            },
            {
                'penalty': ['elasticnet'],
                'C': np.logspace(-20, 10, 100),
                'solver': ['saga'],
                'l1_ratio': np.linspace(0, 1, 10),
                'max_iter': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 10000],
            },
            {
                'penalty': [None],
                'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                'max_iter': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 10000],
            }
        ]

    if 'LinearDiscriminantAnalysis' in base_model_names:
        # When using the eigen method, errors may occur when the algorithm cannot find the correct expansion of
        # the covariance matrix for the data
        param_grids['LinearDiscriminantAnalysis'] = [
            {
                'solver': ['svd'], 'shrinkage': [None]
            },
            {
                'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto'] + list(np.arange(0, 1.1, 0.1))
            }
        ]

    if 'QuadraticDiscriminantAnalysis' in base_model_names:
        param_grids['QuadraticDiscriminantAnalysis'] = [
            {
                'reg_param': list(np.arange(0, 1.1, 0.1))
            }
        ]

    if 'SVC' in base_model_names:
        param_grids['SVC'] = [
            {
                'C': [0.1, 1, 10, 20],
                'degree': [2, 3, 4],
                'gamma': ['scale', 'auto'],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            }
        ]

    if 'RandomForestClassifier' in base_model_names:
        param_grids['RandomForestClassifier'] = [
            {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, None],
                'max_features': [2, 3, 'sqrt'],
                'min_samples_leaf': [1, 5, 10],
                'min_samples_split': [1, 2, 5],
                'n_estimators': [50, 100, 150, 200, 250]
            }
        ]

    if 'XGBClassifier' in base_model_names:
        param_grids['XGBClassifier'] = [
            {
                'colsample_bytree': [0.5, 0.8, 1],
                'gamma': [0, 1, 5],
                'learning_rate': [0.01, 0.1, 1],
                'max_depth': [3, 5, 10, None],
                'min_child_weight': [1, 5, 10],
                'n_estimators': [10, 30, 50, 100],
                'subsample': [0.5, 0.8, 1]
            }
        ]

    if 'LGBMClassifier' in base_model_names:
        param_grids['LGBMClassifier'] = [
            {
                'colsample_bytree': [0.3, 0.5, 0.8, 1, 1.2],
                'learning_rate': [0.01, 0.05, 0.1, 1, 2],
                'max_depth': [1, 3, 5, 7],
                'min_child_samples': [5, 10, 20, 50, 80],
                'n_estimators': [30, 50, 100, 150, 200],
                'num_leaves': [5, 7, 15, 31, 60],
                'subsample': [0.1, 0.2, 0.5, 0.8]
            }
        ]

    if 'GradientBoostingClassifier' in base_model_names:
        param_grids['GradientBoostingClassifier'] = [
            {
                'learning_rate': [0.1, 1, 2],
                'max_depth': [3, 5, 7],
                'min_samples_leaf': [1, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [50, 100, 150]
            }
        ]

    if 'CatBoostClassifier' in base_model_names:
        param_grids['CatBoostClassifier'] = [
            {
                'border_count': [16, 32, 64, 128, 256],
                'l2_leaf_reg': [7, 9, 11, 13, 15],
                'learning_rate': [0.01, 0.05, 0.1, 0.5, 1, 2, 3],
                'max_depth': [1, 3, 5, 7],
                'n_estimators': [30, 50, 100, 150, 200]
            }
        ]

    if 'AdaBoostClassifier' in base_model_names:
        param_grids['AdaBoostClassifier'] = [
            {
                'learning_rate': [0.001, 0.01, 0.1, 1],
                'n_estimators': [10, 30, 50, 100]
            }
        ]

    if 'ExtraTreesClassifier' in base_model_names:
        param_grids['ExtraTreesClassifier'] = [
            {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, None],
                'max_features': [2, 3, 'sqrt'],
                'min_samples_leaf': [1, 5, 10],
                'min_samples_split': [1, 2, 5],
                'n_estimators': [50, 100, 150, 200, 250]
            }
        ]

    if 'MLPClassifier' in base_model_names:
        param_grids['MLPClassifier'] = [
            {
                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'batch_size': ['auto', 100, 200, 300],
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'max_iter': [2000, 5000],
                'solver': ['lbfgs']
            },
            {
                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'batch_size': ['auto', 100, 200, 300],
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'max_iter': [2000, 5000],
                'solver': ['sgd']
            },
            {
                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'batch_size': ['auto', 100, 200, 300],
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'max_iter': [2000, 5000],
                'solver': ['adam']
            }
        ]

    if 'RidgeClassifier' in base_model_names:
        param_grids['RidgeClassifier'] = [
            {
                'alpha': [1, 10, 100, 1000],
                # Use fit_intercept instead of normalize
                'fit_intercept': [True, False],
                'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }
        ]

    if 'Perceptron' in base_model_names:
        param_grids['Perceptron'] = [
            {
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'fit_intercept': [True, False],
                'max_iter': [50, 100, 250, 500, 1000],
                'penalty': [None, 'l2', 'l1', 'elasticnet']
            }
        ]

    if 'PassiveAggressiveClassifier' in base_model_names:
        param_grids['PassiveAggressiveClassifier'] = [
            {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'fit_intercept': [True, False],
                'max_iter': [50, 100, 250, 500, 1000]
            }
        ]

    if 'BaggingClassifier' in base_model_names:
        param_grids['BaggingClassifier'] = [
            {
                'max_features': [0.5, 0.8, 1],
                'max_samples': [0.5, 0.8, 1],
                'n_estimators': [10, 30, 50, 100]
            }
        ]

    return param_grids


# Defining models
def create_models(model_features_support: Dict[str, Dict[str, bool]], config: Dict[str, Any], class_ratio: float) -> (
        Dict)[str, Any]:
    """
    Creates a dictionary of selected machine learning models based on the provided configuration.

    Args:
        model_features_support: A dictionary that maps model names to their features support, including whether
            they support class weighting.
        config: Configuration dictionary containing the random state, information about whether the training data
            is balanced, and a list of base model names.
        class_ratio: A floating-point value representing the ratio of classes in the data.

    Raises:
        ValueError: If any of the base model names provided in the config does not exist in the available models.
        Warning: If any of the selected models do not support class weighting, and the classes in the training data
            are imbalanced.

    Returns:
        selected_models: A dictionary containing the selected models as per the configuration.
    """
    random_state = config['random_state']
    data_is_balanced = config['training']['data_is_balanced']
    base_model_names = config['training']['base_model_names']

    # Prepare class weight parameters
    if data_is_balanced:
        class_weight = None
        scale_pos_weight = None
        is_unbalance = False
        catboost_class_weights = None
    else:
        class_weight = 'balanced'
        scale_pos_weight = round(class_ratio)
        is_unbalance = True
        catboost_class_weights = [1, round(class_ratio)]

        # Warn if now all selected models support class weighting
        for model_name in base_model_names:
            if not model_features_support[model_name]['class_weighting']:
                warn.warn(f'Model {model_name} does not support class weighting, and the classes in the training data '
                          f'are imbalanced')

    all_models = {
        'GaussianNB': GaussianNB(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'LogisticRegression': LogisticRegression(random_state=random_state, class_weight=class_weight),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
        'SVC': SVC(probability=True, random_state=random_state, class_weight=class_weight),
        'RandomForestClassifier': RandomForestClassifier(random_state=random_state, class_weight=class_weight),
        'XGBClassifier': xgb.XGBClassifier(random_state=random_state, eval_metric='logloss',
                                           scale_pos_weight=scale_pos_weight),
        'LGBMClassifier': lgb.LGBMClassifier(objective='binary', random_state=random_state, class_weight=class_weight,
                                             is_unbalance=is_unbalance),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=random_state),
        'CatBoostClassifier': CatBoostClassifier(random_state=random_state, class_weights=catboost_class_weights,
                                                 verbose=False),
        'AdaBoostClassifier': AdaBoostClassifier(random_state=random_state),
        'ExtraTreesClassifier': ExtraTreesClassifier(random_state=random_state, class_weight=class_weight),
        'MLPClassifier': MLPClassifier(random_state=random_state),
        'RidgeClassifier': RidgeClassifier(random_state=random_state, class_weight=class_weight),
        'Perceptron': Perceptron(random_state=random_state, class_weight=class_weight),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(random_state=random_state,
                                                                   class_weight=class_weight),
        'BaggingClassifier': BaggingClassifier(
            estimator=DecisionTreeClassifier(class_weight=class_weight), random_state=random_state)
    }

    # Check if all base_model_names exist in all_models
    for model_name in base_model_names:
        if model_name not in all_models:
            raise ValueError(
                f"The model '{model_name}' does not exist in the available models.")

    selected_models = {name: model for name,
                       model in all_models.items() if name in base_model_names}

    return selected_models


# Define scoring metrics
def define_scoring_metrics(config: Dict[str, Any], model_name: str, model_features_support: Dict[str, Dict[str, bool]])\
        -> Dict[str, Union[str, Callable]]:
    """
    Defines the scoring metrics for a given model based on the provided configuration.

    Args:
        config: Configuration dictionary containing the training settings, including the scoring metrics.
        model_name: A string representing the name of the model.
        model_features_support: A dictionary that maps model names to their features support, including whether
            they support `predict_proba` method.

    Raises:
        ValueError: If an unsupported metric is provided in the configuration.
        Warning: If the model does not support `predict_proba` method and 'roc_auc' or 'gini' metric is requested.

    Returns:
        scoring: A dictionary mapping the names of the scoring metrics to either the metric names or the custom
            scoring functions.
    """
    # Define gini function
    def gini(y_true, y_prob):
        auc = roc_auc_score(y_true, y_prob)
        return 2 * auc - 1

    # Create empty dictionary to store metrics
    scoring = {}

    # Get the list of metrics from the config
    metrics = config.get('training', {}).get('scoring', [])

    # Supported metrics
    supported_metrics = ['accuracy', 'precision',
                         'recall', 'f1', 'roc_auc', 'gini']

    # For each metric in the list
    for metric in metrics:
        if metric not in supported_metrics:
            raise ValueError(f'Unsupported metric: {metric}. Please use one of the following:'
                             f' {", ".join(supported_metrics)}.')

        if metric in ['accuracy', 'precision', 'recall', 'f1']:
            scoring[metric] = metric
        elif metric == 'roc_auc':
            if model_features_support[model_name]['predict_proba']:
                scoring['roc_auc'] = 'roc_auc'
            else:
                warnings.warn(f'The model {model_name} does not support predict_proba. '
                              f'The roc_auc metric will not be used.')
        elif metric == 'gini':
            if model_features_support[model_name]['predict_proba']:
                gini_scorer = make_scorer(gini, needs_proba=True)
                scoring['gini'] = gini_scorer
            else:
                warnings.warn(f'The model {model_name} does not support predict_proba. '
                              f'The gini metric will not be used.')

    return scoring


# Searching for optimal hyperparameters of base models, training, calibration, and evaluation
def train_and_evaluate_base_models(config: Dict[str, Any], base_models: Dict[str, Any], param_grids: Dict[str, Any],
                                   x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series,
                                   feature_names: List[str], path: str = '') -> pd.DataFrame:
    """
    Searches for optimal hyperparameters, trains, calibrates, and evaluates base models according to the provided
     configuration.

    Args:
        config: Configuration dictionary containing various settings including tag, random_state,
         cross_validation_search_type, num_cv_folds, and num_iter.
        base_models: Dictionary containing base models.
        param_grids: Dictionary containing parameter grids for hyperparameter tuning.
        x_train: Training feature DataFrame.
        y_train: Training target Series.
        x_test: Testing feature DataFrame.
        y_test: Testing target Series.
        feature_names: List of feature names.
        path: Optional; Path to store saved models and parameters.

    Raises:
        ValueError: If an invalid search_type is provided in the configuration.

    Returns:
        performance_metrics: DataFrame containing the performance metrics for each model on the training and test sets,
         including accuracy, precision, recall, F1 score, AUC, Gini, and overfitting probabilities.
    """
    tag = config['tag']
    random_state = config['random_state']
    search_type = config['training']['cross_validation_search_type']
    num_cv_folds = config['training']['num_cv_folds']
    num_iter = config['training']['num_iter']

    # Creating a dataframe to store metrics
    performance_metrics = pd.DataFrame(
        columns=['Model', 'Accuracy (Train)', 'Precision (Train)', 'Recall (Train)', 'F1 Score (Train)', 'AUC (Train)',
                 'Gini (Train)', 'Accuracy (Test)', 'Precision (Test)', 'Recall (Test)', 'F1 Score (Test)',
                 'AUC (Test)', 'Gini (Test)', 'Overfitting Probability by Accuracy', 'Overfitting Probability by F1'])

    # Creating a dictionary to store StratifiedKFold objects for each base model
    cv_dict = {model_name: StratifiedKFold(n_splits=num_cv_folds, shuffle=True, random_state=random_state + i)
               for i, model_name in enumerate(base_models.keys())}

    for model_name, model in base_models.items():
        # Displaying current stage on the screen
        print(
            f'Running {search_type.capitalize()}SearchCV for {model_name}...\n')

        # Defining scoring metrics
        scoring = define_scoring_metrics(
            config, model_name, MODEL_FEATURES_SUPPORT)

        first_metric = list(scoring.keys())[0]
        print(f"Using '{first_metric}' as a scoring metric to refit\n")

        # Finding the best hyperparameters
        if search_type == 'grid':
            grid_search = GridSearchCV(
                model, param_grids[model_name], cv=cv_dict[model_name], scoring=scoring, n_jobs=-1, verbose=1,
                refit=first_metric)

        elif search_type == 'randomized':
            grid_search = RandomizedSearchCV(
                model, param_grids[model_name], n_iter=num_iter, cv=cv_dict[model_name], scoring=scoring, n_jobs=-1,
                verbose=1, random_state=random_state, refit=first_metric)

        else:
            raise ValueError(f"Invalid search_type: '{search_type}'. "
                             f"Valid options are 'grid' and 'randomized'.")

        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_

        # Saving the best hyperparameters to a JSON file
        best_params_filename = os.path.join(
            path, f'{model_name}_best_params_{tag}.json')
        dp.save_object_to_file(grid_search.best_params_, best_params_filename)

        # Determine whether the model supports predict_proba
        supports_predict_proba = hasattr(best_model, 'predict_proba')

        # Calibration of the best model
        print(f'Calibrating {model_name}...\n')
        calibrated_model = CalibratedClassifierCV(
            best_model, method='isotonic', cv='prefit')
        calibrated_model.fit(x_train, y_train)

        # Always save the calibrated model
        model_filename = os.path.join(
            path, f'{model_name}_calibrated_model_{tag}.pkl')
        dp.save_object_to_file(calibrated_model, model_filename)

        # If the model supports predict_proba, also save the uncalibrated model
        if supports_predict_proba:
            model_filename = os.path.join(
                path, f'{model_name}_uncalibrated_model_{tag}.pkl')
            dp.save_object_to_file(best_model, model_filename)

        # Testing the calibrated model on the test set
        y_pred_test = calibrated_model.predict(x_test)
        y_pred_train = calibrated_model.predict(x_train)

        # Calculate and display performance metrics
        performance_metrics = calculate_and_display_metrics(config, model_name, y_train, y_pred_train, y_test,
                                                            y_pred_test, performance_metrics, calibrated_model,
                                                            x_train, x_test, path=path)

        # Plot feature importance
        plot_feature_importance_or_coefficients(config, model_name, calibrated_model, x_train, feature_names,
                                                MODEL_FEATURES_SUPPORT, path)

    return performance_metrics


# Load base models for stacking from files
def load_base_models(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Loads base models for stacking from files specified in the configuration.

    For models that support the 'predict_proba' method, it's generally better to use the uncalibrated models for
     stacking,
    as the calibrated probabilities might be less informative for the second-level model.
    For models that do not support 'predict_proba', calibrated models must be used to get probability estimates for
     stacking.

    Args:
        config: Configuration dictionary containing the paths to the base model files under 'training' and
         'base_model_files' keys.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the model does not support 'predict_proba' and is not calibrated.

    Returns:
        base_models_info: List of dictionaries containing information about the loaded base models, including the name
         (path) and the model object.
    """
    model_files = config['training']['base_model_files']
    base_models_info = []

    for model_file in model_files:
        # Make sure the model file exists
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"The model file '{model_file}' does not exist.")

        # Load the model from the file
        model = dp.load_data_from_file(model_file)

        # Check if the model supports predict_proba
        supports_predict_proba = hasattr(model, 'predict_proba')

        # Check the model type
        model_type = type(model).__name__

        if model_type == 'CalibratedClassifierCV':
            warnings.warn(f"The model from the file '{model_file}' supports 'predict_proba'. "
                          f"Please ensure an uncalibrated model is used.")

        elif model_type != 'CalibratedClassifierCV' and not supports_predict_proba:
            raise ValueError(f"The model from the file '{model_file}' does not support 'predict_proba'. "
                             f"Please use a calibrated model to add support for 'predict_proba'.")

        base_models_info.append({
            'name': model_file,
            'model': model
        })

    return base_models_info


# Searching for optimal hyperparameters of stacking model, training, calibration, and evaluation
def train_and_evaluate_stacking_model(config: Dict[str, Any],
                                      base_models_info: List[Dict[str, Any]],
                                      x_train: pd.DataFrame,
                                      y_train: pd.Series,
                                      x_test: pd.DataFrame,
                                      y_test: pd.Series,
                                      path: str = '') -> pd.DataFrame:
    """
    Searches for optimal hyperparameters of a stacking model, trains it, optionally calibrates it, and evaluates its
     performance.

    Args:
        config (dict): Configuration dictionary containing stacking model type, tag, and other hyperparameters.
        base_models_info (list): List of dictionaries containing information about the base models used for stacking.
        x_train (DataFrame): Features of the training set.
        y_train (Series): Target of the training set.
        x_test (DataFrame): Features of the test set.
        y_test (Series): Target of the test set.
        path (str, optional): Path where the stacking model and related information will be saved. Defaults to an empty
         string.

    Raises:
        ValueError: If the specified stacking model type is not one of the supported types ('logistic_regression',
         'gradient_boosting', or 'neural_network').

    Returns:
        pd.DataFrame: A dataframe containing performance metrics of the trained stacking model.

    Note:
        - For the 'neural_network' stacking model type, the architecture and weights of the neural network will be
         saved in separate files.
        - The saved model files are named based on the 'tag' provided in the configuration.
    """
    tag = config['tag']
    stacking_model_type = config['training']['stacking_model_type']

    # Creating a dataframe to store metrics
    performance_metrics = pd.DataFrame(
        columns=['Model', 'Accuracy (Train)', 'Precision (Train)', 'Recall (Train)', 'F1 Score (Train)', 'AUC (Train)',
                 'Gini (Train)', 'Accuracy (Test)', 'Precision (Test)', 'Recall (Test)', 'F1 Score (Test)',
                 'AUC (Test)', 'Gini (Test)', 'Overfitting Probability by Accuracy', 'Overfitting Probability by F1'])

    # Creating a list of base models
    base_models = [model_info['model'] for model_info in base_models_info]

    # Save used models info to file
    used_models_for_stacking = [model_info['name']
                                for model_info in base_models_info]
    used_models_for_stacking_filename = os.path.join(
        path, f'MetaClassifier_used_models_{tag}.json')
    dp.save_object_to_file(used_models_for_stacking,
                           used_models_for_stacking_filename)

    # Displaying current stage on the screen
    print('Creating Stacking Model using the best trained models from each base classifier...\n')

    # Creating a meta-classifier based on the given model
    if stacking_model_type == 'logistic_regression':
        meta_classifier = LogisticRegression()

    elif stacking_model_type == 'gradient_boosting':
        meta_classifier = LGBMClassifier()  # or use XGBoost

    elif stacking_model_type == 'neural_network':
        base_classifiers = [model_dict['model']
                            for model_dict in base_models_info]

        def create_nn(input_dim):
            model = Sequential()
            model.add(Dense(16, input_dim=input_dim, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics=['accuracy'])
            return model

        meta_classifier = KerasClassifier(
            model=create_nn,
            input_dim=len(base_classifiers),
            epochs=20,
            batch_size=32,
            verbose=0
        )

    else:
        raise ValueError(f"Invalid stacking model type: '{stacking_model_type}'. "
                         f"Valid options are 'logistic_regression', 'gradient_boosting', and 'neural_network'.")

    # Obtaining base classifiers' probabilities or predictions on x_train
    meta_features_train = [clf.predict_proba(
        x_train)[:, 1] for clf in base_models]

    # Combining base classifiers' probabilities or predictions
    x_meta_train = np.column_stack(meta_features_train)

    # Training meta-classifier on combined predictions and y_train
    meta_classifier.fit(x_meta_train, y_train)

    # Obtaining base classifiers' probabilities or predictions on x_test
    meta_features_test = [clf.predict_proba(
        x_test)[:, 1] for clf in base_models]

    # Combining base classifiers' probabilities or predictions
    x_meta_test = np.column_stack(meta_features_test)

    # Testing the stacking model on the test data set
    y_pred_stacked_test = meta_classifier.predict(x_meta_test)
    y_pred_stacked_train = meta_classifier.predict(x_meta_train)

    # Saving the meta_classifier
    if stacking_model_type in ['logistic_regression', 'gradient_boosting']:
        model_filename = os.path.join(path, f'MetaClassifier_{tag}.pkl')
        dp.save_object_to_file(meta_classifier, model_filename)

    elif stacking_model_type == 'neural_network':
        # Saving the architecture of the neural network meta_classifier
        model_json = meta_classifier.model_.to_json()
        model_dict = json.loads(model_json)
        architecture_filename = os.path.join(
            path, f'MetaClassifier_architecture_{tag}.json')
        dp.save_object_to_file(model_dict, architecture_filename)

        # Saving the weights of the neural network meta_classifier
        weights_filename = os.path.join(
            path, f'MetaClassifier_weights_{tag}.h5')
        dp.save_object_to_file(meta_classifier.model_, weights_filename)

    else:
        raise ValueError(f"Invalid stacking model type: '{stacking_model_type}'. "
                         f"Valid options are 'logistic_regression', 'gradient_boosting', and 'neural_network'.")

    # Calculating and displaying performance metrics
    performance_metrics = calculate_and_display_metrics(
        config,
        model_name='MetaClassifier',
        y_true_train=y_train,
        y_pred_train=y_pred_stacked_train,
        y_true_test=y_test,
        y_pred_test=y_pred_stacked_test,
        metrics_df=performance_metrics,
        clf=meta_classifier,
        x_train=x_meta_train,
        x_test=x_meta_test,
        path=path)

    return performance_metrics


# Calculate and display metrics
def calculate_and_display_metrics(config: Dict[str, Any],
                                  model_name: str,
                                  y_true_train: Union[pd.Series, np.ndarray],
                                  y_pred_train: Union[pd.Series, np.ndarray],
                                  y_true_test: Union[pd.Series, np.ndarray],
                                  y_pred_test: Union[pd.Series, np.ndarray],
                                  metrics_df: pd.DataFrame,
                                  clf: Any,
                                  x_train: Union[pd.DataFrame, np.ndarray],
                                  x_test: Union[pd.DataFrame, np.ndarray],
                                  decimals: int = 3,
                                  plot_roc: bool = True,
                                  path: str = '') -> pd.DataFrame:
    """
    Calculates various performance metrics for the given model and displays the results.
    Optionally plots the ROC curve and saves it to the file.

    Args:
        config (dict): Configuration dictionary containing tags and other settings.
        model_name (str): The name of the model being evaluated.
        y_true_train (Series/ndarray): Actual target values for the training set.
        y_pred_train (Series/ndarray): Predicted target values for the training set.
        y_true_test (Series/ndarray): Actual target values for the test set.
        y_pred_test (Series/ndarray): Predicted target values for the test set.
        metrics_df (DataFrame): DataFrame to store the metrics.
        clf (Any): Trained classifier object.
        x_train (DataFrame/ndarray): Features of the training set.
        x_test (DataFrame/ndarray): Features of the test set.
        decimals (int, optional): Number of decimal places to round the metrics. Defaults to 3.
        plot_roc (bool, optional): Whether to plot the ROC curve. Defaults to True.
        path (str, optional): Path where the ROC curve plot will be saved. Defaults to an empty string.

    Returns:
        pd.DataFrame: Updated dataframe containing the performance metrics for the given model.

    Notes:
        - The ROC curve will be saved as a PNG file named based on the model name and tag provided in the configuration.
        - The metrics include accuracy, precision, recall, F1 score, AUC, Gini, and overfitting probability based on
         accuracy and F1.
    """
    tag = config['tag']

    train_accuracy = accuracy_score(y_true_train, y_pred_train)
    train_accuracy_rounded = round(train_accuracy, decimals)
    test_accuracy = accuracy_score(y_true_test, y_pred_test)
    test_accuracy_rounded = round(test_accuracy, decimals)
    train_precision_rounded = round(
        precision_score(y_true_train, y_pred_train), decimals)
    test_precision_rounded = round(
        precision_score(y_true_test, y_pred_test), decimals)
    train_recall_rounded = round(recall_score(
        y_true_train, y_pred_train), decimals)
    test_recall_rounded = round(recall_score(
        y_true_test, y_pred_test), decimals)
    train_f1 = f1_score(y_true_train, y_pred_train)
    train_f1_rounded = round(train_f1, decimals)
    test_f1 = f1_score(y_true_test, y_pred_test)
    test_f1_rounded = round(test_f1, decimals)

    # Compute and save the overfitting probability by accuracy and f1
    train_test_accuracy_diff = train_accuracy - test_accuracy
    overfitting_probability_by_accuracy_rounded = round(
        min(max(train_test_accuracy_diff, 0), 1), decimals)

    train_test_f1_diff = train_f1 - test_f1
    overfitting_probability_by_f1_rounded = round(
        min(max(train_test_f1_diff, 0), 1), decimals)

    # Compute probabilities of the positive class
    y_proba_train = clf.predict_proba(x_train)[:, 1]
    y_proba_test = clf.predict_proba(x_test)[:, 1]

    # Compute ROC curve, AUC, and Gini
    fpr_train, tpr_train, _ = roc_curve(y_true_train, y_proba_train)
    fpr_test, tpr_test, _ = roc_curve(y_true_test, y_proba_test)
    auc_train = roc_auc_score(y_true_train, y_proba_train)
    auc_train_rounded = round(auc_train, decimals)
    auc_test = roc_auc_score(y_true_test, y_proba_test)
    auc_test_rounded = round(auc_test, decimals)
    gini_train_rounded = round(2*auc_train - 1, decimals)
    gini_test_rounded = round(2*auc_test - 1, decimals)

    print('Performance Metrics:\n')
    print(f'Model: {model_name}')
    print(f'Train Accuracy: {train_accuracy_rounded}')
    print(f'Test Accuracy: {test_accuracy_rounded}')
    print(f'Train Precision: {train_precision_rounded}')
    print(f'Test Precision: {test_precision_rounded}')
    print(f'Train Recall: {train_recall_rounded}')
    print(f'Test Recall: {test_recall_rounded}')
    print(f'Train F1 Score: {train_f1_rounded}')
    print(f'Test F1 Score: {test_f1_rounded}')
    print(f'Train AUC: {auc_train_rounded}')
    print(f'Test AUC: {auc_test_rounded}')
    print(f'Train Gini: {gini_train_rounded}')
    print(f'Test Gini: {gini_test_rounded}')
    print()

    new_row = pd.DataFrame({
        'Model': [model_name],
        'Accuracy (Train)': [train_accuracy_rounded],
        'Accuracy (Test)': [test_accuracy_rounded],
        'Precision (Train)': [train_precision_rounded],
        'Precision (Test)': [test_precision_rounded],
        'Recall (Train)': [train_recall_rounded],
        'Recall (Test)': [test_recall_rounded],
        'F1 Score (Train)': [train_f1_rounded],
        'F1 Score (Test)': [test_f1_rounded],
        'AUC (Train)': [auc_train_rounded],
        'AUC (Test)': [auc_test_rounded],
        'Gini (Train)': [gini_train_rounded],
        'Gini (Test)': [gini_test_rounded],
        'Overfitting Probability by Accuracy': [overfitting_probability_by_accuracy_rounded],
        'Overfitting Probability by F1': [overfitting_probability_by_f1_rounded]
    })

    # Plot ROC curve if requested
    if plot_roc:
        matplotlib.use('Agg')

        plt.figure()
        plt.plot(fpr_train, tpr_train, color='blue', lw=2,
                 label=f'Train ROC curve (AUC = {auc_train_rounded})')
        plt.plot(fpr_test, tpr_test, color='red', lw=2,
                 label=f'Test ROC curve (AUC = {auc_test_rounded})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc='lower right')

        # Saving ROC curve plot to file
        roc_file_path = os.path.join(path, f'{model_name}_roc_curve_{tag}.png')
        plt.savefig(roc_file_path, dpi=300, bbox_inches='tight')

        # Closing ROC curve plot
        plt.close()

    return pd.concat([metrics_df, new_row], ignore_index=True)


# Plot feature importance
def plot_feature_importance_or_coefficients(config: Dict[str, Any],
                                            model_name: str,
                                            model_to_use: Any,
                                            x_train: Union[pd.DataFrame, np.ndarray],
                                            feature_names: List[str],
                                            model_features_support: Dict[str, Dict[str, bool]],
                                            path: str = '') -> None:
    """
    Plots the feature importance or coefficients for the given model and saves the plot to a file.
    If the model does not support feature importance, attempts to use LIME for an explanation.

    Args:
        config (dict): Configuration dictionary containing tags and other settings.
        model_name (str): The name of the model being evaluated.
        model_to_use (Any): Trained model object.
        x_train (DataFrame/ndarray): Features of the training set.
        feature_names (list): List of feature names.
        model_features_support (dict): Dictionary indicating whether models support feature importance plot.
        path (str, optional): Path where the plot will be saved. Defaults to an empty string.

    Notes:
        - The feature importance or coefficients plot will be saved as a PNG file named based on the model name and tag
         provided in the configuration.
        - If the model does not support feature importance or coefficients, LIME (Local Interpretable Model-agnostic
         Explanations) is used to create an explanation, if possible.
        - Supported models for coefficient extraction include Logistic Regression.
        - Supported models for feature importance include Random Forest, Gradient Boosting, AdaBoost, Extra Trees,
         XGBoost, LightGBM, CatBoost.
        - The function prints a message if it fails to plot the feature importance or LIME explanation.
    """
    tag = config['tag']

    # Checking if model supports feature importance plot
    if model_features_support[model_name].get('feature_importance', False):
        # Checking model type and extracting feature importance
        model_type = type(model_to_use).__name__
        coef_models = ['LogisticRegression']
        importance_models = ['RandomForestClassifier', 'GradientBoostingClassifier', 'AdaBoostClassifier',
                             'ExtraTreesClassifier', 'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier']
        plot_title = 'Feature Importance'

        if model_type in coef_models:
            importance = model_to_use.coef_[0]
        elif model_type in importance_models:
            importance = model_to_use.feature_importances_
        else:
            print(
                f'The model {model_name} does not support feature importance in this function.')
            return

        # Plotting feature importance
        indices = np.argsort(importance)
        plt.figure(figsize=(10, 6))
        plt.title(plot_title)
        plt.barh(range(len(importance)),
                 importance[indices], color='r', align='center')
        plt.yticks(range(len(importance)), [feature_names[i] for i in indices])
        plt.ylim([-1, len(importance)])

        # Saving feature importance plot to file
        importance_file_path = os.path.join(
            path, f'{model_name}_feature_importance_{tag}.png')
        plt.savefig(importance_file_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(
            f'The model {model_name} does not support feature importance plot. Trying to use LIME...')
        try:
            explainer = lime_tabular.LimeTabularExplainer(x_train, feature_names=feature_names, class_names=['0', '1'],
                                                          verbose=True, mode='classification')

            # Here you need to decide what instance you want to explain. we'll use the first one (x_train[0]).
            exp = explainer.explain_instance(
                x_train[0], model_to_use.predict_proba, num_features=len(feature_names))

            # Plotting LIME explanation
            exp.as_pyplot_figure()
            lime_file_path = os.path.join(
                path, f'{model_name}_lime_explanation_{tag}.png')
            plt.savefig(lime_file_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(
                f'Failed to generate LIME explanation for {model_name}. Error: {str(e)}')


# Analyse overfitting probability
def analyze_overfitting_probability(config: Dict[str, Any], performance_metrics: pd.DataFrame) -> None:
    """
    Analyzes the overfitting probability of trained models by comparing performance metrics
    between the training and testing sets. Prints out information about models that may
    be overfitting based on the configured overfitting threshold.

    Args:
        config (dict): Configuration dictionary containing training settings such as the overfitting threshold.
        performance_metrics (DataFrame): A DataFrame containing performance metrics like accuracy and F1 score for both
         train and test datasets.

    Prints:
        - Information about models that may be overfitting, including differences in accuracy and F1 score between
         training and testing sets.
        - A message indicating whether any models show significant signs of overfitting, or that no models appear to be
         overfitting.

    Notes:
        - The overfitting threshold is extracted from the config dictionary. Models whose differences in accuracy or F1
         score between training and testing sets exceed this threshold are considered to be potentially overfitting.
        - The function iterates through all rows in the performance_metrics DataFrame, analyzing each model.
        - A list of potentially overfitting models is printed at the end if any are found.
    """
    overfitting_threshold = config['training']['overfitting_threshold']

    overfitting_models = []

    for index, row in performance_metrics.iterrows():
        model_name = row['Model']
        train_accuracy = row['Accuracy (Train)']
        test_accuracy = row['Accuracy (Test)']
        train_f1 = row['F1 Score (Train)']
        test_f1 = row['F1 Score (Test)']

        overfit = False

        if train_accuracy - test_accuracy > overfitting_threshold:
            print('-> Accuracy difference between train and test is significant.')
            overfit = True

        if train_f1 - test_f1 > overfitting_threshold:
            print('-> F1 Score difference between train and test is significant.')
            overfit = True

        if overfit:
            overfitting_models.append(model_name)
            print(f'-> {model_name} may be overfitting.')
        else:
            print(f'-> {model_name} seems to generalize well.')

        print('\n')

    if overfitting_models:
        print('Models that may be overfitting:')
        print(', '.join(overfitting_models))
    else:
        print('No models show significant signs of overfitting.')


# Train and evaluate models
def train_and_evaluate_models(train_file_path: str, test_file_path: str, config_file_path: str, delimiter: str,
                              model_features_support: Dict[str, Any], path: str = '') -> None:
    """
    Train and evaluate machine learning models based on the provided configuration file. This function loads training
     and testing datasets,
    splits them into features and labels, defines hyperparameter search spaces, trains models, and evaluates them.
    The approach for training and evaluation can be either 'individual' for base models or 'stacking' for ensemble
     learning.

    Args:
        train_file_path (str): Path to the training dataset file (CSV format).
        test_file_path (str): Path to the testing dataset file (CSV format).
        config_file_path (str): Path to the configuration file (JSON format), containing model settings and
         hyperparameters.
        delimiter (str): Delimiter used in the CSV files.
        model_features_support (dict): Dictionary containing information about feature support for different models.
        path (str, optional): Directory path to save results, such as performance metrics and plots. Default is an
         empty string, meaning the current directory.

    Processes:
        - Load configuration, training, and testing data.
        - Split data into features and labels.
        - Depending on the approach type (individual or stacking), either train and evaluate base models or a stacking
         model.
        - Analyze overfitting probability.
        - Save performance metrics to a CSV file.

    Raises:
        ValueError: If the provided 'approach_type' in the config file is unknown.

    Notes:
        - The 'config' file should include information like the target column name, approach type
         (individual or stacking), and other training parameters.
        - The 'model_features_support' argument must contain information about which models support certain features,
         used in defining models.
    """
    # Loading config from JSON file
    config = dp.load_data_from_file(config_file_path)

    # Getting parameters
    target_column = config['target_column']
    approach_type = config['training']['approach_type']

    # Loading the training and testing sets directly from the provided file paths, using the specified delimiter
    train_data = pd.read_csv(train_file_path, delimiter=delimiter)
    test_data = pd.read_csv(test_file_path, delimiter=delimiter)

    # Splitting data into features and labels
    x_train, y_train, x_test, y_test = dp.split_data_into_feats_and_labels(
        train_data, test_data, config)

    if approach_type == 'individual':
        # Defining the hyperparameter search space
        param_grids = create_param_grids(x_train, config)

        # Defining models
        class_ratio = dp.calculate_class_ratio(
            train_data, config)
        base_models = create_models(
            model_features_support, config, class_ratio)

        # Getting feature names
        feature_names = [
            col for col in train_data.columns if col != target_column]

        # Searching for optimal hyperparameters of base models, training, calibration, and evaluation
        performance_metrics = train_and_evaluate_base_models(config, base_models, param_grids, x_train, y_train,
                                                             x_test, y_test, feature_names, path)

    elif approach_type == 'stacking':
        # Loading base models
        base_models = load_base_models(config)

        # Searching for optimal hyperparameters of stacking model, training, calibration, and evaluation
        performance_metrics = train_and_evaluate_stacking_model(config, base_models, x_train, y_train,
                                                                x_test, y_test, path)

    else:
        raise ValueError(f'Unknown approach_type: {approach_type}')

    # Analyzing performance metrics
    analyze_overfitting_probability(config, performance_metrics)

    # Saving performance metrics in a CSV file
    performance_metrics_file_path = os.path.join(
        path, f"performance_metrics_{config['tag']}.csv")
    dp.save_object_to_file(performance_metrics, performance_metrics_file_path)


if __name__ == '__main__':
    # Parsing command-line arguments
    parser = argparse.ArgumentParser(description='Create models.')
    parser.add_argument('train_file_path',
                        help='Path to the CSV file containing the training data')
    parser.add_argument('test_file_path',
                        help='Path to the CSV file containing the test data')
    parser.add_argument('config_file_path',
                        help='Path to the JSON configuration file')
    parser.add_argument('-d', '--delimiter', default=',',
                        help="Delimiter used in the CSV file (default: ',')")

    args = parser.parse_args()

    # Getting the directory of the train data file
    train_file_dir = os.path.dirname(args.train_file_path)

    train_and_evaluate_models(args.train_file_path, args.test_file_path, args.config_file_path, args.delimiter,
                              MODEL_FEATURES_SUPPORT, path=train_file_dir)
