from config.logging import set_logger

logger = set_logger()

import time
from statistics import mean

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def set(models_list):
    all_models_available = [
        {"model_name": "XGBClassifier", "model_obj": XGBClassifier()},
        {
            "model_name": "MLPClassifier",
            "model_obj": MLPClassifier(
                solver="adam",
                activation="relu",
                hidden_layer_sizes=(25, 25, 25),
                random_state=1,
            ),
        },
        {"model_name": "GaussianNB", "model_obj": GaussianNB()},
        {"model_name": "LogisticRegression", "model_obj": LogisticRegression()},
        {"model_name": "SVC", "model_obj": SVC()},
        {"model_name": "RandomForestClassifier", "model_obj": RandomForestClassifier()},
    ]

    final_models = []
    for model in all_models_available:
        if model["model_name"] in models_list:
            final_models.append(model)

    return final_models


def solve_class_problem(data_features, data_target, cv, model, encoder_data):
    skfolds = StratifiedKFold(n_splits=cv, random_state=42, shuffle=True)
    fold_counter = 1
    f1_score_results = []
    accuracy_score_result = []
    recall_score_result = []
    elapsed_times = []
    for train_index, test_index in skfolds.split(data_features, data_target):
        logger.info(f"Applying fold number {fold_counter} out of {cv}")
        X_train_fold = data_features.iloc[train_index]
        y_train_fold = data_target[train_index]
        X_test_fold = data_features.iloc[test_index]
        y_test_fold = data_target[test_index]
        start_timer = time.time()

        # apply categorical encoder
        encoder_obj = encoder_data["encoder_obj"]
        logger.info(f"Fitting the encoder")
        encoder_obj = encoder_obj.fit(X_train_fold, y_train_fold)
        logger.info(f"Transforming data for X_train_fold by the encoder")
        X_train_fold = encoder_obj.transform(X_train_fold, y_train_fold)
        X_train_fold = X_train_fold.fillna(X_train_fold.mean())

        # train model
        model_obj = model["model_obj"]
        logger.info(f"Fitting the model")
        model_obj.fit(X_train_fold, y_train_fold)

        # apply encoder transform to X Test
        logger.info("Transforming data for X_test_fold by the encoder")
        X_test_fold = encoder_obj.transform(X_test_fold, y_test_fold)
        X_test_fold = X_test_fold.fillna(X_test_fold.mean())

        y_pred = model_obj.predict(X_test_fold)
        end_timer = time.time()
        elapsed_time = end_timer - start_timer
        accuracy_score_result.append(accuracy_score(y_pred, y_test_fold))
        f1_score_results.append(f1_score(y_pred, y_test_fold))
        recall_score_result.append(recall_score(y_pred, y_test_fold))
        elapsed_times.append(elapsed_time)
        # increment counter
        fold_counter += 1

    f1_score_results_mean = mean(f1_score_results)
    avg_elapsed_time = mean(elapsed_times)
    accuracy_score_result_mean = mean(accuracy_score_result)
    recall_score_result_mean = mean(recall_score_result)
    return (
        f1_score_results_mean,
        avg_elapsed_time,
        accuracy_score_result_mean,
        recall_score_result_mean,
    )


def encode_datasets(data_features, data_target, encoder_data):
    logger.info(f"encoding dataset")

    # apply categorical encoder
    encoder_obj = encoder_data["encoder_obj"]
    logger.info(f"Fitting the encoder")
    encoder_obj = encoder_obj.fit(data_features, data_target)
    logger.info(f"Transforming data for X_train_fold by the encoder")
    data_features_ecoded = encoder_obj.transform(data_features, data_target)
    return data_features_ecoded
