# set log
import logging
from nis import cat

from config.logging import set_logger

log = set_logger()
import json
import os
from datetime import datetime
from time import gmtime, strftime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_json(path_and_name):
    """
    Load JSON file from the specified path and return the data.

    Args:
        path_and_name (str): Path and name of the JSON file.

    Returns:
        dict: Data loaded from the JSON file.
    """
    with open(path_and_name, "r") as read_file:
        return json.load(read_file)


def load_csv(path_and_name):
    """
    Load CSV file from the specified path and return as DataFrame.

    Args:
        path_and_name (str): Path and name of the CSV file.

    Returns:
        DataFrame: DataFrame loaded from the CSV file.
    """
    return pd.read_csv(path_and_name)


def clean_data(df, dataset_config):
    """
    Clean the DataFrame based on the specified configuration.

    Args:
        df (DataFrame): Input DataFrame.
        dataset_config (dict): Configuration for cleaning.

    Returns:
        DataFrame: Cleaned DataFrame.
    """
    # remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    # set target
    df["target"] = (
        df[dataset_config["target"]]
        == dataset_config["positive_values_are_represented_by"]
    )
    # if target column is already called target, cannot be deleted
    delete_original_target_col = []
    if dataset_config["target"] != "target":
        delete_original_target_col = [dataset_config["target"]]
    df.drop(
        delete_original_target_col
        + dataset_config["id_cols"]
        + dataset_config["cols_to_delete"],
        axis=1,
        inplace=True,
    )

    # remove empty strings or only spaces
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # scale numeric columns
    apply_scaler = True
    if "is_scaled" in dataset_config:
        logging.info("independent variables are already scaled")
        if dataset_config["is_scaled"]:
            apply_scaler = False

    if apply_scaler:
        mms = MinMaxScaler()

        if dataset_config["num_cols"]:
            df[dataset_config["num_cols"]] = mms.fit_transform(
                df[dataset_config["num_cols"]]
            )

    # remove NaN values
    df.dropna(axis=0, inplace=True)
    # convert categorical columns to Object (strings)
    for cat_col in dataset_config["cat_cols"]:
        df[cat_col] = df[cat_col].astype(str)

    return df


def split_features_and_target(data):
    """
    Split the DataFrame into features and target.

    Args:
        data (DataFrame): Input DataFrame.

    Returns:
        DataFrame: Features.
        array-like: Target.
    """
    data_features = data.drop(["target"], axis=1)
    data_target = data["target"].to_numpy()
    return data_features, data_target


def save_results(
    dataset_name,
    model_name,
    encoder_name,
    f1_score,
    elapsed_time,
    path_to_dir,
    overwrite=True,
    accuracy=None,
    recall=None,
):
    """
    Save results to a CSV file.

    Args:
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model.
        encoder_name (str): Name of the encoder.
        f1_score (float): F1 score.
        elapsed_time (float): Elapsed time.
        path_to_dir (str): Path to the directory.
        overwrite (bool, optional): Whether to overwrite existing file. Defaults to True.
        accuracy (float, optional): Accuracy score. Defaults to None.
        recall (float, optional): Recall score. Defaults to None.
    """
    path = os.path.join(path_to_dir, dataset_name)
    # Check whether the specified path exists or not
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
    formated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = {
        "dataset_name": [dataset_name],
        "model_name": [model_name],
        "encoder_name": [encoder_name],
        "f1_score": [f1_score],
        "accuracy": [accuracy],
        "recall": [recall],
        "exec_date": [formated_date],
    }

    # current_date=strftime("%Y%m%d%H%M%S", gmtime())
    current_date = datetime.now().strftime("%Y%m%d%H%M%S")
    df = pd.DataFrame(result)
    # save dataframe
    result_file_name = None
    if overwrite:
        result_file_name = f"{model_name}_{encoder_name}.csv"
    else:
        result_file_name = f"{current_date}_{model_name}_{encoder_name}.csv"
    final_path_and_name = os.path.join(path, result_file_name)
    df.to_csv(final_path_and_name, index=False)
    log.info(f"Resuls saved in {final_path_and_name}")
