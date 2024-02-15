
# Categorical-encoders-benchmark 

## Overview

This project aims to solve classification problems using various machine learning models and categorical encoders. The main routine of the project loads datasets, applies preprocessing steps, trains models, and evaluates their performance.

## Files

### 1. `data_manager.py`

This file contains functions for loading and cleaning data.

### 2. `main_routine.py`

The main routine of the project resides in this file. It loads data, applies preprocessing steps, trains machine learning models, and evaluates their performance. Results are saved in the `results/performance` directory.

### 3. `model_manager.py`

Functions for setting up and training machine learning models are defined in this file.

### 4. `general.py`

This file contains general configuration settings for the project.

### 5. `dataset_metadata/metadata_example.json`

Metadata for an example dataset used in the project. It includes information about the dataset, such as column names, categorical columns, numerical columns, and the target variable. For any dataset that you want to evaluate, it is required to include it's metadata.json file.

### 6. `datasets/dataset_example.csv`

An example dataset used in the project.

## Results

The results of the project are stored in the following format:

- **Directory:** `results/performance/{dataset_name}/{model_name}_{encoder_name}.csv`

- **Content:**
  - `dataset_name`: Name of the dataset.
  - `model_name`: Name of the machine learning model used.
  - `encoder_name`: Name of the categorical encoder used.
  - `f1_score`: F1 score achieved by the model.
  - `accuracy`: Accuracy score achieved by the model.
  - `recall`: Recall score achieved by the model.
  - `exec_date`: Date and time when the results were recorded.

## Usage

To run the project, follow these steps:

1. Install the required dependencies specified in `requirements.txt`.
2. Add the datasets to evaluate and its metadata.
3. Execute:
```
python entrypoint.py
```
4. Consult results in `results/performance` directory