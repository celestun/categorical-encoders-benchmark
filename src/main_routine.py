from config.logging import set_logger

logger = set_logger()

import os

from config import general as general_config
from src import cat_encoders, data_manager, model_manager


def start_benchmark(problems_to_solve, encoders_to_use, models_to_use):
    logger.info(f"datasets to solve: {problems_to_solve}")
    logger.info(f"encoders to use: {encoders_to_use}")
    logger.info(f"models to use: {models_to_use}")

    for dataset_name in problems_to_solve:
        os.environ["DATASET_NAME"] = str(dataset_name)

        # Load dataset config
        metadata_dataset_path = f"./dataset_metadata/{dataset_name}.json"
        metadata_dataset = data_manager.load_json(metadata_dataset_path)
        cat_cols = metadata_dataset["cat_cols"]
        # load dataset in a dataframe
        raw_df = data_manager.load_csv(metadata_dataset["relative_path_to_dataset"])

        # Clean data
        df = data_manager.clean_data(raw_df, metadata_dataset)

        # Split DataFrame into features and target.
        features_data, target = data_manager.split_features_and_target(df)

        # Estimate the number of columns to add for the case of encoders that adds extra columns ,
        # like one-hot encoder
        cols_to_add_lc = sum([df[x].nunique() for x in cat_cols])
        logger.info(f"estimated added cols for OHE-based methods: {cols_to_add_lc}")
        counts = df["target"].value_counts().tolist()
        logger.info(
            f"class distribution, positive %: {(counts[1]/(counts[0]+counts[1]))*100}"
        )
        logger.info(
            f"class distribution, negative % {(counts[0]/(counts[0]+counts[1]))*100}"
        )

        # Initialize encoders array.
        encoders = cat_encoders.set(cat_cols, encoders_to_use)

        # Initialize models
        models = model_manager.set(models_to_use)

        # Iterate over models to solve the classification problem.
        for model in models:
            # Apply encoder to the features data.
            for encoder in encoders:
                logger.info(
                    f'Solving classification problem: {dataset_name}, with model: {model["model_name"]} and encoder: {encoder["encoder_name"]}'
                )
                (
                    f1_score,
                    elapsed_time,
                    accuracy,
                    recall,
                ) = model_manager.solve_class_problem(
                    features_data, target, 3, model, encoder
                )
                logger.info(f"f1_score achieved: {f1_score}")
                data_manager.save_results(
                    dataset_name,
                    model["model_name"],
                    encoder["encoder_name"],
                    f1_score,
                    elapsed_time,
                    general_config.path_to_performance_results,
                    accuracy=accuracy,
                    recall=recall,
                )
