from os import walk

from src import main_routine

path_to_metadata = "./dataset_metadata"
path_to_datasets = "./datasets"

metadata_filenames = next(walk(path_to_metadata), (None, None, []))[2]  # [] if no file
if ".DS_Store" in metadata_filenames:
    metadata_filenames.remove(".DS_Store")
if ".gitkeep" in metadata_filenames:
    metadata_filenames.remove(".gitkeep")


def main():
    problems_to_solve = []

    for file in metadata_filenames:
        problems_to_solve.append(file.replace(".json", ""))

    encoders_to_use = [
        "OrdinalEncoder",
        #'OneHotEncoder',
        # "CesamoEncoder",
        # "CountEncoder",
        # "BinaryEncoder",
        # "HashingEncoder",
    ]
    models_to_use = [
        "XGBClassifier",
        #'MLPClassifier',
        #'GaussianNB',
        #'LogisticRegression',
        #'SVC',
    ]

    main_routine.start_benchmark(problems_to_solve, encoders_to_use, models_to_use)


if __name__ == "__main__":
    main()
