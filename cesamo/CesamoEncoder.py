# set log
from nis import cat

from config.logging import set_logger

logger = set_logger()
import json
import math
import os

# CESAMO ORIGINAL
import random
from random import randint

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


class CesamoEncoder:
    def __init__(self, cols, run_with_cache=True):
        self.encoded_metadata = {}
        self.cols = cols
        self.stop_on_accepted_error = True
        self.min_error = 0.05
        self.stop_on_iter_limit = True
        self.iter_limit = 10000
        self.handle_unknown = "value"

        self.n_el_error_list = 30
        self.least_el_in_avg_error_list = 3
        self.code_digits = 5
        self.code_and_errors = {}
        self.run_with_cache = run_with_cache
        self.cache_base_path = "./cesamo/cache/"
        self.min_el_in_each_decil = 5
        self.p_value = 0.05

    def fit(self, X, y=None, dataset_name=None, store_in_cache=True):
        """
        Fit CesamoEncoder to the data.

        Args:
            X (DataFrame): Input data.
            y (array-like): Target variable.
            dataset_name (str): Name of the dataset.
            store_in_cache (bool): Flag indicating whether to store results in cache.

        Returns:
            self
        """
        # check for cache
        dataset_name = None
        cache = None
        if "DATASET_NAME" in os.environ:
            logger.info("Looking for cache data")
            dataset_name = os.environ["DATASET_NAME"]
            cache = self.check_for_cache(os.environ["DATASET_NAME"])

        else:
            logger.info("env var DATASET_NAME not found, cannot apply for cache")

        if cache and dataset_name:
            self.code_and_errors = cache

            return self

        stop_condition = False
        all_features = X.columns
        logger.debug(f"all_features: {all_features}")
        logger.debug(f"catf: {self.cols}")
        for catf in self.cols:
            code_and_errors_tmp_list = []
            error_list = []
            error_avg_list = []
            final_code_and_errors = {}
            iteration = 0
            is_gaussian = False
            logger.debug(f"Applying CESAMO to cat att: {catf}")
            while not stop_condition:
                # make a copy of the trainset
                X_tmp = X.copy()
                # set a list of all attributes and remove current categorical attribute to convert
                features_tmp = all_features.tolist().copy()
                features_tmp.remove(catf)
                # Randomly select variable j (j!=i), (is the variable to be approximated)
                secondary_var = random.choice(features_tmp)
                # Assign random values to all instances of catf
                # if second var is cat, assign random codes
                if secondary_var in self.cols:
                    # change every attribute for the col to random numbers
                    X_tmp, _ = self.convert_catcol_to_random(X_tmp, secondary_var)

                # Replace current cat att by proposed codes
                X_tmp, map_catin_code = self.convert_catcol_to_random(X_tmp, catf)

                # set dependent and independent var
                independent_var = X_tmp[[secondary_var]].copy()
                dependent_var = X_tmp[[catf]].copy()

                # Apply Polynomial feature expansion in order to apply an polynomial regression
                poly_features = PolynomialFeatures(degree=11, include_bias=False)
                X_poly = poly_features.fit_transform(independent_var)

                # remove even degrees in polinomial
                X_poly = self.keep_odd_degrees(X_poly)
                lin_reg = LinearRegression()

                # approximate cat_at as a function of the randomly selected secondary variable
                lin_reg.fit(X_poly, dependent_var)

                # Measure RMSE
                # predictions must be done on unseen data
                predictions = lin_reg.predict(X_poly)
                lin_mse = mean_squared_error(dependent_var, predictions)
                lin_rmse = np.sqrt(lin_mse)

                # store codes and errors
                code_and_errors_tmp_list.append(
                    {"codes": map_catin_code, "error": lin_rmse}
                )
                if (lin_rmse < self.min_error) and self.stop_on_accepted_error:
                    logger.info(f"acceptable error found {lin_rmse}")
                    final_code_and_errors = self.get_best_codes(
                        code_and_errors_tmp_list
                    )
                    logger.info(f"final_code_and_errors: {final_code_and_errors}")

                    # break while loop
                    break

                # Store the errors to calculate if the distribution is Gaussian
                error_list.append(lin_rmse)
                # check if error list has at least N elements
                if len(error_list) == self.n_el_error_list:
                    error_avg_list.append(self.calc_avg(error_list))
                    # check if error avg list has at least N elements
                    if len(error_avg_list) >= self.least_el_in_avg_error_list:
                        is_gaussian = self.check_if_gaussian(error_avg_list)

                        if is_gaussian:
                            logger.info(
                                "Normal distriution of error codes has been reached"
                            )
                            final_code_and_errors = self.get_best_codes(
                                code_and_errors_tmp_list
                            )
                            logger.info(
                                f"final_code_and_errors: {final_code_and_errors}"
                            )
                            break

                    # empty the list
                    error_list = []

                if iteration % 1000 == 0:
                    logger.info(f"iteration: {iteration}")

                iteration += 1

            # end of while
            self.code_and_errors[catf] = final_code_and_errors

        if store_in_cache and dataset_name:
            self.save_in_cache(dataset_name, self.code_and_errors)

        return self

    def get_code(self, n):
        """
        Generate a random decimal code of n digits after the decimal point.

        Args:
            n (int): Number of digits after the decimal point.

        Returns:
            float: Random decimal code.
        """
        return round(random.random(), n)

    def convert_catcol_to_random(self, df, col):
        """
        Convert categorical column to random codes.

        Args:
            df (DataFrame): Input dataframe.
            col (str): Name of the column.

        Returns:
            DataFrame: DataFrame with converted categorical column.
            dict: Mapping of original categories to codes.
        """
        # get every cat instance
        instances = df[col].unique()
        # for every instance, propose a code and set the map cat_instance - code
        map = {}
        for inx, val in enumerate(instances):
            # simplified map
            map[val] = self.get_code(self.code_digits)
        # replace in df the map
        df[col] = df[col].map(map)
        return df, map

    def calc_avg(self, numbers):
        """
        Calculate the average of a list of numbers.

        Args:
            numbers (list): List of numbers.

        Returns:
            float: Average of numbers.
        """
        return np.mean(numbers)

    # normality test
    def check_if_gaussian(self, data):
        """
        Determine if the distribution of data is Gaussian.

        Args:
            data (list): List of data points.

        Returns:
            bool: True if the distribution is Gaussian, False otherwise.
        """
        d1 = -1.28  # (from 0 to 10.03 = 10.03)
        d2 = -0.84  # (from 10.03 to 29.95= 10.02)
        d3 = -0.52  # (from 29.95 to 40.05= 10.02)
        d4 = -0.25  # (from 29.95 to 40.05= 10.02)
        d5 = 0
        d6 = 0.25  # (from 0 to 10.03 = 10.03)
        d7 = 0.52  # (from 10.03 to 29.95= 10.02)
        d8 = 0.84  # (from 29.95 to 40.05= 10.02)
        d9 = 1.28  # (from 29.95 to 40.05= 10.02)

        # iterate over data
        decils = {
            "d1": [],
            "d2": [],
            "d3": [],
            "d4": [],
            "d5": [],
            "d6": [],
            "d7": [],
            "d8": [],
            "d9": [],
            "d10": [],
        }
        zscore_data = zscore(data)
        for num in zscore_data:
            if num < d1:
                decils["d1"].append(num)
            elif d1 <= num < d2:
                decils["d2"].append(num)
            elif d2 <= num < d3:
                decils["d3"].append(num)
            elif d3 <= num < d4:
                decils["d4"].append(num)
            elif d4 <= num < d5:
                decils["d5"].append(num)
            elif d5 <= num < d6:
                decils["d6"].append(num)
            elif d6 <= num < d7:
                decils["d7"].append(num)
            elif d7 <= num < d8:
                decils["d8"].append(num)
            elif d8 <= num < d9:
                decils["d9"].append(num)
            elif num >= d9:
                decils["d10"].append(num)
            else:
                raise Exception("Z-score didnt fit in decils: ", num)

        elements_per_decil = []
        for _, value in decils.items():
            elements_per_decil.append(len(value))
        logger.debug(f"Elements_per_decil: {elements_per_decil}")
        for _, value in decils.items():
            # if there are less than min_el_in_each_decil, we cannot apply yet the chi squared test
            if len(value) < self.min_el_in_each_decil:
                return False

        # At this point, all elements have at least 5 elements
        # Apply chi square tests
        chi_square_test_result = stats.chisquare(f_obs=elements_per_decil)
        logger.debug("Chisquare test value: {chi_square_test_result[1]}")
        if chi_square_test_result[1] > self.p_value:
            return True
        else:
            return False

    def get_best_codes(self, codes_and_errors):
        """
        Get the best codes based on the lowest errors.

        Args:
            codes_and_errors (list): List of dictionaries containing codes and errors.

        Returns:
            dict: Dictionary containing the best codes and error.
        """
        smallest_error_val = None
        smallest_error_index = None
        for index, item in enumerate(codes_and_errors):
            if index == 0:
                smallest_error_val = item["error"]
                smallest_error_index = index
                continue
            if smallest_error_val > item["error"]:
                smallest_error_val = item["error"]
                smallest_error_index = index
        return codes_and_errors[smallest_error_index]

    def transform(self, X, y=None):
        """
        Transform categorical attributes to numerical values.

        Args:
            X (DataFrame): Input data.
            y (array-like): Target variable.

        Returns:
            DataFrame: Transformed data.
        """
        X_tmp = X.copy()
        for cat_col in self.cols:
            logger.debug(
                f'self.code_and_errors[cat_col]["codes"]:{self.code_and_errors[cat_col]["codes"]}'
            )
            logger.debug(f"cat_col {cat_col}")
            X_tmp[cat_col] = X_tmp[cat_col].map(str)
            X_tmp[cat_col] = X_tmp[cat_col].map(self.code_and_errors[cat_col]["codes"])

        if self.handle_unknown == "value":
            # take the target mean
            dependent_var = y
            dependent_var = dependent_var.astype(float)
            X_tmp = X_tmp.fillna(dependent_var.mean())

        return X_tmp

    def check_for_cache(self, dataset_name):
        """
        Check if cache exists for the dataset.

        Args:
            dataset_name (str): Name of the dataset.

        Returns:
            dict: Cached data if exists, otherwise an empty dictionary.
        """
        logger.debug(f"checking if exist file for dataset_name: {dataset_name}")
        # check in cache dir if dataset match the json file.
        cache_base_path = self.cache_base_path
        files = os.listdir(cache_base_path)
        files_name = [x.replace(".json", "") for x in files]

        # if not, return empty dict
        if dataset_name not in files_name:
            logger.info(f"cache NOT found for dataset")
            return {}

        logger.info(f"cache found for dataset")
        json_to_load = cache_base_path + dataset_name + ".json"
        # if exists, load the json into a dict
        with open(json_to_load, "r") as jsonFile:
            jsonObject = json.load(jsonFile)
            jsonFile.close()

        return jsonObject

    def save_in_cache(self, dataset_name, codes_and_error):
        """
        Save data in cache.

        Args:
            dataset_name (str): Name of the dataset.
            codes_and_error (dict): Dictionary containing codes and errors.
        """
        logger.debug("Saving in cache")
        # save in a $$data_name$$.json the codes and erros
        file_name = self.cache_base_path + dataset_name + ".json"
        with open(file_name, "w") as fp:
            json.dump(codes_and_error, fp)
        return

    def keep_odd_degrees(self, X):
        """
        Keep only the odd degrees in polynomial features.

        Args:
            X (array-like): Polynomial features.

        Returns:
            array-like: Polynomial features with only odd degrees.
        """
        X_poly = X
        X_poly_odds = []
        for idx, item in enumerate(X_poly):
            X_poly_odds.append([])
            for idx_2_loop, item_2_loop in enumerate(X_poly[idx]):
                if ((idx_2_loop + 1) % 2) == 1:
                    X_poly_odds[idx].append(item_2_loop)
        return X_poly_odds
