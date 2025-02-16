""" Generate features of one dataframe to predict, to be able to train a model."""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

if TYPE_CHECKING:
    import logging

    import pandas as pd


class FeaturesGenerator:
    """ Generate features of data to predict."""

    def __init__(self, logger: logging.Logger, df: pd.DataFrame):
        """ Instantiate FeatureGenerator.

        :param logger: python logger
        :param df: dataframe containing client data
        """
        self.logger = logger
        self.features = df

        # "BAD" exist during training, but not for prediction in production
        if "BAD" in df.columns:
            self.features = self.features.drop("BAD", axis=1)
            self.target = df["BAD"]
        self.num_cols = self.features.select_dtypes(include='number').columns.to_list()
        self.str_cols = list(set(self.features.columns.to_list()) - set(self.num_cols))

    def feature_encoding(self) -> None:
        """ Convert the categorical values of string columns into numerical ones.
        :return: None
        """
        job_map = {"Other": 1, "Office": 2, "Sales": 3, "Mgr": 4, "ProfExe": 5, "Self": 6}
        reason_map = {"DebtCon": 1, "HomeImp": 2, "Other": 3}
        self.features["JOB"] = self.features["JOB"].map(job_map)
        self.features["REASON"] = self.features["REASON"].map(reason_map)

    def impute_missing_values(self) -> None:
        """ Missing value imputation.
        :return: None
        """
        if len(self.num_cols) == 0 or len(self.str_cols) == 0:
            self.logger.warning("Numerical features or string features no detected")
        # For numerical features, replace missing values with median
        self.features[self.num_cols] = self.features[self.num_cols].fillna(self.features[self.num_cols].median())
        # For string features, replace missing values with the most frequent values

        if len(self.str_cols) != 0:
            imp = SimpleImputer(strategy="most_frequent")
            self.features[self.str_cols] = imp.fit_transform(self.features[self.str_cols])
        # Edge case, e.g. all values in one column are missing, replace missing values with 0 or empty string
        if sum(self.features[self.num_cols].isna().sum()) != 0:
            self.logger.warning("Fill missing values with 0 for columns : %s", self.num_cols)
            self.features[self.num_cols] = self.features[self.num_cols].fillna(0)
        if sum(self.features[self.str_cols].isna().sum()) != 0:
            self.logger.warning("Fill missing values with empty string for columns : %s", self.str_cols)
            self.features[self.str_cols] = self.features[self.str_cols].fillna('')

    def resampling(self) -> None:
        """ Resampling the train dataset using SMOTE.
        :return: None
        """
        counter = Counter(self.target)
        self.logger.info('Before smote, label distribution = %s', counter)
        smt = SMOTE()

        self.features, self.target = smt.fit_resample(self.features, self.target)

        counter = Counter(self.target)
        self.logger.info('After smote, label distribution = %s', counter)

    def generate(self, tasks: list) -> None:
        """ Call methods in tasks to process data.
        :return: None
        """
        self.logger.info("tasks to run: %s", tasks)
        if len(tasks) != 0:
            self.logger.info("Functions to run = %s", tasks)
            for method in tasks:
                getattr(self, method)()
