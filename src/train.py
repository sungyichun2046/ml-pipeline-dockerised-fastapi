""" Train model and see model evaluation metrics."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import pandas as pd
from joblib import load
from sklearn.metrics import classification_report

from src.features_generator import FeaturesGenerator
from src.model import ExtraTrees
from src.model_trainer import ModelTrainer

if TYPE_CHECKING:
    import numpy as np

    from src.model import Model

BASE_DIR = Path(__file__).resolve(strict=True).parent


def load_and_split_data(filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Split data in training set and test set.

    :param filename: filename to store training set and test set
    :return: train_df, test_df as dataframes
    """
    folder = os.path.join(BASE_DIR, "data")
    train_path = os.path.join(folder, "train_df.csv")
    test_path = os.path.join(folder, "test_df.csv")

    if os.path.isfile(train_path):
        train_df = pd.read_csv(train_path, index_col=0)
        test_df = pd.read_csv(test_path, index_col=0)

    else:
        data_df = pd.read_csv(filename)

        # 80/20 train/test split
        train_df = data_df.sample(frac=0.8)
        test_df = data_df.loc[~data_df.index.isin(train_df.index)]

        # Store them to use always the same train_df, test_df for model benchmarking
        train_df.to_csv(os.path.join(folder, "train_df.csv"))
        test_df.to_csv(os.path.join(folder, "test_df.csv"))
    return train_df, test_df


def train(
    model: Model = ExtraTrees, model_path: str = os.path.join(BASE_DIR, "trained_models", "extraTrees_model.sav"),
) -> None:
    """ Train model and predict on test set to get evaluation metrics.

    :param model: Model object
    :param model_path: path to store trained model
    :return: None
    """
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )

    train_df, test_df = load_and_split_data(
        filename=os.path.join(BASE_DIR, "data", "hmeq.csv"),
    )
    logging.info(
        "data split: train_df = %d, test_df = %d",
        train_df.shape[0], test_df.shape[0]
    )

    train_features = FeaturesGenerator(logger=logging, df=train_df)
    train_features.generate(tasks=["feature_encoding", "impute_missing_values", "resampling"])

    model_trainer = ModelTrainer(logger=logging)
    model_trainer.train(
        train_features=train_features.features,
        train_labels=train_features.target,
        model=model,
        model_path=model_path,
    )
    # Predict
    predictions = predict(test_df)

    # Evaluation
    logging.info(classification_report(y_true=test_df["BAD"], y_pred=predictions))


def predict(data_df: pd.DataFrame) -> np.ndarray:
    """ Predict value "Bad" of client data as pandas dataframe.

    :param data_df: client data
    :return: numpy ndarray containing predictions of model
    """
    if data_df.empty:
        logging.warning("Empty dataframe to predict")
    features = FeaturesGenerator(logger=logging, df=data_df)
    tasks = ["impute_missing_values"]
    if data_df["REASON"].dtype != int or data_df["REASON"].dtype != int:
        tasks.append("feature_encoding")
    features.generate(tasks)

    with open(os.path.join(BASE_DIR, "trained_models", "extraTrees_model.sav"), "rb") as infile:
        model = load(infile)

    predictions = model.predict(features.features)
    return predictions


if __name__ == "__main__":
    train()
