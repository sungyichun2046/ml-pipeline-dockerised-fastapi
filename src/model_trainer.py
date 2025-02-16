""" Use a concrete class in modeling to train a model."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Type

import pandas as pd

if TYPE_CHECKING:
    from model import Model

BASE_DIR = Path(__file__).resolve(strict=True).parent


class ModelTrainer:
    """ Train and save models."""
    def __init__(self, logger: logging.Logger) -> None:
        """Instantiate ModelTrainer

        :param logger: logger object
        """
        self.logger = logger

    @staticmethod
    def train(
        train_features: pd.DataFrame, train_labels: list, model: Type[Model], model_path: str,
    ) -> None:
        """
        Train model

        :param train_features: dataframe containing features to train
        :param train_labels: training labels
        :param model: a model class, with a function train
        :param model_path: path to store model

        :return: None
        """
        model = model()
        model.train(
            train_features=train_features,
            train_labels=train_labels,
        )
        model.export_model(model_path)
