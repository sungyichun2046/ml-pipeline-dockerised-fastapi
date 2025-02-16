""" Abstract class Model and concrete class ExtraTrees."""
from __future__ import annotations

from abc import ABC, abstractmethod

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


class Model(ABC):
    """ Abstract Model class having method train to implement."""
    def __init__(self):
        """ Instantiate Model."""
        self.model = None

    @abstractmethod
    def train(self, train_features: pd.DataFrame, train_labels: np.ndarray) -> None:
        """ Train the model.

        :param train_features: features to train model
        :param train_labels: pandas series "BAD" to train model
        :return: None
        """
        pass

    def export_model(self, path: str) -> None:
        """ Export a model for it to be reused in production.

        :param path: path to save the model
        :return: None
        """
        joblib.dump(self.model, path)


class ExtraTrees(Model):
    """ Extra-trees classifier modeling."""
    def train(self, train_features: pd.DataFrame, train_labels: pd.Series) -> None:
        """ Train extra-trees classifier.

        :param train_features: features to train model
        :param train_labels:  pandas series "BAD" to train model
        :return: None
        """

        self.model = ExtraTreesClassifier()
        # Train the model using the training sets
        self.model.fit(train_features, train_labels)
