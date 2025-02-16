""" Test predict function in src."""
import numpy as np
import pandas as pd

from src.train import predict


def test_predict_with_missing_yoc_feature() -> None:
    """ Test data containing missing yoc value.
    :return: None
    """
    data = {
        'LOAN': 2000,
        'MORTDUE': 25000.0,
        'VALUE': 39025.0,
        'REASON': 2,
        'JOB': 1,
        'YOJ': np.nan,
        'DEROG': 0.0,
        'DELINQ': 0.0,
        'CLAGE': 95.366666667,
        'NINQ': 1.0,
        'CLNO': 9.0,
        'DEBTINC': 4,
    }
    data_df = pd.DataFrame([data], columns=data.keys())
    predictions = predict(data_df)
    assert predictions[0] in [0, 1]


def test_predict_with_missing_job_feature() -> None:
    """ Test data containing missing job value.
    :return: None
    """
    data = {
        'LOAN': 2000,
        'MORTDUE': 25000.0,
        'VALUE': 39025.0,
        'REASON': 1,
        'JOB': np.nan,
        'YOJ': 12.5,
        'DEROG': 0.0,
        'DELINQ': 0.0,
        'CLAGE': 95.366666667,
        'NINQ': 1.0,
        'CLNO': 9.0,
        'DEBTINC': 3,
    }
    data_df = pd.DataFrame([data], columns=data.keys())
    predictions = predict(data_df)
    assert predictions[0] in [0, 1]
