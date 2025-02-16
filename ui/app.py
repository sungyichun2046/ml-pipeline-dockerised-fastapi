""" User interface to predict credit score of a client. """
from __future__ import annotations

import logging
import socket
import threading
import time
from pathlib import Path
from typing import TypedDict

import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator

from src.train import predict

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent
logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
)

logging.info("Modal loaded, ready for serving")


class Client(BaseModel):
    """ Define model class that represent data's fields and their desired type."""
    loan: int  # amount of the loan request
    mortdue: float  # amount due on existing mortgage
    value: float  # value of current property
    reason: int  # 1'DebtCon' (debt consolidation), 2'HomeImp' (home improvement), 3'Other'
    job: int  # 1'Other', 2'Office', 3'Sales', 4'Mgr', 5'ProfExe', 6'Self'
    yoj: float  # years at present job
    derog: float  # number of major derogatory reports
    delinq: float  # number of delinquent credit lines
    clage: float  # age of oldest trade line in months
    ninq: float  # number of recent credit lines
    clno: float  # number of credit lines
    debtinc: float  # debt-to-income ratio

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, reason: int) -> int:
        """ Data validation for reason attribute.

        :param reason: integer value
        :return: integer reason value
        """
        if reason not in [1, 2, 3]:
            raise ValueError("reason should be between 1 and 3")
        return reason

    @field_validator("job")
    @classmethod
    def validate_job(cls, job: int) -> int:
        """ Data validation for job attribute.

        :param job: integer value
        :return: integer job value
        """
        if job not in [1, 2, 3, 4, 5, 6]:
            raise ValueError("job should be between 1 and 6")
        return job


app = FastAPI()


# Mount the "static" folder to serve CSS and other static files
app.mount(
    "/static", StaticFiles(directory=f"{BASE_DIR}/static"), name="static")

# Mount the "templates" folder to load HTML templates
templates = Jinja2Templates(directory=f"{BASE_DIR}/templates")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request) -> TypedDict:
    """ Get method returns root url content.

    :param request: Request object
    :return: root url content
    """
    return templates.TemplateResponse("index.html", {"request": request, })


@app.get('/cpu-intensive')
def cpu_intensive() -> TypedDict:
    """ Multithreading which allows concurrent execution of multiple threads within a single process.

    Called to be able to test kubernetes autoscaling

    :return: json response
    """
    def cpu_task():
        end_time = time.time() + 8
        while time.time() < end_time:
            pass  # Busy-wait to simulate CPU load

    thread = threading.Thread(target=cpu_task)
    thread.start()
    thread.join()  # Wait for the thread to complete

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return JSONResponse({
        'message': 'CPU intensive task completed',
        'hostname': hostname,
        'ip_address': ip_address
    })


@app.post("/predict_with_ui", response_class=HTMLResponse, )
def predict_with_ui(
    request: Request, loan: int = Form(...), mortdue: float = Form(...),
    value: float = Form(...), reason: int = Form(...), job: int = Form(...),
    yoj: float = Form(...), derog: float = Form(...), delinq: float = Form(...),
    clage: float = Form(...), ninq: float = Form(...), clno: float = Form(...), debtinc: float = Form(...)
) -> TypedDict:  # sentence: str = Form(...)
    """ Post method to predict credit score on user interface.

    :param request: Request object
    :param loan: amount of the loan
    :param mortdue: amount due on existing mortgage
    :param value: value of current property
    :param reason: integer, 1 means 'DebtCon'(debt consolidation), 2 means 'HomeImp'(home improvement), 3 means 'Other'
    :param job: integer, 1 means 'DebtCon' (debt consolidation), 2 means 'HomeImp' (home improvement), 3 means 'Other'
    :param yoj: years at present job
    :param derog: number of major derogatory reports
    :param delinq: number of delinquent credit lines
    :param clage: age of oldest trade line in months
    :param ninq: number of recent credit lines
    :param clno: number of credit lines
    :param debtinc: debt-to-income ratio
    :return: Json response
    """
    # Data validation
    client = Client(
        loan=loan, mortdue=mortdue, value=value, reason=reason, job=job,
        yoj=yoj, derog=derog, delinq=delinq, clage=clage, ninq=ninq, clno=clno, debtinc=debtinc,
    )
    data = client.__dict__
    data_df = pd.DataFrame([data], columns=data.keys())

    # Column names are uppercase in training dataset
    data_df.columns = map(str.upper, data_df.columns)
    predictions = predict(data_df)
    logging.info("predictions=%s, type=%s", predictions, type(predictions))
    return templates.TemplateResponse(
        "prediction.html",
        {"request": request, "prediction": predictions.item(0)},
    )


@app.post("/predict_score_no_ui")
async def predict_score_no_ui(data: Client) -> TypedDict:
    """ Predict credit score without user interface.

    :param data: Client instance
    :return: json response
    """
    time.sleep(2)
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    data = data.__dict__
    data_df = pd.DataFrame([data], columns=data.keys())

    # Column names are uppercase in training dataset
    data_df.columns = map(str.upper, data_df.columns)
    predictions = predict(data_df)
    logging.info("predictions=%s, type=%s", predictions, type(predictions))
    return JSONResponse({
        "prediction": predictions.item(0),
        'hostname': hostname,
        'ip_address': ip_address,
    })
