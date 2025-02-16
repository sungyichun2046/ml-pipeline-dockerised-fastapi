""" Autoscaling with Kubernetes test."""
import argparse
import concurrent.futures
import csv
import os
import time
from pathlib import Path

import requests

KUBERNETES_URL = "http://192.168.49.2:31230"


def make_request(url: str, payload: dict = None) -> dict:
    """ Sends HTTP requests to the given URL and returns the JSON response.

    :param url: endpoint of prediction to test
    :param payload: payload to send to endpoint
    :return: response of request
    """
    try:
        if payload:
            response = requests.post(url, json=payload)
        else:
            response = requests.get(url)

        # Check if the response is JSON
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError:
            return {'error': 'Invalid JSON response', 'content': response.text}
    except requests.RequestException as error:
        return {'error': str(error)}


def save_to_csv(data: list, filename: str = 'concurrent_requests.csv') -> None:
    """ Save responses to a csv file.

    :param data: list of request responses
    :param filename: filename to store list of request responses as a dataframe
    :return: None
    """
    # Determine the set of all keys present in the responses
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())

    base_dir = Path(__file__).resolve(strict=True).parent

    with open(os.path.join(base_dir, filename), 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=list(all_keys))
        dict_writer.writeheader()
        dict_writer.writerows(data)


def main():
    """
    Generate concurrent requests to API endpoint and saves the responses to a CSV file.
    To analyze the performance and behavior of scaled API
    """
    responses = []
    # Define the URLs and payloads for the requests
    url_predict = f"{KUBERNETES_URL}/predict_score_no_ui"

    # Generate 100 payloads
    payload = {
        "loan": 2000, "mortdue": 25000.0, "value": 39025.0, "reason": 1, "yoj": 12.5, "derog": 0.0,
        "delinq": 0.0, "clage": 95.366666667, "ninq": 1.0, "clno": 9.0, "debtinc": 1.3, "job": 3,
    }
    payloads = [payload] * 500

    # Submit the tasks to send requests to endpoint to predict credit score
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:

        # Make concurrent POST requests to the reverse endpoint
        future_reverse = [executor.submit(make_request, url_predict, payload) for payload in payloads]
        # concurrent.futures.as_completed() returns each Future as it completes, allowing us to process results.
        for future in concurrent.futures.as_completed(future_reverse):
            responses.append(future.result())

    # Save responses to CSV
    save_to_csv(responses, filename=f'concurrent_requests_CPU_Intensive_{argument.cpu_intensive}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-intensive', default=False, type=lambda x: x == 'True')
    argument = parser.parse_args()

    if argument.cpu_intensive:
        url_cpu_intensive = f'{KUBERNETES_URL}/cpu-intensive'
        requests.get(url_cpu_intensive)

    time.sleep(5)
    main()
