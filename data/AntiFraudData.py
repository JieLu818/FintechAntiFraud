import os
import pandas as pd

RAW_DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + '/anti_fraud_data.csv'
TRAINING_PATH = os.path.dirname(os.path.abspath(__file__)) + '/training_data_clean.csv'
TESTING_PATH = os.path.dirname(os.path.abspath(__file__)) + '/testing_data_clean.csv'

class AntiFraudData:
    def __init__(self, path=RAW_DATA_PATH):
        self._raw = pd.read_csv(path)

    def retrieve_no_missing_value_data(self, data_path=TRAINING_PATH):
        return pd.read_csv(data_path)

    @property
    def raw(self):
        return self._raw


if __name__ == '__main__':
    data = AntiFraudData()
    print(data._raw)