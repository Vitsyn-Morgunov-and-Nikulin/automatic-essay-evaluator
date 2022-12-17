import pandas as pd


class BasePreprocessor:

    def preprocess_data(self, data: pd.Series) -> pd.Series:
        """Performs preprocessing of raw texts, returns cleaned texts

        :param X: raw texts
        :return: cleaned texts
        """

        raise NotImplementedError(f"Abstract class {type(self).__name__} is used")
