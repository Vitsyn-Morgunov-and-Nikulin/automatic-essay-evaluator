import pandas as pd


class BaseExtractor:

    def generate_features(self, X: pd.Series) -> pd.DataFrame:
        """Generates features for model

        :param X: Series, that contains texts
        :return: Dataframe with columns - features, generated from text
        """
        raise NotImplementedError(f"Abstract class {type(self).__name__} is used")
