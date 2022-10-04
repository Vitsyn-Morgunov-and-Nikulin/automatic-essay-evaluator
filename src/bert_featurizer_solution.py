import os
from pathlib import Path
from typing import Union

import pandas as pd
import torch.cuda
from sklearn.model_selection import train_test_split

from bert_featurizer import BertFeatureExtractor
from src.constant_predictor import set_env_if_kaggle_environ
from src.text_cleaning.text_feature_extractor import TextFeatureExtractor
from src.solution import Solution
from catboost import CatBoostRegressor
from src.metrics import MSEMetric


class BertFeaturePredictor(Solution):
    def __init__(self):
        super(BertFeaturePredictor, self).__init__()
        self.device = 'GPU' if torch.cuda.is_available() else None

        self.feature_extractor = TextFeatureExtractor()
        self.bert = BertFeatureExtractor(model_name='bert-base-uncased')

        # classification model for each column
        self.columns = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        self.models = [CatBoostRegressor(task_type=self.device, verbose=False, iterations=5000) for _ in range(len(self.columns))]

    def _preprocess_texts(self, X: pd.Series):
        cleaned_text = self.feature_extractor.preprocess_texts(X)
        bert_features = self.bert.extract_features(cleaned_text)
        handcrafted_features = self.feature_extractor.extract_features(X)
        features_df = pd.concat([bert_features, handcrafted_features], axis='columns')

        return features_df

    def fit(self, X: pd.Series, y: pd.DataFrame):
        features_df = self._preprocess_texts(X)

        for ii, column in enumerate(self.columns):
            print(f"-> Training model on: {column}...")
            model = self.models[ii]
            target = y[column]

            model.fit(X=features_df, y=target)

    def predict(self, X: pd.Series) -> pd.DataFrame:
        features_df = self._preprocess_texts(X)

        prediction = {}
        for ii, column in enumerate(self.columns):
            print(f"-> Predicting model on: {column}")
            model = self.models[ii]
            prediction[column] = model.predict(features_df)

        return pd.DataFrame(prediction)

    def save(self, directory: Union[str, Path]):
        pass

    def load(self, directory: Union[str, Path]):
        pass


def main():
    set_env_if_kaggle_environ()

    train_df_path = Path(os.environ['DATA_PATH']) / 'train.csv'
    test_df_path = Path(os.environ['DATA_PATH']) / 'test.csv'

    if not test_df_path.is_file():
        raise OSError(f"File not found: {test_df_path.absolute()}")

    if not train_df_path.is_file():
        raise OSError(f"File not found: {train_df_path.absolute()}")

    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)

    predictor = BertFeaturePredictor()
    metric = MSEMetric()

    train_data, val_data = train_test_split(train_df, test_size=0.2)

    predictor.fit(train_data.full_text, train_data[predictor.columns])
    y_pred = predictor.predict(val_data.full_text)

    print(f"Calculation class metric: {metric.evaluate_mse_class(y_pred, val_data[predictor.columns])}")
    print(f"Calculating kaggle metric: {metric.evaluate(y_pred, val_data[predictor.columns])}")

    submission_df = predictor.predict(test_df.full_text)

    submission_df.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    main()
