from pathlib import Path
from typing import Union

import pandas as pd
import torch.cuda
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from bert_featurizer import BertFeatureExtractor
from src.constant_predictor import load_train_test_df
from src.metrics import MSEMetric
from src.solution import Solution
from src.text_cleaning.spell_checker import SmartSpellChecker
from src.text_cleaning.text_feature_extractor import HandcraftedTextFeatureExtractor
from src.text_cleaning.text_preprocessing import TextPreprocessor


class BertWithHandcraftedFeaturePredictor(Solution):
    device = 'GPU' if torch.cuda.is_available() else None

    def __init__(self, config: dict):
        super(BertWithHandcraftedFeaturePredictor, self).__init__()
        spellcheck = SmartSpellChecker()

        self.feature_extractor = HandcraftedTextFeatureExtractor(spellcheck)
        self.text_preprocessing = TextPreprocessor(spellcheck)
        self.bert = BertFeatureExtractor(model_name=config['model_name'])

        # classification model for each column
        self.columns = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        self.models = [
            CatBoostRegressor(
                iterations=config['catboost_iter'],
                task_type=self.device,
                verbose=False,
            ) for _ in range(len(self.columns))
        ]

    def _preprocess_texts(self, X: pd.Series):
        cleaned_text = self.text_preprocessing.preprocess_texts(X)
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
        directory = Path(directory)
        if not directory.exists():
            directory.mkdir(parents=True)

        for ii, model in enumerate(self.models):
            column = self.columns[ii]
            path = directory / f'catboost_{column}.cbm'
            model.save_model(str(path))

        print("Successfully saved model!")

    def load(self, directory: Union[str, Path]):
        directory = Path(directory)
        if not directory.is_dir():
            raise OSError(f"Dir. {directory.absolute()} does not exist")

        for ii, model in enumerate(self.models):
            column = self.columns[ii]
            path = directory / f'catboost_{column}.cbm'
            model.load_model(str(path))

        print("Successfully loaded model!")


def main():
    config = {
        'model_name': 'bert-base-uncased',
        'catboost_iter': 5000,
    }

    train_df, test_df = load_train_test_df()

    predictor = BertWithHandcraftedFeaturePredictor(config)
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
