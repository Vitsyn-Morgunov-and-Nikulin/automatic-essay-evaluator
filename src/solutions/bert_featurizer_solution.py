from pathlib import Path
from typing import Union

import pandas as pd
import torch.cuda
from catboost import CatBoostRegressor
from transformers import logging as transformers_logging

from src.cross_validate import CrossValidation
from src.feature_extractors.bert_pretrain_extractor import \
    BertPretrainFeatureExtractor
from src.feature_extractors.text_statistics_extractor import \
    HandcraftedTextFeatureExtractor
from src.solutions.base_solution import BaseSolution
from src.solutions.constant_predictor import load_train_test_df
from src.spell_checker import SmartSpellChecker
from src.text_preprocessings.spellcheck_preprocessing import \
    SpellcheckTextPreprocessor
from src.utils import get_x_columns, seed_everything, validate_x, validate_y

transformers_logging.set_verbosity_error()

seed_everything()

spellcheck = SmartSpellChecker()


class BertWithHandcraftedFeaturePredictor(BaseSolution):
    device = 'GPU' if torch.cuda.is_available() else None

    def __init__(
        self,
        model_name: str,
        catboost_iter: int,
        saving_dir: str
    ):
        super(BertWithHandcraftedFeaturePredictor, self).__init__()

        self.feature_extractor = HandcraftedTextFeatureExtractor(spellcheck)
        self.text_preprocessing = SpellcheckTextPreprocessor(spellcheck)
        self.bert = BertPretrainFeatureExtractor(model_name=model_name)

        # classification model for each column
        self.columns = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        self.models = [
            CatBoostRegressor(
                iterations=catboost_iter,
                task_type="CPU",  # self.device,
                verbose=True,
            ) for _ in range(len(self.columns))
        ]

    def transform_data(self, X: pd.Series) -> pd.DataFrame:
        cleaned_text = self.text_preprocessing.preprocess_data(X)
        bert_features = self.bert.generate_features(cleaned_text)
        bert_features = self.bert.generate_features(X)
        handcrafted_features = self.feature_extractor.generate_features(X)
        features_df = pd.concat([bert_features, handcrafted_features], axis='columns')

        features_df = bert_features

        return features_df

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> None:
        validate_x(X)
        validate_y(y)

        features_df = self.transform_data(X.full_text)

        for ii, column in enumerate(self.columns):
            print(f"-> Training model on: {column}...")
            model = self.models[ii]
            target = y[column]

            model.fit(X=features_df, y=target)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        validate_x(X)

        features_df = self.transform_data(X.full_text)

        prediction = {}
        for ii, column in enumerate(self.columns):
            print(f"-> Predicting model on: {column}")
            model = self.models[ii]
            prediction[column] = model.predict(features_df)

        y_pred = pd.DataFrame(prediction, index=X.index)
        y_pred['text_id'] = X.text_id

        return y_pred

    def save(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        if not directory.exists():
            directory.mkdir(parents=True)

        for ii, model in enumerate(self.models):
            column = self.columns[ii]
            path = directory / f'catboost_{column}.cbm'
            model.save_model(str(path))

    def load(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        if not directory.is_dir():
            raise OSError(f"Dir. {directory.absolute()} does not exist")

        for ii, model in enumerate(self.models):
            column = self.columns[ii]
            path = directory / f'catboost_{column}.cbm'
            model.load_model(str(path))


def main():
    config = {
        'model_name': 'bert-base-uncased',
        'catboost_iter': 5000,
        'n_splits': 5,
        'saving_dir': 'checkpoints/BertWithHandcraftedFeaturePredictor',
    }
    saving_dir = Path("checkpoints/bert_featurizer_solution")
    saving_dir.mkdir(exist_ok=True, parents=True)

    train_df, test_df = load_train_test_df()
    # train_df = train_df.iloc[:10]

    x_columns = get_x_columns()
    train_x, train_y = train_df[x_columns], train_df.drop(columns=['full_text'])

    predictor = BertWithHandcraftedFeaturePredictor(
        model_name='bert-base-uncased',
        catboost_iter=5000,
        saving_dir='checkpoints/BertWithHandcraftedFeaturePredictor'
    )
    cv = CrossValidation(saving_dir=str(saving_dir), n_splits=config['n_splits'])

    results = cv.fit(predictor, train_x, train_y)
    print("CV results")
    print(results)
    results.to_csv("cv_results.csv")

    print(f"CV mean: {results.iloc[len(results) - 1].mean()}")

    submission_df = cv.predict(test_df)
    submission_df.to_csv("submission.csv", index=False)

    cv.save(config['saving_dir'])

    print("Finished training!")


if __name__ == '__main__':
    main()
