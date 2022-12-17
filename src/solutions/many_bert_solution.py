import os
from pathlib import Path
from typing import Union

import pandas as pd
import torch.cuda
from catboost import CatBoostRegressor
from catboost.utils import get_gpu_device_count
from easydict import EasyDict as edict

from src.cross_validate import CrossValidation
from src.feature_extractors.bert_pretrain_extractor import \
    ManyBertPretrainFeatureExtractor
from src.feature_extractors.text_statistics_extractor import \
    HandcraftedTextFeatureExtractor
from src.solutions.base_solution import BaseSolution
from src.solutions.constant_predictor import load_train_test_df
from src.spell_checker import SmartSpellChecker
from src.text_preprocessings.spellcheck_preprocessing import \
    SpellcheckTextPreprocessor
from src.utils import get_x_columns, seed_everything, validate_x, validate_y

seed_everything()

spellcheck = SmartSpellChecker()


class ManyBertWithHandcraftedFeaturePredictor(BaseSolution):

    def __init__(
        self,
        model_names: list,
        catboost_iter: int,
        saving_dir: str,
    ):
        super(ManyBertWithHandcraftedFeaturePredictor, self).__init__()

        self.feature_extractor = HandcraftedTextFeatureExtractor(spellcheck)
        self.text_preprocessing = SpellcheckTextPreprocessor(spellcheck)
        self.berts = ManyBertPretrainFeatureExtractor(model_names=model_names)

        self.device = 'GPU' if torch.cuda.is_available() else None
        self.task_type = 'GPU' if get_gpu_device_count() > 0 else 'CPU'

        # classification model for each column
        self.columns = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        self.models = [
            CatBoostRegressor(
                iterations=catboost_iter,
                task_type=self.task_type,
                verbose=True,
            ) for _ in range(len(self.columns))
        ]

    def transform_data(self, X: pd.Series) -> pd.DataFrame:
        cleaned_text = self.text_preprocessing.preprocess_data(X)
        bert_features = self.berts.generate_features(cleaned_text)
        handcrafted_features = self.feature_extractor.generate_features(X)
        features_df = pd.concat([bert_features, handcrafted_features], axis='columns')
        return features_df

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> None:
        validate_x(X)
        validate_y(y)

        features_df = self.transform_data(X.full_text)

        for ii, column in enumerate(self.columns):
            print(f"-> Training model on: {column}...")
            model = self.models[ii]
            target = y[column]
            torch.cuda.empty_cache()
            model.fit(X=features_df, y=target)
            torch.cuda.empty_cache()

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
    config = edict(
        dict(
            model_names=[
                'bert-base-uncased',
                'bert-base-cased',
                'vblagoje/bert-english-uncased-finetuned-pos',
                'bert-base-multilingual-cased',
                'unitary/toxic-bert',
                'bert-large-uncased'
            ],
            catboost_iter=5000,
            n_splits=5,
            saving_dir='checkpoints/ManyBertWithHandcraftedFeaturePredictor',
        )
    )

    train_df, test_df = load_train_test_df()

    x_columns = get_x_columns()
    train_x, train_y = train_df[x_columns], train_df.drop(columns=['full_text'])

    predictor = ManyBertWithHandcraftedFeaturePredictor(
        model_names=config.model_names,
        catboost_iter=config.catboost_iter,
        saving_dir=config.saving_dir,
    )
    cv = CrossValidation(saving_dir=config.saving_dir, n_splits=config.n_splits)

    results = cv.fit(predictor, train_x, train_y)
    print("CV results")
    print(results)

    print(f"CV mean: {results.iloc[len(results) - 1].mean()}")

    cv.save(config.saving_dir)

    submission_df = cv.predict(test_df)
    submission_path = os.path.join(config.saving_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    cv_results_path = os.path.join(config.saving_dir, "cv_results.csv")
    results.to_csv(cv_results_path)

    print("Finished training!")


if __name__ == '__main__':
    main()
