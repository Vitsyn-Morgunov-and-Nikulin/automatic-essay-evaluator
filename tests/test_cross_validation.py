from pathlib import Path

from src.cross_validate import CrossValidation
from src.data_reader import load_train_test_df
from src.solutions.constant_predictor import ConstantPredictorSolution
from src.utils import pandas_set_print_options

pandas_set_print_options()


def test_cross_validation():
    train_df, test_df = load_train_test_df(is_testing=True)

    x_columns = ['text_id', 'full_text']
    X, y = train_df[x_columns], train_df.drop(columns=['full_text'])

    n_splits = 3
    cv = CrossValidation(saving_dir=Path('/tmp/sdfjsld'), n_splits=n_splits)
    predictor = ConstantPredictorSolution()
    cv_scores = cv.fit(predictor, X, y)
    assert cv_scores.shape == (n_splits + 1, 6)

    prediction_df = cv.predict(train_df[x_columns])

    assert prediction_df.shape == (len(train_df), 7)
