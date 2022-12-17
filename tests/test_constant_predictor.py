from src.data_reader import load_train_test_df
from src.solutions.constant_predictor import ConstantPredictorSolution


def test_constant_predictor():
    _, test_df = load_train_test_df(is_testing=True)

    predictor = ConstantPredictorSolution()
    submission_df = predictor.predict(test_df)

    assert submission_df.shape == (len(test_df), 7)
