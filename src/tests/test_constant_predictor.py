import pandas as pd

from src.constant_predictor import ConstantPredictorSolution


def test_constant_predictor():
    test_df = pd.DataFrame({
        "text_id": [0],
        "full_data": ["I was thinking about the studies"]
    })

    predictor = ConstantPredictorSolution()

    submission_df = predictor.predict(test_df)

    assert len(submission_df) == 1
