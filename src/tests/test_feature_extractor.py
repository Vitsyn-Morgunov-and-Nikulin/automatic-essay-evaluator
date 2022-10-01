import pandas as pd

from src.bert_featurizer import BertFeatureExtractor


def test_feature_extractor():
    models = ['bert-base-uncased', 'distilbert-base-uncased-finetuned-sst-2-english']

    text_examples = pd.DataFrame({
        'full_text': [
            "I think that students would benefit from learning at home,because "
            "they wont have to change and get up early in the morning to shower and "
            "do there hair. taking only classes helps them because at there house "
            "they'll be pay more attention. they will be comfortable at home. ",

            "Have you ever solved a math problem in less than 30 seconds? "
            "Math is important to many people. Even president Trump uses "
            "math to buy food or pay back to important people. "
        ]
    }).full_text

    for model_name in models:
        feature_extractor = BertFeatureExtractor(model_name=model_name)

        output_features = feature_extractor.extract_features(text_examples)
        assert len(output_features) == 2 and len(output_features.columns) == 768
