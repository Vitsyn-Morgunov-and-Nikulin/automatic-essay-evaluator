import pandas as pd

from src.bert_featurizer import BertFeatureExtractor


def test_feature_extractor():
    models = ['bert-base-uncased', 'distilbert-base-uncased-finetuned-sst-2-english']

    row = pd.Series({
        'full_text': "I think that students would benefit from learning at home,because "
                     "they wont have to change and get up early in the morning to shower and "
                     "do there hair. taking only classes helps them because at there house "
                     "they'll be pay more attention. they will be comfortable at home. "
    })

    for model_name in models:
        feature_extractor = BertFeatureExtractor(model_name=model_name)

        output_features = feature_extractor.extract_features(row)
        assert len(output_features) == 768
