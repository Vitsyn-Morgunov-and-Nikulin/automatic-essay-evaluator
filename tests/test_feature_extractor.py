from src.data_reader import load_train_test_df
from src.feature_extractors.bert_pretrain_extractor import \
    BertPretrainFeatureExtractor


def test_pretrain_feature_extractor():
    models = ['distilbert-base-uncased-finetuned-sst-2-english', 'bert-base-uncased']
    train_df, _ = load_train_test_df(is_testing=True)

    for model_name in models:
        feature_extractor = BertPretrainFeatureExtractor(model_name=model_name)

        output_features = feature_extractor.generate_features(train_df.full_text)

        assert len(output_features) == 5 and len(output_features.columns) == 768
