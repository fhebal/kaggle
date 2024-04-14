from category_encoders import BinaryEncoder, HashingEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (  # noqa E501
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

TEXT_PREP = {
    'CountVectorizer': CountVectorizer(),
    'TfidfVectorizer': TfidfVectorizer(),
}

NUM_SCALE = {
    'MinMaxScaler': MinMaxScaler(),
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'Normalizer': Normalizer(),
}

CAT_ENCODE = {
    'OneHotEncoder': OneHotEncoder(handle_unknown='ignore'),
    'OrdinalEncoder': OrdinalEncoder(handle_unknown='ignore'),
    'BinaryEncoder': BinaryEncoder(),
    'TargetEncoder': TargetEncoder(),
    'HashingEncoder': HashingEncoder(),
}


def build_preprocessor(config):
    num = config["numeric_transformer"]
    cat = config["categorical_transformer"]
    txt1 = config["text_transformer_1"]['vectorizer']
    txt2 = config["text_transformer_2"]['vectorizer']

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=num["imputer"]["strategy"]),),  # noqa: E501
        ("scaler", NUM_SCALE[num['scaler']]),
        ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cat["imputer"]["strategy"], fill_value=cat["imputer"]["fill_value"],),),  # noqa: E501
        ("encoder", CAT_ENCODE[cat['encoder']]),
        ])
    text_transformer_1 = Pipeline(steps=[
        ("vectorizer", TEXT_PREP[txt1])
        ])
    text_transformer_2 = Pipeline(steps=[
        ("vectorizer", TEXT_PREP[txt2])
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config["numeric_features"]),
            ("text_1", text_transformer_1, config["text_feature_1"][0]),
            ("text_2", text_transformer_2, config["text_feature_2"][0]),
            ("cat", categorical_transformer, config["categorical_features"],),
        ])

    return preprocessor


def build_model(config, preprocessor):
    model_class = getattr(
        __import__(
            "sklearn.linear_model",
            fromlist=[config["type"]]), config["type"],
    )
    model = model_class(**config["parameters"])
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif, k=1500)),
        ("model", model)])  # noqa E501

    return pipeline
