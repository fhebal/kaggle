from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(config):
    nis = config["numeric_transformer"]["imputer"]["strategy"]
    cis = config["categorical_transformer"]["imputer"]["strategy"]  # noqa: E501
    cif = config["categorical_transformer"]["imputer"]["fill_value"]  # noqa: E501

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=nis),),
        ("scaler", StandardScaler()),
        ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cis, fill_value=cif,),),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
    text_transformer = Pipeline(steps=[
        ("vectorizer", CountVectorizer())
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config["numeric_features"]),
            ("text", text_transformer, config["text_features"][0]),
            ("cat", categorical_transformer, config["categorical_features"],),
        ])

    return preprocessor


def build_model(config, preprocessor):
    model_class = getattr(
        __import__("sklearn.linear_model", fromlist=[config["type"]]),
        config["type"],
    )
    model = model_class(**config["parameters"])
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])  # noqa E501

    return pipeline
