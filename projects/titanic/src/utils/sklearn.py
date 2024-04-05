from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# from sklearn.feature_extraction.text import CountVectorizer


def build_preprocessor(config):
    nis = config["preprocessing"]["numeric_transformer"]["imputer"]["strategy"]
    cis = config["preprocessing"]["categorical_transformer"]["imputer"][
        "strategy"
    ]  # noqa: E501
    cif = config["preprocessing"]["categorical_transformer"]["imputer"][
        "fill_value"
    ]  # noqa: E501
    pn = config["preprocessing"]["numeric_features"]
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy=nis),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=cis,
                    fill_value=cif,
                ),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # text_transformer = Pipeline(
    #     steps=[
    #         ("imputer", SimpleImputer(strategy="most_frequent", fill_value="")), # noqa: E501
    #         ("vectorizer", CountVectorizer()),
    #     ]
    # )

    # text_transformer = Pipeline(steps=[
    #     ('vectorizer', CountVectorizer())
    # ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, pn),
            (
                "cat",
                categorical_transformer,
                config["preprocessing"]["categorical_features"],
            ),
            # ('text', text_transformer, config['preprocessing']['text_features']) # noqa: E501
        ]
    )

    return preprocessor
