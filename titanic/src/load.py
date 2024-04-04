from dotenv import load_dotenv  # noqa: E402 isort:skip

load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402 isort:skip

api = KaggleApi()
api.authenticate()

api.competition_download_files("titanic", path="data/")
