from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()


api = KaggleApi()
api.authenticate()

api.competition_download_files("titanic", path="data/")
