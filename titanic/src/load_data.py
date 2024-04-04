import os
from dotenv import load_dotenv
load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi


# Initialize Kaggle API and authenticate
api = KaggleApi()
api.authenticate()

api.competition_download_files('titanic', path='data/')