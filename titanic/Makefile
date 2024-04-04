update-requirements:
	python3 -m pip install -r requirements.txt
	pre-commit install

load:
	python3 -m src.load
	unzip -o data/*.zip -d data/
	rm data/*.zip

prep:
	python3 -m src.prep

model:
	python3 src.model

profile:
	python3 src.profile

pipeline:
	python3 -m src.prep
	python3 src.model
	python3 src.profile
