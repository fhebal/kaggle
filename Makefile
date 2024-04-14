update-requirements:
	python3 -m pip install -r requirements.txt
	pre-commit install

lint:
	pre-commit run --all-files

load:
	python3 -m src.load
	unzip -o data/*.zip -d data/
	rm data/*.zip

profile:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE variable is not set. Use make prep FILE=filename.csv"; \
	else \
		python3 -m src.profile $(FILE); \
	fi

model:
	python3 -m src.model

features:
	python3 -m src.features

eda:	
	python3 -m src.eda

mlfow:
	mlflow server --default-artifact-root ./mlartifacts/
	export MLFLOW_TRACKING_URI='http://127.0.0.1:5000'
	mlflow server --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000
