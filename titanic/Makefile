update-requirements:
	python3 -m pip install -r requirements.txt
	pre-commit install

test:
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