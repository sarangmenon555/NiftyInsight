PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

check-sources:
	$(PYTHON) -m src.pipeline check-sources

demo:
	bash scripts/run_demo.sh

full:
	$(PYTHON) -m src.pipeline all --source download

daily:
	bash scripts/run_daily.sh

dashboard:
	streamlit run dashboard/app.py

test:
	$(PYTHON) -m pytest -q

docker-build:
	docker build -t niftyinsight .
