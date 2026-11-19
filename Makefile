# Convenience commands (macOS/Linux, or Windows with Git Bash + make).
# On Windows PowerShell, use the equivalent commands listed in the README.

PY := .venv/bin/python
ifeq ($(OS),Windows_NT)
    PY := .venv/Scripts/python.exe
endif

.PHONY: setup data train evaluate explain business report pipeline test serve mlflow eda

setup:      ## create venv and install dependencies
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:       ## download the raw dataset
	$(PY) -m src.data

train:      ## train models + log to MLflow + save best pipeline
	$(PY) -m src.train

evaluate:   ## choose threshold on validation, score test set once
	$(PY) -m src.evaluate

explain:    ## SHAP plots
	$(PY) -m src.explain

business:   ## cost/benefit simulation
	$(PY) -m src.business

report:     ## build reports/churn_report.pdf
	$(PY) -m src.report

pipeline: train evaluate explain business report   ## everything, in order

eda:        ## re-run the EDA notebook
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

test:
	$(PY) -m pytest -q

serve:
	$(PY) -m uvicorn api.main:app --reload

mlflow:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db
