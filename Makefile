.PHONY: setup ingest registry features sample models analysis all

setup:
	python -m venv .venv && .venv/bin/pip install -r requirements.txt

ingest:    ; python -m src.ingest.run
registry:  ; python -m src.registry.run
features:  ; python -m src.features.run
sample:    ; python -m src.sampling.run
models:    ; python -m src.models.run
analysis:  ; python -m src.analysis.run

all: ingest registry features sample models analysis
