.PHONY: install test compat gpu-test lint format build
install:
	pip install -e ".[dev,train]"
test:
	pytest tests/ -q -m "not gpu"
lint:
	ruff check .
format:
	ruff format .
build:
	python -m build && twine check dist/*
compat:
	value-steer-compat
gpu-test:
	@if [ -z "$$VALUE_STEER_TEST_MODEL" ]; then echo "set VALUE_STEER_TEST_MODEL=<small model>"; exit 1; fi
	pytest tests/test_gpu_behavioral.py tests/test_gpu_smoke.py -q -m gpu
