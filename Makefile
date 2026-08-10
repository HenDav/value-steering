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
	@# vLLM V1 retains the KV-cache reservation in-process (it lives outside torch's caching
	@# allocator), so running the whole suite in one process accumulates engines and OOMs later
	@# tests at init. Run each GPU test in its OWN process (process exit frees everything).
	@rc=0; \
	for t in $$(pytest tests/test_gpu_behavioral.py tests/test_gpu_smoke.py -m gpu --collect-only -q 2>/dev/null | grep '::'); do \
		echo "=== $$t ==="; \
		pytest "$$t" -q -m gpu -p no:cacheprovider || rc=1; \
	done; \
	exit $$rc
