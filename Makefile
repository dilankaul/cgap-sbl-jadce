TEST_ENV = OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src

.PHONY: test-fast test-gui test-integration test-plotting test-all test-profile

test-fast:
	$(TEST_ENV) pytest -q -m unit

test-gui:
	$(TEST_ENV) pytest -q -m gui

test-integration:
	$(TEST_ENV) pytest -q -m integration

test-plotting:
	$(TEST_ENV) pytest -q -m plotting

test-all:
	$(TEST_ENV) pytest -q -n 2 --dist worksteal

test-profile:
	$(TEST_ENV) pytest -q --durations=20
