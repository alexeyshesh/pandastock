.PHONY: test
test:
	PYTHONPATH=. pytest

.PHONY: flake
flake:
	flake8 pandastock

.PHONY: publish
publish:
	rm -rf ./dist
	python -m build
	twine upload dist/*
