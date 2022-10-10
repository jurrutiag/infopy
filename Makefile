SHELL=/bin/bash
LINT_PATHS=src/ tests/ setup.py
MAX_LINE_LENGTH=100

tox:
	tox

# type:
#     pytype -j auto

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://lintlyci.github.io/Flake8Rules/
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics --max-line-length ${MAX_LINE_LENGTH}
	# exit-zero treats all errors as warnings.
	flake8 ${LINT_PATHS} --count --exit-zero --statistics --max-line-length ${MAX_LINE_LENGTH}

format:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black -l ${MAX_LINE_LENGTH} ${LINT_PATHS}

check-codestyle:
	# Sort imports
	isort --check ${LINT_PATHS}
	# Reformat using black
	black --check -l ${MAX_LINE_LENGTH} ${LINT_PATHS}

commit-checks: format lint # type before lint

# # PyPi package release
# release:
#     python setup.py sdist
#     python setup.py bdist_wheel
#     twine upload dist/*

# # Test PyPi package release
# test-release:
#     python setup.py sdist
#     python setup.py bdist_wheel
#     twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: clean lint format check-codestyle commit-checks
