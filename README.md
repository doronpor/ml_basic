# Machine Learning Basics

[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Black Code Format Check](https://github.com/doronpor/ml_basic/actions/workflows/black.yml/badge.svg)](https://github.com/doronpor/ml_basic/actions/workflows/black.yml)
[![Python Tests](https://github.com/doronpor/ml_basic/actions/workflows/python-tests.yml/badge.svg)](https://github.com/doronpor/ml_basic/actions/workflows/python-tests.yml)
[![Python Linting](https://github.com/doronpor/ml_basic/actions/workflows/linting.yml/badge.svg)](https://github.com/doronpor/ml_basic/actions/workflows/linting.yml)

A collection of machine learning algorithms implemented from scratch.

## Installation

### Quick Install
The easiest way to set up the project is to run the installation script:

```bash
python install.py
```

This will:
1. Install all main requirements (`requirements.txt`)
2. Install development requirements (`requirements-dev.txt`)
3. Set up pre-commit hooks

### Manual Installation
If you prefer to install manually:

1. Install main requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Install development requirements (optional):
   ```bash
   pip install -r requirements-dev.txt
   ```

3. Set up pre-commit hooks (optional):
   ```bash
   pre-commit install
   ```

## Workflows

This project uses several GitHub Actions workflows for quality assurance:

1. **Black Formatting**: Ensures consistent code style
   - Runs on every push and PR
   - Uses Black formatter with line length of 88

2. **Python Tests**: Runs the test suite
   - Tests on Python 3.8 and 3.9
   - Includes code coverage reporting
   - Uploads coverage to Codecov

3. **Python Linting**: Multiple code quality checks
   - isort: Checks import sorting
   - flake8: Style guide enforcement
   - mypy: Static type checking
   - pylint: Code analysis
