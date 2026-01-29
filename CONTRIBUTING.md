# Contributing to nERdy

Thank you for your interest in contributing to nERdy! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. Please be kind and considerate in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/nERdy.git
   cd nERdy
   ```
3. Add the original repository as upstream:
   ```bash
   git remote add upstream https://github.com/NanoscopyAI/nERdy.git
   ```
4. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Types of Contributions

We welcome contributions in the following areas:

- **Bug fixes**: Fix issues in the existing codebase
- **Documentation**: Improve README, docstrings, or add tutorials
- **New features**: Add new analysis methods or model architectures
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Examples**: Add example notebooks or scripts

### Before Contributing

1. Check existing [issues](https://github.com/NanoscopyAI/nERdy/issues) to avoid duplicating work
2. For major changes, open an issue first to discuss your proposal
3. Make sure your contribution aligns with the project's goals

## Development Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Conda or pip for package management

### Installation for Development

```bash
# Create a virtual environment
conda create -n nerdy-dev python=3.10
conda activate nerdy-dev

# Install the package in development mode
pip install -e ".[dev]"

# Or using pip with requirements
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 isort
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nERdy --cov=analysis --cov-report=html

# Run specific test file
pytest test/test_graph_metrics_plotter.py
```

## Coding Standards

### Code Style

We follow PEP 8 guidelines with the following tools:

- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting

```bash
# Format code
black .
isort .

# Check for issues
flake8 .
```

### Documentation

- All public functions and classes should have docstrings
- Use Google-style docstrings:

```python
def function_name(param1: type, param2: type) -> return_type:
    """Brief description of function.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ExceptionType: Description of when this exception is raised.

    Example:
        >>> function_name(value1, value2)
        expected_output
    """
```

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (Add, Fix, Update, Remove, etc.)
- Reference issues when applicable: `Fix #123: description`

Examples:
```
Add support for multi-channel input images
Fix memory leak in batch processing
Update documentation for inference module
Remove deprecated preprocessing function
```

## Pull Request Process

1. **Update your branch** with the latest changes from upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests** to ensure nothing is broken:
   ```bash
   pytest
   ```

3. **Format your code**:
   ```bash
   black .
   isort .
   ```

4. **Push your changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what was changed and why
   - Reference to any related issues
   - Screenshots/examples if applicable

6. **Address review feedback** promptly

### PR Checklist

Before submitting your PR, ensure:

- [ ] Code follows the project's style guidelines
- [ ] All tests pass
- [ ] New code is covered by tests (when applicable)
- [ ] Documentation is updated (when applicable)
- [ ] Commit messages are clear and descriptive

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment information**:
   - OS and version
   - Python version
   - PyTorch/CUDA version
   - Package versions (`pip freeze`)

2. **Steps to reproduce** the issue

3. **Expected behavior** vs **actual behavior**

4. **Error messages** or tracebacks (if any)

5. **Sample data** or code to reproduce (if possible)

### Feature Requests

For feature requests, please describe:

1. The problem you're trying to solve
2. Your proposed solution
3. Any alternatives you've considered
4. Whether you're willing to implement it

## Questions?

If you have questions about contributing, feel free to:

- Open an issue with the "question" label
- Contact the maintainers directly

Thank you for contributing to nERdy!
