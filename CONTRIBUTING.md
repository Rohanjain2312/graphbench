# Contributing to GraphBench

Thank you for your interest in contributing!

## Setup

```bash
git clone https://github.com/Rohanjain2312/graphbench.git
cd graphbench
poetry install
cp .env.example .env  # fill in your credentials
```

## Code Style

This project enforces:
- **Black** (line length 88) for formatting
- **isort** for import sorting
- **ruff** for linting

Run all checks:

```bash
poetry run black .
poetry run isort .
poetry run ruff check .
```

## Testing

```bash
poetry run pytest tests/ -v
poetry run pytest --cov=graphbench tests/
```

All PRs must pass CI before merging.

## Pull Request Guidelines

1. Fork the repo and create a feature branch from `main`
2. Write tests for any new functionality
3. Ensure `poetry run pytest` passes
4. Ensure `poetry run ruff check .` and `poetry run black --check .` pass
5. Update `CHANGELOG.md` with your changes
6. Open a PR with a clear description of what changed and why

## Reporting Issues

Use [GitHub Issues](https://github.com/Rohanjain2312/graphbench/issues).
Include: Python version, OS, error message, minimal reproducible example.

## License

By contributing, you agree your contributions will be licensed under the MIT License.
