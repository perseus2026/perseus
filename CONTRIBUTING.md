# Contributing to Perseus

Thanks for your interest in the project!
We welcome any contribution – from fixing typos to adding new features.
This document explains how to set up your environment, prepare your changes, and submit them for review.

Before you start, please read the [Code of Conduct](CODE_OF_CONDUCT.md).

## Requirements

- Python `>=3.12,<3.13`
- [uv](https://docs.astral.sh/uv/) — dependency and environment manager
- `make` (optional, for convenience commands)

## Setting up the environment

```bash
# Install dependencies and pre-commit hooks
make install

# or manually
uv sync && uv run prek prepare-hooks
```

Copy the example environment file if it is needed to run the project:

```bash
cp .env.example .env
```

## Workflow

1. Create a branch off `main`:
   ```bash
   git checkout -b feature/short-description
   ```
2. Make your changes. Try to follow the style of the surrounding code.
3. Add or update tests if you change behavior.
4. Run linters and tests locally (see below).
5. Commit with a clear message and open a merge request / pull request.

## Linting and formatting

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Checks run through `prek`:

```bash
# Run all hooks across the whole repository
make lint

# or
uv run prek run -a
```

Hooks run automatically on commit. Make sure they pass before submitting your changes.

## Tests

```bash
# Fast tests with coverage (threshold — 80%)
make test

# or
uv run pytest -n auto --cov --cov-report=term-missing --cov-fail-under=80
```

Coverage must not drop below 80%. New code should generally come with tests.

## Submitting changes

- Keep changes focused: one MR/PR — one logical task.
- Describe what was changed and why in the description.
- Make sure CI passes (lint + tests).

## Bug reports and suggestions

If you found a bug or want to propose an improvement, open an issue with a clear title and:

- steps to reproduce (for bugs);
- expected and actual behavior;
- Python version and environment, if relevant.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

Thanks for contributing!
