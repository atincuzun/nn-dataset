# Contributing to nn-dataset (LEMUR)

Thanks for your interest in improving the LEMUR framework. This document explains
how to add models, datasets, metrics, and transformations, and how to propose code
changes.

By participating you agree to abide by our [Code of Conduct](CODE-OF-CONDUCT.md).

## Ways to contribute

- Report bugs and request features via GitHub Issues.
- Add neural-network models, dataset loaders, metrics, or transformations.
- Improve the training pipeline, API, database layer, or analysis tools.
- Contribute evaluated models and their statistics back to the archive.

## Development setup

```bash
git clone https://github.com/ABrain-One/nn-dataset
cd nn-dataset
python -m pip install --upgrade pip
pip install -e .
```

An NVIDIA GPU with CUDA is required for training; the database, API, and
architecture-analysis tools run on CPU.

## Adding a model

A model is a single Python file implementing the framework contract:

- a module-level `supported_hyperparameters()` function returning the tunable
  hyperparameter names;
- a `Net(nn.Module)` class whose constructor accepts `in_shape`, `out_shape`,
  `prm`, and `device`, and which implements `train_setup` and `learn`.

Place the file under `ab/nn/nn/`. For a model to be merged, include training
statistics from the documented review workflow (ten Optuna trials over five
epochs; three trials are accepted for very large models).

## Running the tests

```bash
pip install pytest
pytest -q test
```

## Pull-request process

1. Fork the repository and create a branch from `main`.
2. Keep changes focused and add or update tests where practical.
3. Make sure `python -m compileall -q ab` and the test suite pass.
4. Describe the motivation and the change in the pull-request description.
5. A maintainer will review and, once approved, merge the change.

## Reporting issues

Include the command you ran, the task/dataset/metric configuration, the expected
and observed behaviour, and the full traceback.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
