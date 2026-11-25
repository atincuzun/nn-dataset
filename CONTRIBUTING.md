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

Training uses an NVIDIA GPU (CUDA) when one is available and otherwise falls
back to Apple MPS or the CPU, so the test suite and small models run on any
machine; large studies are practical only on a GPU. The database, API, and
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

The same tests run on every push and pull request through GitHub Actions
(`.github/workflows/tests.yml`). To run them locally:

```bash
pip install pytest
# fast suite (CPU only, no large downloads)
python -m pytest -q -o python_files="test_*.py test.py" test test-ext \
  --ignore=test-ext/test_utkface.py --ignore=test/test.py \
  --ignore=test/test_nonnull.py --ignore=test/val_split
# validation-split tests: one test per process, each from an empty directory
for t in $(python -m pytest --collect-only -q test/val_split | grep '::'); do
  (cd "$(mktemp -d)" && python -m pytest -q "$OLDPWD/$t")
done
```

`test/test.py` and `test/test_nonnull.py` download the full LEMUR database
from Hugging Face and take longer; run them before changes to the database or
API layers.

## Pull-request process

1. Fork the repository and create a branch from `main`.
2. Keep changes focused and add or update tests where practical.
3. Make sure `python -m compileall -q ab` and the test suite pass.
4. Describe the motivation and the change in the pull-request description.
5. A maintainer will review and, once approved, merge the change.

## Reporting issues

Report bugs at https://github.com/ABrain-One/nn-dataset/issues. Include the
command you ran, the task/dataset/metric configuration, the expected and
observed behaviour, and the full traceback.

## Getting help

Questions about installing or using the framework are welcome as GitHub
issues at https://github.com/ABrain-One/nn-dataset/issues. Please search
existing issues first and describe what you are trying to achieve.

## Governance

The project is maintained by the ABrain-One team at the Computer Vision Lab,
University of Würzburg. Maintainers review every pull request, and changes are
merged after maintainer approval.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
