# Verified 3,000-Model NN Dataset Submission

This package contains 3,000 selected, nonduplicate neural-network models and
their matching real evaluation statistics. All 3,000 models have one real measured epoch, and no epoch data was fabricated. Native NN Dataset files are under
`ab/nn/nn/<model_id>.py` and
`ab/nn/stat/train/img-classification_cifar-10_acc_<model_id>/1.json`. The
original model bundle and normalized statistics archive remain under `models/`
and `statistics/`. Each model has only `1.json` because real measurements for
epochs 2 through 5 were not available.

## Provenance

The primary source is the audited set of prior submission-ready results. The
selection is deterministic: source IDs were sorted and the first 3,000 were
selected. Statistics were extracted from recorded `eval_info.json` artifacts
and are not synthetic. `previous_results_audit.json` documents the audit, and
`selection_report.json` records selection and deduplication decisions.

## Required fields

Each native statistics record contains the measured `accuracy`, `batch`, `lr`,
`momentum`, `transform`, `uid`, `weight_decay`, all required hyperparameters,
all scheduler-specific parameters, epoch-related metrics, and the model
`code_hash`. No epoch data was fabricated: missing epochs were not copied,
estimated, or filled with synthetic values. `native_layout_report.json` lists
all 3,000 models as having fewer than five real epoch measurements. The model
source implements `Net`, `supported_hyperparameters`, `train_setup`, and
`learn`.

## Reproducibility

Use `manifest.json` as the model-to-statistics mapping and
`generation_config.json` as the generation configuration. The manifest contains
both native paths and normalized archive paths. The manifest
records architecture, scheduler, decay, all exact hyperparameters, source
evaluation path, and code hash for every selected model. The package is
reproducible for the recorded one-epoch experiment only; it does not claim
five-epoch results, and no additional evaluation is required to reproduce the
recorded package contents.

This is an additive isolated submission package. It does not modify canonical
NN Dataset models, statistics, databases, or historical NN-GPT artifacts.
