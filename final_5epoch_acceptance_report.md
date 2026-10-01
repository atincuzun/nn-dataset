# Final 5-Epoch Acceptance Candidate

- Eligible models: **1085**
- Exact measured records copied: **9398**
- Epoch coverage: every selected model has real epochs **1, 2, 3, 4, 5**
- Source hash matches: **1085/1085**
- Required hyperparameters present in every record: **['batch', 'lr', 'momentum', 'transform', 'weight_decay']**
- Statistics are copied only from recovered measured JSON records; no records were fabricated, copied across epochs, interpolated, or estimated.
- Historical source artifacts were not modified.
- Submission-attributable underscore LR model/statistics files are excluded; the candidate contains only hyphenated names.

## Trial-count distribution

| Trial records per model | Models |
|---:|---:|
| 5 | 502 |
| 6 | 2 |
| 8 | 3 |
| 9 | 1 |
| 10 | 369 |
| 11 | 3 |
| 12 | 4 |
| 13 | 1 |
| 14 | 3 |
| 15 | 184 |
| 17 | 2 |
| 19 | 1 |
| 20 | 9 |
| 24 | 1 |

## Required separate groups

- Complete 5-epoch models with fewer than 3 trials: **0** (`complete_5epoch_fewer_than_3_trials.json`)
- Complete 5-epoch models with at least 3 trials: **1085** (`complete_5epoch_at_least_3_trials.json`)
- Complete 5-epoch models with 10 trials: **369** (`complete_5epoch_10_trials.json`)
- Models with only epoch-1 data: **1401** (`models_with_only_epoch1.json`)
- Models with partial multi-epoch data: **506** (`models_with_partial_multi_epoch.json`)
- Models with partial or epoch-1 data combined: **1907** (`partial_or_epoch1_models.json`)

## Validation status

Importer and complete-repository test results:


## Post-normalization recheck

After normalization, a candidate-only import into a temporary SQLite database
completed successfully for all **1085** candidate directories and **9398**
records with zero import errors. The corrected model API audit also passed for
all **1085** model files. A subsequent full-repository test invocation was
interrupted during the repository-wide historical database import before the
test cases started; therefore this post-normalization check does not claim a
new full-suite result. The earlier complete-checkout result remains recorded
above, and the candidate-only importer result is the authoritative check for
this submission's statistics structure.

## Statistics structure correction

The candidate statistics were normalized with
`util/py/restructure_lr_scheduler_submission.py` before submission. Each
trial record now places `batch`, `lr`, `momentum`, `transform`,
`weight_decay`, and all scheduler hyperparameters at the top level. Measured
training diagnostics are stored only in the optional `train_stat` object.
The provenance fields `architecture`, `code_hash`, `model_id`, `scheduler`,
`source_id`, `statistics_source`, and the nested `hyperparameters` and
`epoch_metrics` objects are removed because they are not NN Dataset training
parameters. The epoch number is represented by the JSON filename, as expected
by the importer. Historical source files and unrelated students' statistics
are not modified.
