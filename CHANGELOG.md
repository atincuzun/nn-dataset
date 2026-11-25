# Changelog

All notable changes to nn-dataset (LEMUR) are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Releases are tagged on GitHub and published
on PyPI as `nn-dataset`.

Each entry was checked against the release tags, merged pull requests, and commits.
Reasons for a change are given only where a pull request, commit, or code comment
states them. Bulk model and training-statistic additions are summarised as counts.
Dates are GitHub release dates, or tag dates for tags without a release.

## [Unreleased]

### Added
- Loaders: `imagenet16-120` (NAS-Bench-201 ImageNet16-120; downloads checksum-pinned
  files from Hugging Face on first use)
  [#356](https://github.com/ABrain-One/nn-dataset/pull/356),
  [#358](https://github.com/ABrain-One/nn-dataset/pull/358),
  [#359](https://github.com/ABrain-One/nn-dataset/pull/359); denoising loaders
  `awgn-color`, `denoise-awgn`, `sidd` and `sidd-bench` with RGB metrics `psnrrgb` and
  `ssimrgb` [#364](https://github.com/ABrain-One/nn-dataset/pull/364).
- `ab/nn/imp/prun_adaptive.py`: adaptive pruning pipeline.
  [#362](https://github.com/ABrain-One/nn-dataset/pull/362)
- JOSS paper (`paper/paper.md`, `paper/paper.bib`, overview figure) and a GitHub Actions
  workflow that builds the draft PDF.
- GitHub Actions test workflow: byte-compiles `ab` and `test`; runs the test suite on
  Python 3.10 to 3.13, with `test/val_split` one test per process; builds and checks the
  sdist and wheel; installs the package with all declared dependencies and imports the
  public API; checks the latest PyPI release; and runs the tests that need the full LEMUR
  database weekly and on demand. `.gitignore` no longer hides `.github/`.
- README: statement of need, requirements and the first-use database download (network
  access and disk space), a worked `data()`/`check_nn()` example, the remaining query
  functions of `ab.nn.api`, and corrected command examples and contents links;
  `CONTRIBUTING.md` explains how to add loaders, transformations and metrics.
- `CITATION.cff`, `CONTRIBUTING.md` (development setup, tests, pull requests, getting
  help, governance), this changelog, and a README section "Support and Contributing".
- Models: 8,136 new models in `ab/nn/nn` (45,457 in total), among them 5,036 `divpo-*`
  CIFAR-10 models [#365](https://github.com/ABrain-One/nn-dataset/pull/365), 1,056
  denoising models, 608 `nnweb-*` models
  [#375](https://github.com/ABrain-One/nn-dataset/pull/375) and 91 loss/optimizer
  variants of LEMUR models [#368](https://github.com/ABrain-One/nn-dataset/pull/368);
  8,135 new folders in `ab/nn/stat/train`.

### Changed
- Training holds out a validation set by default for `cifar-10`, `cifar-100`, `mnist`,
  `svhn` and `celeba-gender`, taken from the training split (e.g. CIFAR-10: 40,000
  training and 10,000 validation images instead of 50,000 training images). Validation
  accuracy selects the best epoch, drives the minimum-accuracy check, the save decision
  and the Optuna objective, and is stored as `accuracy_val`; `accuracy` is still the
  test accuracy. `--test_only` (CLI) or `test_only=True` (API) restores the previous
  behaviour. [#366](https://github.com/ABrain-One/nn-dataset/pull/366),
  [#376](https://github.com/ABrain-One/nn-dataset/pull/376)
- `utkface` and WikiText loaders return a held-out test set in the test slot; reported
  accuracies for these datasets are not comparable with earlier records
  [#367](https://github.com/ABrain-One/nn-dataset/pull/367). Loader `imdb_wiki` renamed
  to `imdb-wiki`.
- `check_nn()` and `Train.train_new()` return `(name, accuracy, accuracy_to_time)`: the
  code-quality score is gone with `ab/nn/util/CodeEval.py` and the `flake8`,
  `flake8-json`, `radon` and `pylint` requirements.
- `syncDB2HF.py` stops without a valid Hugging Face token, takes the version from the
  installed package (else the `version` file) and skips the upload when the two differ;
  READMEs renamed to `info_README.md` and `syncDB2HF_README.md`.
  [#360](https://github.com/ABrain-One/nn-dataset/pull/360)
- `psutil` declared in `requirements.txt`; `ab/nn/util/Train.py`, `Util.py` and
  `Inference.py` import it directly, and it was previously installed only as a
  dependency of `accelerate`.

## [2.2.13] - 2026-10-26

### Added
- Architecture identity: `ab/nn/util/ArchUID.py` fingerprints a model's architecture
  from its source code; `ab/nn/util/ArchUIDMerge.py` merges models with the same
  architecture and writes a JSON log.
  [#316](https://github.com/ABrain-One/nn-dataset/pull/316)
- Data API: `data()` gains `min_accuracy` (applied in SQL, so it composes with
  `max_rows`) [#323](https://github.com/ABrain-One/nn-dataset/pull/323),
  `include_layer_stats` [#324](https://github.com/ABrain-One/nn-dataset/pull/324) and
  `pair_seed` (reproducible `sql=JoinConf` pairing, shuffled by default); `run_data()`
  gains `duration` and `type` filters (`"tflite"` for mobile, `"pt"` for workstation
  PyTorch runs, now loaded into the `run` table)
  [#337](https://github.com/ABrain-One/nn-dataset/pull/337),
  [#345](https://github.com/ABrain-One/nn-dataset/pull/345).
- Layer statistics can be restricted to selected features via `ACTIVE_LAYER_FEATURES` in
  `ab/nn/util/LayerAnalysis.py` or the `NN_DATASET_LAYER_FEATURES` environment variable
  (default: all seven). [#352](https://github.com/ABrain-One/nn-dataset/pull/352)
- Models: 14,486 new models in `ab/nn/nn`, among them 8,297 `divpo-*` CIFAR-10 models
  [#355](https://github.com/ABrain-One/nn-dataset/pull/355), 2,828 denoising models,
  2,000 `ast-dimension-AirNet-*` variants with CIFAR-100 statistics
  [#344](https://github.com/ABrain-One/nn-dataset/pull/344) and 1,085 `lr-*`
  learning-rate-scheduler models
  [#335](https://github.com/ABrain-One/nn-dataset/pull/335); layer-wise learning-rate
  study records [#348](https://github.com/ABrain-One/nn-dataset/pull/348),
  [#349](https://github.com/ABrain-One/nn-dataset/pull/349), ImageNet-100 training
  statistics [#329](https://github.com/ABrain-One/nn-dataset/pull/329) and Android
  TFLite benchmarks [#330](https://github.com/ABrain-One/nn-dataset/pull/330),
  [#332](https://github.com/ABrain-One/nn-dataset/pull/332).

### Changed
- `uuid4()` returns the architecture certificate for model code, so `nn.id` identifies
  an architecture; `code_to_db` rejects a model whose architecture is already stored
  under another name and attaches its statistics to the stored model.
  [#325](https://github.com/ABrain-One/nn-dataset/pull/325)
- Dependencies: `torch==2.13.0`, `torchvision==0.28.0`, `transformers==5.17.0`,
  `diffusers==0.41.0`, `datasets>=4.8.5,<5.0.0`, `huggingface_hub>=1.5.0,<2.0`. New
  optional groups `nn-dataset[litert]`, `[tflite]` and `[tflite-run]`
  (`requirements-*.txt`); `litert-torch` moves from the core requirements to `[litert]`,
  which cannot be combined with `[tflite]` (the file comments give the reasons: pins
  that conflict with vLLM, no Windows wheels).
- `ResNet` and `MobileNetV2` decay the learning rate with a cosine schedule over `epoch`
  and accept an `augment` hyperparameter that applies `BatchTransform` batch
  augmentation. [#351](https://github.com/ABrain-One/nn-dataset/pull/351)
- UTKFace validation and test images use fixed preprocessing (resize to 224 x 224,
  ImageNet normalization) instead of the training augmentation.
  [#350](https://github.com/ABrain-One/nn-dataset/pull/350)
- BLIP-2 captioning: only `Blip2Fast` and `Blip2FastOpt` are kept (`Blip2Sota`,
  `Blip2FastSota`, `Blip2Improved` removed), shared code lives in
  `ab/nn/util/captioning`, the `cached_blip2fast_processor` transform is removed, and
  `Blip2Fast` loads a pinned GPT-2 tokenizer revision.
  [#326](https://github.com/ABrain-One/nn-dataset/pull/326),
  [#328](https://github.com/ABrain-One/nn-dataset/pull/328)
- ImageNet-100 TFLite run statistics moved into the shared
  `ab/nn/stat/run/tflite/<mode>` folders.
  [#354](https://github.com/ABrain-One/nn-dataset/pull/354)

### Removed
- 2,701 models that duplicated another model's architecture (merged onto 1,611 surviving
  models, which received their statistics)
  [#316](https://github.com/ABrain-One/nn-dataset/pull/316), and the transform
  `ab/nn/transform/base/norm_256.py`.

### Fixed
- The missing-`HF_TOKEN` warning is printed only when uploading, not on import;
  downloading the public database needs no token.
  [#338](https://github.com/ABrain-One/nn-dataset/pull/338)

## [2.2.12] - 2026-08-08

### Added
- `info.py`: read-only command-line query tool: list tasks, show models and best
  accuracy per task (`--task`) or per dataset (`--best-accuracy`), save model source
  code by name prefix (`--code`, `--out-dir`), export with `--csv` (`.xlsx` or CSV by
  file extension) or `--txt`; documented in `LEMUR_QUERY_README.md`.
  [#306](https://github.com/ABrain-One/nn-dataset/pull/306)
- `syncDB2HF.py` replaces `lemur_archival_pipeline.py`: ten phases, now including
  download of the database from Hugging Face when no local copy exists and ingestion of
  new statistics; local files are deleted only after verification and a successful
  upload; documented in `LEMUR_PIPELINE_README.md`.
  [#306](https://github.com/ABrain-One/nn-dataset/pull/306)
- Models: 3,153 new models in `ab/nn/nn`, among them 2,137 `edge-*` models with
  one-epoch CIFAR-10 statistics
  [#307](https://github.com/ABrain-One/nn-dataset/pull/307), 880 `FractalFusion-*`
  models [#295](https://github.com/ABrain-One/nn-dataset/pull/295),
  [#300](https://github.com/ABrain-One/nn-dataset/pull/300),
  [#303](https://github.com/ABrain-One/nn-dataset/pull/303), 132 denoising models
  [#302](https://github.com/ABrain-One/nn-dataset/pull/302) and `TitanV12`-`TitanV14`
  [#301](https://github.com/ABrain-One/nn-dataset/pull/301); 6,795 new folders in
  `ab/nn/stat/train`; inference statistics for 16,058 models
  [#289](https://github.com/ABrain-One/nn-dataset/pull/289).

### Changed
- Layer-analysis results are stored in the per-epoch training JSON again;
  `json_train_to_db()` loads them into `layer_stat` and `per_layer_stat` and skips
  malformed `layers` entries with a warning, and `save_stat()` no longer keeps
  `layer_stat` inside `prm`. [#308](https://github.com/ABrain-One/nn-dataset/pull/308)
- Inference skips a model only if its output exists for the same dataset.
  [#289](https://github.com/ABrain-One/nn-dataset/pull/289)
- `util/sh/run_layer_analysis.sh` rewritten: freezes the highest-numbered epoch JSON,
  retrains `records[0]` for 50 epochs, and backs up the SQLite database and restores the
  JSON before writing results back. The top-level copy of the script is removed.
- Tests moved: `test.py` to `test/test.py`, `test-nonnull.py` to `test/test_nonnull.py`,
  `test_utkface.py` to `test-ext/test_utkface.py`.

### Removed
- Transform `norm_160_flip` (added in 2.2.11).

### Fixed
- `Train.py` creates the output directory before writing `training_summary.json`.

## [2.2.11] - 2026-07-10

### Added
- `imagenet` loader: streams `ILSVRC/imagenet-1k` from Hugging Face and requires
  `HF_TOKEN`.
- `imagenet100` loader: downloads `ambityga/imagenet100` with `kagglehub` and merges the
  shards into one `ImageFolder` through symlinks.
- Transforms `complex_192` (super-resolution: random 192-pixel HR crop and the matching
  64-pixel LR crop for scale 3, with the same random flips and 90-degree rotations on
  both), `norm_160_flip` and `norm_256_224_flip`.
- `lemur_archival_pipeline.py`: archival and cleanup pipeline (inventory, dependency
  mapping, database verification with SHA-256, Excel audit, zstd compression, upload to
  Hugging Face, local deletion).
  [#273](https://github.com/ABrain-One/nn-dataset/pull/273)
- Models: 1,238 new models in `ab/nn/nn`, among them 1,214 `GenFractalNet-*`, 20
  denoising models [#286](https://github.com/ABrain-One/nn-dataset/pull/286) and
  `TitanV8`-`TitanV11` [#276](https://github.com/ABrain-One/nn-dataset/pull/276),
  [#279](https://github.com/ABrain-One/nn-dataset/pull/279),
  [#285](https://github.com/ABrain-One/nn-dataset/pull/285); inference statistics for
  1,969 models [#278](https://github.com/ABrain-One/nn-dataset/pull/278) and FP32
  PyTorch inference results [#277](https://github.com/ABrain-One/nn-dataset/pull/277).

### Changed
- Inference: `ab/nn/util/inference.py` merged into `ab/nn/util/Inference.py`
  (`run_test_drive()`); `python -m ab.nn.util.Inference` takes `--model-name`, `--task`,
  `--dataset`, `--batch`, `--limit` and `--skip` and skips models already processed.
  [#271](https://github.com/ABrain-One/nn-dataset/pull/271),
  [#278](https://github.com/ABrain-One/nn-dataset/pull/278)
- TFLite export in `ab/nn/imp/upload2HF.py` uses `litert-torch`, the successor of the
  deprecated `ai-edge-torch`, for all five conversion modes.
  [#281](https://github.com/ABrain-One/nn-dataset/pull/281)
- Dependencies: `torch==2.9.1`, `torchvision==0.24.1`, `litert-torch` (non-Windows)
  instead of `ai-edge-torch`; new `kagglehub`, `matplotlib` and `openpyxl`.
- Training JSON keeps `epoch_max` at the top level instead of inside `train_stat`,
  because existing code reads it there.
  [#282](https://github.com/ABrain-One/nn-dataset/pull/282)

### Removed
- `ab/nn/util/db/Query_CLI.py`.

## [2.2.10] - 2026-06-25

### Added
- `ab/nn/util/inference.py`: inference benchmark (CPU timing with `perf_counter`, GPU
  timing with CUDA events, 10 warm-up passes; `--batch`, `--limit`, `--skip`).
  [#268](https://github.com/ABrain-One/nn-dataset/pull/268)
- Models: 1,584 new models in `ab/nn/nn` (1,163 `GenFractalNet-*`, 420
  `rl-bb-struct1-v2-*`, `TitanV6`
  [#263](https://github.com/ABrain-One/nn-dataset/pull/263)) and 1,584 new folders in
  `ab/nn/stat/train`.

### Changed
- `-c/--config` can be repeated to run several pipeline configurations in one
  `ab.nn.train` invocation. [#269](https://github.com/ABrain-One/nn-dataset/pull/269)
- Layer-analysis JSON export restructured and `util/sh/run_layer_analysis.sh` switched
  to the multi-configuration form
  [#269](https://github.com/ABrain-One/nn-dataset/pull/269); the per-epoch helpers
  `_save_layer_analysis()` and `_save_layer_table()` are removed from `Train.py`.
- `save_stat()` inserts with `INSERT OR IGNORE`, so a statistic row that is already
  stored is skipped.

### Fixed
- The `train_stat` table is filled when the database is built from JSON;
  `json_train_to_db()` used to drop the nested `train_stat` object, leaving the table
  empty. [#262](https://github.com/ABrain-One/nn-dataset/pull/262)

## [2.2.9] - 2026-06-20

### Added
- `ab/nn/imp/upload2HF.py`: ONNX export with `--format onnx`
  (`ab/nn/imp/onnx_pipeline.py`, opset 18, re-evaluated on 1,000 test images with
  `onnxruntime`) [#219](https://github.com/ABrain-One/nn-dataset/pull/219), and TFLite
  modes `fp16`, `mixed` and `dynamic` besides `fp32` and `static`, resumable per mode
  [#258](https://github.com/ABrain-One/nn-dataset/pull/258).
- Layer analysis: `--layer_analysis` flag (every epoch for the first five, then every
  fifth), `ab/nn/util/LayerAnalysis.py`, tables `layer_stat` and `per_layer_stat`, and
  `layer_data()` in `ab.nn.api`.
  [#259](https://github.com/ABrain-One/nn-dataset/pull/259)
- `train_stat` table for per-epoch diagnostics (losses, gradient norm, resource use),
  grouped under `train_stat` in the per-epoch JSON;
  `util/py/restructure_train_stat_json.py` migrates older files.
  [#251](https://github.com/ABrain-One/nn-dataset/pull/251),
  [#260](https://github.com/ABrain-One/nn-dataset/pull/260)
- Age estimation: `MobileAgeNet`, `mae` metric (score `max(0, 1 - MAE/15)`),
  `ab/nn/imp/run_tflite_validation_age.py`
  [#199](https://github.com/ABrain-One/nn-dataset/pull/199), and conversion and
  mobile-testing scripts in `util/py/age/`.
- Transforms `Resize_ColorJit_Flip_Blur`, `norm_224`
  [#210](https://github.com/ABrain-One/nn-dataset/pull/210) and `sr`
  [#250](https://github.com/ABrain-One/nn-dataset/pull/250).
- Models: 5,404 new models in `ab/nn/nn`, among them 2,521 `GenFractalNet-*`, 1,295
  `MoE4*`, 938 layer-wise learning-rate variants (`llr-*`, `llr2-*`, `llr3-*`), 197
  `del-*` delta-NAS models [#204](https://github.com/ABrain-One/nn-dataset/pull/204),
  BLIP-2 captioning models [#188](https://github.com/ABrain-One/nn-dataset/pull/188),
  [#193](https://github.com/ABrain-One/nn-dataset/pull/193),
  [#203](https://github.com/ABrain-One/nn-dataset/pull/203),
  [#254](https://github.com/ABrain-One/nn-dataset/pull/254) and super-resolution and
  denoising models [#233](https://github.com/ABrain-One/nn-dataset/pull/233),
  [#246](https://github.com/ABrain-One/nn-dataset/pull/246),
  [#250](https://github.com/ABrain-One/nn-dataset/pull/250); 6,319 new folders in
  `ab/nn/stat/train`; Android TFLite benchmarks
  [#226](https://github.com/ABrain-One/nn-dataset/pull/226),
  [#258](https://github.com/ABrain-One/nn-dataset/pull/258).

### Changed
- Database writes are serialised across processes with a file lock, and SQLite waits up
  to 300 s for a lock, so concurrent workers no longer fail with "database is locked".
  [#208](https://github.com/ABrain-One/nn-dataset/pull/208)
- Similarity bands for `JoinConf.similarity_band`: `very_high`, `high`, `medium`, `low`,
  `very_low_near`, `very_low_far`; `very_low` is gone and `high` now covers 0.95-0.98.
- Dependencies: `torch>=2.4,<2.9`, `torchvision>=0.21,<0.24`, `ai-edge-torch` only on
  non-Windows platforms (its dependencies ship no Windows wheels); new `bitsandbytes`
  and `onnxruntime`.

### Fixed
- `--pretrained 1` loads the best saved checkpoint; before, the flag was parsed but no
  weights were loaded. [#238](https://github.com/ABrain-One/nn-dataset/pull/238)
- `pip install --user` installs no longer fail with `no such table: loader`.
  [#209](https://github.com/ABrain-One/nn-dataset/pull/209)

## [2.2.8] - 2026-03-26

### Added
- `ab/nn/imp/upload2HF.py`: converts models to FP32 and INT8 TFLite, checks accuracy on
  1,000 CIFAR-10 images and uploads the results to Hugging Face.
  [#160](https://github.com/ABrain-One/nn-dataset/pull/160),
  [#166](https://github.com/ABrain-One/nn-dataset/pull/166)
- `ab/nn/imp/prHF.py`: structured L1 layer-wise pruning with `torch_pruning` (ratio
  0.30). [#162](https://github.com/ABrain-One/nn-dataset/pull/162),
  [#165](https://github.com/ABrain-One/nn-dataset/pull/165),
  [#167](https://github.com/ABrain-One/nn-dataset/pull/167),
  [#174](https://github.com/ABrain-One/nn-dataset/pull/174)
- Database tables `tflite` and `prun`, and `precision_type`, `accuracy` and `transform`
  in `run` [#171](https://github.com/ABrain-One/nn-dataset/pull/171),
  [#172](https://github.com/ABrain-One/nn-dataset/pull/172),
  [#173](https://github.com/ABrain-One/nn-dataset/pull/173); in `ab.nn.api`,
  `run_data()` joins `run` and `tflite`, and `prun_data()` and `data_withnonnullvalue()`
  are new. [#182](https://github.com/ABrain-One/nn-dataset/pull/182),
  [#187](https://github.com/ABrain-One/nn-dataset/pull/187)
- `JoinConf` queries: SQL-only selection of several models (`num_joint_nns > 1`) and
  similarity mode `anchor_band_db_minhash` with `similarity_band` (`high`, `medium`,
  `low`, `very_low`), using the `nn_minhash` table and the `jaccard_blobs` SQLite
  function [#161](https://github.com/ABrain-One/nn-dataset/pull/161),
  [#164](https://github.com/ABrain-One/nn-dataset/pull/164); faster SQL read path
  [#170](https://github.com/ABrain-One/nn-dataset/pull/170),
  [#180](https://github.com/ABrain-One/nn-dataset/pull/180).
- Metrics `bleu1` to `bleu4`; transforms `ab/nn/transform/base/BatchTransform.py`
  (CutMix, MixUp or CutOut, switched by progress through the epoch)
  [#177](https://github.com/ABrain-One/nn-dataset/pull/177), `base/norm_256` and
  `norm_cc_256`.
- Models: 292 new models in `ab/nn/nn` (289 `rl-bb-test1-*`
  [#175](https://github.com/ABrain-One/nn-dataset/pull/175), 3 `MoE4-*`
  [#179](https://github.com/ABrain-One/nn-dataset/pull/179)).

### Changed
- `ab/nn/run.py` is a thin command-line wrapper around `run_inference()` in the new
  `ab/nn/util/Inference.py`.
- Dependencies: `torch==2.9.1`, `diffusers==0.33.0`; the package location is found with
  `importlib.metadata` instead of `pkg_resources`.

### Removed
- 2,613 `FractalNet-*` models with their 4,955 folders in `ab/nn/stat/train`, and the
  `lemur-v2` loader.

### Fixed
- Statistics JSON: NaN and infinite values are written as `null`, and files with bare
  `NaN`/`Infinity` tokens can be read; JSON import into the database avoids
  platform-specific decoding problems on Windows.
  [#170](https://github.com/ABrain-One/nn-dataset/pull/170)

## [2.2.7] - 2026-02-12

### Added
- `ab/nn/imp/quant8.py`: INT8 TFLite export of CIFAR-10 models (`convert_int8()`); accuracy
  is checked in a subprocess and Android latency is measured over `adb` on CPU, GPU and
  NNAPI. Requires `tensorflow`, which is not a declared dependency.
- Image captioning: `GIT-SOTA` and `Blip2Sota` models with `git_processor.py` and
  `blip2_processor.py` transforms, and the CIDEr metric `ab/nn/metric/cider.py` [#141](https://github.com/ABrain-One/nn-dataset/pull/141).
- `metric` accepts a comma-separated list (e.g. `bleu,meteor,cider`); the first entry is the
  primary metric and `Train.eval()` returns `(primary_accuracy, all_results)`.
- `nn_stat` table, filled from the per-model statistics in `ab/nn/stat/nn/` (which now include
  code-MinHash signatures); `api.data(include_nn_stats=True)`, `api.nn_stat_data()` [#140](https://github.com/ABrain-One/nn-dataset/pull/140)
  [#144](https://github.com/ABrain-One/nn-dataset/pull/144) [#148](https://github.com/ABrain-One/nn-dataset/pull/148).
- Architecture similarity: `nn_sim.jsonl.gz` index [#146](https://github.com/ABrain-One/nn-dataset/pull/146) and `nn_similarity` table, filled
  by `ab/nn/util/db/build_nn_similarity.py` so that queries need not compute it [#149](https://github.com/ABrain-One/nn-dataset/pull/149).
- Per-epoch system metrics in training statistics: CPU and GPU type, RAM and GPU memory (KB),
  CPU and RAM usage [#142](https://github.com/ABrain-One/nn-dataset/pull/142).
- Models: 1,322 `rl-bb-init-*` [#152](https://github.com/ABrain-One/nn-dataset/pull/152) and 963 `FractalNet--*` variants, with CIFAR-10
  statistics.

### Changed
- `ab/nn/run.py` evaluates with `Train.Train.eval` (its own `SimpleAccuracy` is removed),
  records durations in nanoseconds and writes
  `stat/run/<config>_<arch>-<timestamp>/windows_<hostname>.json`.
- Super-resolution: the `div2k` loader switches from x2 to x4 pairs (synthetic pairs when
  the data is absent), `sr_transforms.py` is added, and PSNR and SSIM are computed on the
  Y channel (ITU-R BT.601) [#145](https://github.com/ABrain-One/nn-dataset/pull/145).
- Repository helper scripts moved from `cmd/` to `util/` (e.g. `util/py/log_nn_stat.py`).

### Known issues
- `ab/nn/util/db/Query_CLI.py` passes `similarity_mode`, `anchor_nn` and `similarity_band`
  to `JoinConf`, which no longer accepts them (the curriculum logic in `Query.py` was
  reverted before this release), so the CLI fails with a `TypeError`.
- `ab.nn.util.db.Util.min_accuracy()` returns 0.0 for every dataset instead of the loader's
  `MINIMUM_ACCURACY`, so `ab.nn.train` resets its failed-attempt counter for any accuracy
  above 0.

## [2.2.6] - 2026-01-11

### Added
- Quantisation and TFLite scripts in `ab/nn/imp/`: `quantize_fx_static.py` (FX static INT8,
  fbgemm, CIFAR-10 calibration), `quantize_hybrid_fallback.py` (falls back to dynamic
  `qint8` when accuracy drops by more than 0.10) and `run_tflite_validation.py`
  (`ai_edge_torch` export, re-evaluation with `ai_edge_litert`).
- Loaders `denoise`, `div2k`, `imdb_wiki`, `utkface`; metrics `meteor`, `ssim`, `mai_psnr`
  and a `METRICS` registry in `ab/nn/metric/__init__.py` [#123](https://github.com/ABrain-One/nn-dataset/pull/123) [#128](https://github.com/ABrain-One/nn-dataset/pull/128) [#129](https://github.com/ABrain-One/nn-dataset/pull/129) [#130](https://github.com/ABrain-One/nn-dataset/pull/130) [#131](https://github.com/ABrain-One/nn-dataset/pull/131).
- Database from Hugging Face: when neither a local database nor `ab/nn/stat/` exists,
  `init_population()` downloads `ab.nn-<version>.zst` from `NN-Dataset/LEMUR_DB`.
- Per-model architecture statistics in `ab/nn/stat/nn/<model>.json`, written by
  `log_nn_stat()` during `ab.nn.train` and by `cmd/py/log_nn_stat.py` [#137](https://github.com/ABrain-One/nn-dataset/pull/137);
  `ab/nn/util/NNAnalysisSimilarity.py` for MinHash/LSH code similarity [#133](https://github.com/ABrain-One/nn-dataset/pull/133).
- `api.data(unique_nn=True)`: one row per model, the one with the highest accuracy.
- Per-epoch training metrics (`EpochMetrics`: losses, accuracies, learning rate, gradient
  norm, throughput) and `out/training_summary.json` with learning curves [#134](https://github.com/ABrain-One/nn-dataset/pull/134).
- Models `DeepSeek-Coder-V2-Lite`, `DenoiseUNet2` [#136](https://github.com/ABrain-One/nn-dataset/pull/136) and `rlfn`; transforms
  `denoise` and seven `bf-v1-*_256` [#135](https://github.com/ABrain-One/nn-dataset/pull/135).
- Models: 2,861 `unq-*` variants with CIFAR-10 statistics (2,316 in one commit, 545 in
  [#127](https://github.com/ABrain-One/nn-dataset/pull/127)) and 874 `rl-back-init-*` variants with CIFAR-10 statistics [#139](https://github.com/ABrain-One/nn-dataset/pull/139).

### Changed
- PSNR metric `psnr` is clipped to [0, 48] dB and scaled to [0, 1] (0.5 for empty input);
  in 2.1.3 it returned dB.
- `run` table: 24 new columns (CPU/GPU/NPU timing statistics, RAM, input dimensions), also
  returned by `api.run_data()` [#138](https://github.com/ABrain-One/nn-dataset/pull/138).
- Packaging: only `ab` and `ab.*` packages are installed; the `hf` extra is removed;
  `ai-edge-torch`, `einops`, `datasketch` and `zstandard` are new dependencies.

### Removed
- `ab/nn/imp/auto_train.py`.

## [2.1.3] - 2025-12-11

### Added
- `ab/nn/run.py`: inference profiler (`run_inference()`) that builds a model from a dotted
  class path, scores it, samples `nvidia-smi` and process memory and writes `report.json`
  under `ab/nn/stat/run/<name>-<timestamp>/`.
- `ab/nn/imp/auto_train.py`: retrains models with their best stored hyperparameters and
  writes `metadata.json`; batch size 32 and `num_workers=0` (code comments: avoid OOM).
- Image denoising: `lemur-v2` loader (downloads and splits the data on first use), `psnr`
  metric, `DenoiseUNet` model and `resize-lemur-256` transform [#126](https://github.com/ABrain-One/nn-dataset/pull/126).
- `ab/nn/util/NNAnalysis.py`: static architecture descriptors (parameters, depth, layer
  statistics, FLOPs estimate) in `ab/nn/stat/nn.json` [#121](https://github.com/ABrain-One/nn-dataset/pull/121).
- Image captioning: `C10C-RESNETLSTM-IMG-CAP-IMPROVED` with `demo/img-caption-single.py` and
  `demo/infer_alexnet_caps.py` [#118](https://github.com/ABrain-One/nn-dataset/pull/118); ten `C5C-RESNETLSTM-*` variants [#119](https://github.com/ABrain-One/nn-dataset/pull/119);
  `cmd/py/infer_caption.py`.
- `cmd/py/simple_stat_viz/`: accuracy-statistics plots [#122](https://github.com/ABrain-One/nn-dataset/pull/122).
- `hf` optional dependency group (`nn-hf`).

### Changed
- `ab.nn.train.main()`: parameter `n_epochs` renamed to `epoch_max`, which is also stored in
  the hyperparameters as `epoch_max`; the CLI flag `-e/--epochs` is unchanged.
- `Util.accuracy_to_time_metric()`: parameter `training_duration` renamed to `duration`.
- `optuna_objective()` re-raises the original training exception instead of `NNException`.

### Removed
- `NNException`; the root-level `run_all_models.sh`.

## [2.1.2] - 2025-11-15

### Added
- Transforms: 155 `bf-v1-*` augmentation pipelines in `ab/nn/transform/` [#115](https://github.com/ABrain-One/nn-dataset/pull/115).
- `cmd/py/` auxiliary scripts `add_prm.py`, `uid_gen.py` and `duration_checker.py`,
  described in `cmd/README.md`.
- `evaluate` dependency.
- Models: 22 `rl-init1-*` variants with CIFAR-10 statistics.

### Changed
- Licences moved from `Doc/Licenses/` to `doc/licenses/` and `sh/rename-dir.sh` to
  `cmd/sh/`; package data follows (`'ab' = ['**/*.json']`, `'doc' = ['licenses/*']`).

### Removed
- Statistics: the 1,015 `0.json` files of the `ga-*` models [#112](https://github.com/ABrain-One/nn-dataset/pull/112); records without a
  `duration` removed from 17,675 statistics files [#113](https://github.com/ABrain-One/nn-dataset/pull/113).

## [2.1.1] - 2025-10-27

### Added
- `run` table for on-device runtime reports imported from `ab/nn/stat/run/`
  (`json_run_to_db()`), indexed by model and device; `api.run_data()` [#108](https://github.com/ABrain-One/nn-dataset/pull/108).
- Paired-model queries: `api.data(sql=JoinConf(...))` joins each row with another model that
  matches `same_columns`/`diff_columns` (optionally with higher accuracy, `enhance_nn`);
  the partner is returned as `nn_2`, `accuracy_2`, `prm_2`, etc.
- `api.data(nn_prefixes=...)` filters models by name prefix.
- `--save_onnx_weights` flag: `Util.save_if_best` also exports the best model to ONNX [#110](https://github.com/ABrain-One/nn-dataset/pull/110).
- `MoE-hetero4-Alex-Dense-Air-Bag` heterogeneous mixture-of-experts model [#109](https://github.com/ABrain-One/nn-dataset/pull/109);
  `demo/Text-ImageUnet.py` with HTML page [#110](https://github.com/ABrain-One/nn-dataset/pull/110).

### Changed
- `api.data(sql=...)` takes a `JoinConf` instead of the SQL string introduced in 2.1.0.
- `json_n_code_to_db()` replaced by `json_train_to_db()` and `json_run_to_db()`.
- Model `Unet-D` renamed to `UnetD`.

### Removed
- `ab/nn/metric/utils/` (`flatten_logits_and_labels`).

## [2.1.0] - 2025-10-14

### Added
- `api.data(sql=...)`: SQL appended to an `all_data` CTE (default `SELECT * FROM all_data`).
- `--save_pth_weights` flag (off by default) to save PyTorch weights through `save_if_best`.
- Demos `demo/Text-Image-GAN.py` (FastAPI, `ConditionalGAN` weights from Hugging Face
  `NN-Dataset/ConditionalGAN-checkpoints`) and `demo/Text-ImageCVAE.py`, with HTML pages
  [#104](https://github.com/ABrain-One/nn-dataset/pull/104).
- Models `ConditionalVAE2` to `ConditionalVAE4` [#93](https://github.com/ABrain-One/nn-dataset/pull/93) and four `MoE-0707-*`
  mixture-of-experts variants [#94](https://github.com/ABrain-One/nn-dataset/pull/94).
- On-device (Android) runtime reports for 2,476 model runs under `ab/nn/stat/run/` [#105](https://github.com/ABrain-One/nn-dataset/pull/105).
- Models: 2,712 `ga-mut-*` [#106](https://github.com/ABrain-One/nn-dataset/pull/106), 964 `FractalNet-*` and 101 `rl-init-*` [#99](https://github.com/ABrain-One/nn-dataset/pull/99) variants
  with CIFAR-10 statistics; 647 generated `alt-nn<N>-*` variants (N = number of supporting
  models in the generation prompt) with image-classification statistics [#90](https://github.com/ABrain-One/nn-dataset/pull/90) [#97](https://github.com/ABrain-One/nn-dataset/pull/97) [#101](https://github.com/ABrain-One/nn-dataset/pull/101)
  [#102](https://github.com/ABrain-One/nn-dataset/pull/102) [#103](https://github.com/ABrain-One/nn-dataset/pull/103).

### Changed
- Training statistics moved from `ab/nn/stat/` to `ab/nn/stat/train/` (`stat_train_dir`)
  [#105](https://github.com/ABrain-One/nn-dataset/pull/105).
- The 1,127 `rl-init-*` models of 2.0.6 renamed to `alt-<hash>` [#99](https://github.com/ABrain-One/nn-dataset/pull/99).
- COCO text-to-image loader takes only the image size from the framework's transform and
  applies its own `Resize`, `CenterCrop`, `ToTensor`, `Normalize` pipeline (square images).

### Removed
- `api.accuracy_to_time_metric()` (`ab.nn.util.Util.accuracy_to_time_metric` remains).
- `demo/app.py` and `demo/index.html`, replaced by `demo/Text-Image-GAN.py` and its page.

### Known issues
- The evaluation loop in `ab/nn/util/Train.py` again moves labels to the device
  unconditionally, undoing the 2.0.5 fix for non-tensor labels.

## [2.0.6] - 2025-10-03

### Added
- `demo/app.py`: FastAPI text-to-image demo (`POST /generate_image` runs the `ConditionalGAN`
  generator on a CLIP-tokenised prompt), with `demo/index.html` [#86](https://github.com/ABrain-One/nn-dataset/pull/86).
- Mixture-of-experts models with top-2 expert gating: `MoEv2` to `MoEv8` [#84](https://github.com/ABrain-One/nn-dataset/pull/84),
  `MoEv9-AlexNet` [#87](https://github.com/ABrain-One/nn-dataset/pull/87), `MoEv9-AlexNetv1` to `v4` [#89](https://github.com/ABrain-One/nn-dataset/pull/89).
- Models: 685 `FractalNet-*` and 494 `rag-*` [#82](https://github.com/ABrain-One/nn-dataset/pull/82) variants with CIFAR-10 statistics; 1,127
  `rl-init-*` [#79](https://github.com/ABrain-One/nn-dataset/pull/79) and 901 `alt-nn<N>-*` [#78](https://github.com/ABrain-One/nn-dataset/pull/78) [#80](https://github.com/ABrain-One/nn-dataset/pull/80) [#81](https://github.com/ABrain-One/nn-dataset/pull/81) [#83](https://github.com/ABrain-One/nn-dataset/pull/83) [#88](https://github.com/ABrain-One/nn-dataset/pull/88) variants with
  image-classification statistics on several datasets.

### Changed
- `Util.release_memory()` (garbage collection, CUDA cache release) runs after each trained
  model is deleted.
- The statistics import `json_n_code_to_db()` shows a progress bar.

## [2.0.5] - 2025-09-26

### Added
- `api.check_nn(transform_dir=...)`: loads the transform from `<transform_dir>/<name>.py`,
  falling back to the built-in transform when the file is absent [#67](https://github.com/ABrain-One/nn-dataset/pull/67).
- `run_all_models.sh`: runs `python -m ab.nn.train` for each `ast*.py` model in a hard-coded
  `~/Desktop/nn-dataset/ab/nn/nn`, logging to `mutation_tracking_logs/` [#71](https://github.com/ABrain-One/nn-dataset/pull/71).
- Text-to-image models `ConditionalGAN` [#69](https://github.com/ABrain-One/nn-dataset/pull/69) [#70](https://github.com/ABrain-One/nn-dataset/pull/70) and `Unet-D` [#76](https://github.com/ABrain-One/nn-dataset/pull/76).
- Ten captioning models generated with DeepSeek-R1-0528-Qwen3-8B (`C5C-`, `C8C-`, `C10C-`
  `RESNETLSTM` and `ResNetTransformer` variants) [#66](https://github.com/ABrain-One/nn-dataset/pull/66) [#75](https://github.com/ABrain-One/nn-dataset/pull/75).
- Models: 1,166 `ast-*` AlexNet variants (1,036 with CIFAR-10 statistics) [#71](https://github.com/ABrain-One/nn-dataset/pull/71), 227
  `rag-*` with CIFAR-10 statistics [#72](https://github.com/ABrain-One/nn-dataset/pull/72), 199 `alt-*` with image-classification statistics
  [#74](https://github.com/ABrain-One/nn-dataset/pull/74) [#77](https://github.com/ABrain-One/nn-dataset/pull/77).

### Changed
- COCO text-to-image loader uses the transform passed by the framework and reports the
  output shape from the transform and `vocab_size` as input shape.
- CLIP metric score is `logits_per_image.diag()` divided by 100.
- `setuptools` is no longer pinned below 80.

### Removed
- `C1C-RESNETLSTM` model (added in 2.0.4).

### Fixed
- Evaluation moves labels to the device only when they are tensors (code comment: fixes an
  image-generation error).
- CLIP metric tolerates malformed or empty batches; COCO image-size detection falls back
  with a warning.

## [2.0.4] - 2025-08-26

### Added
- `-p/--nn_prm` CLI option: JSON object of fixed hyperparameters, for reproducing results
  (README "Reproducing Results with Fixed Training Parameters"); others are still searched.
- `C1C-RESNETLSTM` captioning model [#65](https://github.com/ABrain-One/nn-dataset/pull/65).
- Models: 51 `rag-*` variants with CIFAR-10 statistics [#64](https://github.com/ABrain-One/nn-dataset/pull/64).

## [2.0.3] - 2025-08-20

### Added
- CLIP metric `ab/nn/metric/clip.py` (`openai/clip-vit-base-patch32`, cached per device)
  and `ConditionalVAE` text-to-image model (VGG16 perceptual loss, `kld_warmup_epochs`)
  [#63](https://github.com/ABrain-One/nn-dataset/pull/63).
- Models: 20 `rag-*` variants with CIFAR-10 statistics [#62](https://github.com/ABrain-One/nn-dataset/pull/62).

### Changed
- `ConditionalDiffusion`: hyperparameters `lr`, `beta1`, `beta2`, `steps_per_epoch`; mixed
  precision, xformers attention and a frozen VAE; its standalone inference CLI is removed
  [#63](https://github.com/ABrain-One/nn-dataset/pull/63).
- COCO text-to-image loader restricted to `TARGET_CATEGORIES = ['car']`; resize then
  centre-crop; unreadable images are skipped with a warning.
- New dependencies: `transformers`, `accelerate`, `diffusers`, `ftfy`, `scikit-learn`.
- The 1,015 `ga-alexnet-*` models renamed to `ga-*`.

## [2.0.2] - 2025-08-12

### Changed
- Packaging: JSON files are declared as `[tool.setuptools.data-files] '*' = ['**/*.json']`;
  `package-data` for `ab` lists only `*.md`.

## [2.0.1] - 2025-08-12

### Changed
- Packaging moved from `setup.py` to `pyproject.toml` (PEP 621); version and dependencies
  are read from `version` and `requirements.txt`.

### Removed
- `setup.py`, `MANIFEST.in` and the empty top-level `__init__.py`.

## [2.0.0] - 2025-08-11

### Added
- Text-to-image task `txt-image`: COCO loader `ab/nn/loader/coco_/Text2Image.py` (one
  random caption per image) and `ConditionalDiffusion` model [#58](https://github.com/ABrain-One/nn-dataset/pull/58) [#59](https://github.com/ABrain-One/nn-dataset/pull/59) [#60](https://github.com/ABrain-One/nn-dataset/pull/60).
- FID metric `ab/nn/metric/fid.py`, negated because the framework maximises the result
  (code comment) [#58](https://github.com/ABrain-One/nn-dataset/pull/58).
- Models: 1,015 `ga-alexnet-*` genetic-algorithm AlexNet variants with CIFAR-10
  statistics [#61](https://github.com/ABrain-One/nn-dataset/pull/61).

### Changed
- Perplexity metric is min-max normalised in log space to [0, 1] (`PPL_MIN = 1`,
  `PPL_MAX = 1000`); in 1.2.6 it returned raw perplexity.

### Fixed
- `Util.accuracy_to_time_metric()` and `good()` accept `None` for the accuracy or minimum
  accuracy (treated as 0.0).

## [1.2.6] - 2025-07-17

### Added
- `--train_missing_pipelines` flag: trains every task/dataset/metric/model combination built
  from the matched configurations that has no statistics yet.
- Perplexity metric `ppl` (`Perplexity`): exponential of the mean per-token cross-entropy,
  with optional `ignore_index`.
- Transforms `center_crop`, `five_crop`, `random_horizontal_flip`, `random_vertical_flip`
  (each resizes to 64x64) [#56](https://github.com/ABrain-One/nn-dataset/pull/56).
- Shell entry points: `train.sh` (activates `.venv` if present, then runs
  `python -m ab.nn.train "$@"`), `train-loop.sh` (restarts training endlessly) and
  `train-loop-50.sh` (image classification, 50 epochs, random order); README examples use `train.sh`.
- Models: `ConvNeXtTransformer` and `ResNetTransformer` for image captioning
  [#52](https://github.com/ABrain-One/nn-dataset/pull/52) and `MoE-0707`, a sparse
  mixture-of-experts model [#54](https://github.com/ABrain-One/nn-dataset/pull/54); statistics
  grow to 664 pipeline configurations and 14,791 per-epoch files (305 and 8,811 in 1.2.5).

### Changed
- `check_nn` returns `(name, accuracy, accuracy_to_time, code_score)` and decides whether to
  save by accuracy; since 1.0.0 its "accuracy" value was the accuracy-to-time value, as
  noted in [#37](https://github.com/ABrain-One/nn-dataset/pull/37). Optuna now also maximises
  accuracy instead of the accuracy-to-time value.
- Per-epoch time limit is set in minutes: `--epoch_limit_minutes` and
  `check_nn(epoch_limit_minutes=...)` (default 30) replace `check_nn(epoch_duration_limit_sec=...)`
  and `Const.max_epoch_seconds`.
- Loader attribute `minimum_accuracy` renamed to `MINIMUM_ACCURACY` in all loaders (the caption
  loader's `minimum_bleu` too); custom loaders must use the new name.
- WikiText loader returns output shape `(vocab_size,)` (was `None`) and builds 128-character
  windows at stride 1 with `torch.long` inputs and targets.
- Database: code tables get an `id` column (MD5 of the code) and `data()` also returns `nn_id`,
  `metric_id`, `transform_id` [#51](https://github.com/ABrain-One/nn-dataset/pull/51); the
  hyperparameter `uid` is a hash of the hyperparameters only (no `accuracy`, `duration`, `epoch`),
  and uids in the shipped statistics are replaced by such hashes
  [#53](https://github.com/ABrain-One/nn-dataset/pull/53). A `db/` from an older version has
  the old schema; the README install steps delete it.
- Image captioning: captions are tokenised with `nltk.word_tokenize` (the loader downloads
  `punkt_tab`), the vocabulary is shared through `GLOBAL_CAPTION_VOCAB`, and `RESNETLSTM`
  trains with mixed precision (`autocast`, `GradScaler`).

### Fixed
- `python -m ab.nn.train` passes `--min_dropout` and `--max_dropout` to `main()`; since 1.2.1
  they were missing from the positional call, which shifted `--transform` and every later flag.

### Removed
- `run.py` and `run.sh` (use `train.sh` or `python -m ab.nn.train`); `cast_prm` argument of
  `data()` and `check_nn()`; model `LSTMK.py` and the disabled `MoE-1.py_`, `MoE-2.py_`,
  `MoEHierarchical.py_`.

## [1.2.5] - 2025-06-06

### Changed
- `requirements.txt`: version floors dropped for `tqdm`, `optuna`, `numpy`, `pandas`, `scipy`
  and `pillow`; `setuptools<80.0.0` (was `<=80.0.0`). `torch>=2.6.0`, `torchvision>=0.21.0`,
  `datasets>=3.2.0` and `nltk>=3.9.1` keep their floors.

### Known issues
- `python -m ab.nn.train` passes flags to the wrong parameters from `--transform` onward (see
  1.2.3); fixed in 1.2.6.

## [1.2.4] - 2025-06-06

### Added
- `test_nn_train` in `test.py`: trains AlexNet on CIFAR-10 for one epoch through `train.main()`.

### Changed
- Names are content hashes: `Util.uuid4(obj)` is the MD5 of `str(obj)` without whitespace and
  replaces random UUIDs for code submitted via `check_nn`, code-table entries and statistic
  `uid`s [#46](https://github.com/ABrain-One/nn-dataset/pull/46).
- Task name `image-captioning` renamed to `img-captioning`.
- BLEU metric (`BLEUMetric`, no longer a `BaseMetric` subclass) computes BLEU-1 to BLEU-4 with
  NLTK smoothing, reports BLEU-4, and skips padding tokens and empty references.
- Caption loader: `minimum_bleu` lowered from 0.05 to 0.001; on-the-fly download of a missing
  COCO image uses a 10 s timeout [#45](https://github.com/ABrain-One/nn-dataset/pull/45),
  [#47](https://github.com/ABrain-One/nn-dataset/pull/47).
- `requirements.txt`: `~=` pins become `>=` floors; adds `nltk>=3.9.1`, `gdown` and
  `setuptools<=80.0.0`.

### Known issues
- `python -m ab.nn.train` passes flags to the wrong parameters from `--transform` onward (see
  1.2.3); fixed in 1.2.6.

## [1.2.3] - 2025-05-20

### Added
- Image captioning on COCO: `coco_/Caption.py` (`COCOCaptionDataset`, vocabulary with `<PAD>`,
  `<SOS>`, `<EOS>`, `<UNK>` cached in `data/coco/vocab.pth`), task `image-captioning`, BLEU
  metric `bleu.py`, and models `RESNETLSTM` and `LSTMK`
  [#44](https://github.com/ABrain-One/nn-dataset/pull/44).
- `python -m ab.nn.util.DeleteNN <model_name>`: deletes the model file, its statistics
  directories and the `db/` directory.

### Fixed
- Command-line training starts again: `Const` defines `default_min_dropout = 0.0` and
  `default_max_dropout = 0.5`.
- `check_nn` cleanup no longer raises `NameError` when it fails before the datasets are loaded.

### Removed
- 24 UUID-suffixed generated models and their 24 statistics directories (85 to 61 such models).
- `gdown` dependency (re-added in 1.2.4).

### Known issues
- `python -m ab.nn.train` calls `main()` positionally without `--min_dropout`/`--max_dropout`,
  so `--transform` and every later flag reach the wrong parameter; `run.py` passes them
  correctly. Fixed in 1.2.6.

## [1.2.2] - 2025-05-02

### Changed
- Hyperparameters are stored with native SQLite types in `prm(uid, name, value)`; the `type`
  column is gone and `data(cast_prm=...)` is accepted but has no effect
  [#43](https://github.com/ABrain-One/nn-dataset/pull/43). A `db/` from an older version has
  the old schema; the README install steps delete it.
- `data()` loads hyperparameters in bulk (chunks of 900 uids) using new `uid` indexes
  [#43](https://github.com/ABrain-One/nn-dataset/pull/43).
- `api.data()` is wrapped in `functools.lru_cache(maxsize=10)`: repeated calls with the same
  arguments return the cached `DataFrame`.
- `check_nn` deletes its datasets and model after each call and exports ONNX inside its
  error handling.
- `test.py` adds tests for native parameter types, a fetch-time budget, remaining trials,
  transforms, unique configurations and filters
  [#43](https://github.com/ABrain-One/nn-dataset/pull/43).
- README cites the LEMUR article (arXiv:2504.10552).

### Known issues
- Command-line training (`python -m ab.nn.train`, `run.py`) fails on import with `NameError`:
  `default_min_dropout` and `default_max_dropout` are not defined in `Const.py`. Fixed in 1.2.3.

## [1.2.1] - 2025-04-13

### Added
- `--min_dropout` and `-d/--max_dropout` flags set the dropout search range, previously fixed
  at 0.0 to 0.5; `validate_prm` rejects a minimum above the maximum.
- `data(max_rows=...)` limits the number of returned rows.
- `check_nn(epoch_duration_limit_sec=...)` overrides the per-epoch time limit (default 30 min).
- `onnx` dependency.

### Changed
- `check_nn` writes submitted code to `out/nn/tmp/<name>.py` under the project root (was the
  `ab.nn.tmp` package), deletes it afterwards, and always returns the generated name.
- ONNX export writes `out/onnx/nn.onnx` (directory created) and traces the model with one
  batch from the training set.
- `Train` takes the full module path of the network (`nn_module`, e.g. `ab.nn.nn.ResNet`);
  `Util.get_attr` no longer prepends `ab.nn` (`get_ab_nn_attr` does).

### Known issues
- Command-line training (`python -m ab.nn.train`, `run.py`) fails on import with `NameError`:
  `default_min_dropout` and `default_max_dropout` are not defined in `Const.py`. Fixed in 1.2.3.

## [1.2.0] - 2025-03-31

### Added
- `check_nn(save_path=...)`: writes the per-epoch statistics of submitted code to the given
  directory, which is created if missing [#37](https://github.com/ABrain-One/nn-dataset/pull/37).
- `check_nn(export_onnx=True)`: exports the trained model to ONNX (`Util.export_model_to_onnx`).
- `api.accuracy_to_time_metric(accuracy, training_duration, dataset)` and
  `Util.min_accuracy(dataset)`.
- Models: `Diffuser` [#38](https://github.com/ABrain-One/nn-dataset/pull/38) and 85 models
  generated or modified by NNGPT language models, named with UUID suffixes; #40 notes that some
  may be duplicates [#40](https://github.com/ABrain-One/nn-dataset/pull/40),
  [#41](https://github.com/ABrain-One/nn-dataset/pull/41). Statistics: 328 pipeline
  configurations and 8,854 per-epoch files (236 and 7,337 in 1.1.0).

### Changed
- `check_nn` returns the code-quality score scaled to [0, 1] (was 0 to 100).
- Loaders expose a public module attribute `minimum_accuracy` (was private
  `__minimum_accuracy`), read by `Util.min_accuracy`.
- `ab.nn.util.Loader`: the `Loader.load_dataset` static method is replaced by the functions
  `load_dataset()` and `get_obj()`.
- README explains the naming of NNGPT-generated models and installs PyTorch from the CUDA 12.6
  index (`cu126`).

## [1.1.0] - 2025-03-09

### Added
- `--pretrained {1,0}` flag: always or never use pretrained weights; when unset, Optuna chooses
  for models that list `pretrained` [#31](https://github.com/ABrain-One/nn-dataset/pull/31).
- Metric interface `BaseMetric` (`ab/nn/metric/base/base.py`: `reset`, `update`, `__call__`,
  `result`); `acc`, `iou` and `map` implement it and each metric module provides
  `create_metric(out_shape)`, replacing the per-metric `compute()`
  [#35](https://github.com/ABrain-One/nn-dataset/pull/35).
- `check_nn(prefix=...)`: the saved model is named `<prefix>-<uuid>`
  [#36](https://github.com/ABrain-One/nn-dataset/pull/36).
- `check_nn` returns `(name, accuracy, code_score)` with the `CodeEval` score
  [#27](https://github.com/ABrain-One/nn-dataset/pull/27).
- `version` file as the single version source, read by `setup.py` and shipped via `MANIFEST.in`.
- Models: `FractalNet` [#30](https://github.com/ABrain-One/nn-dataset/pull/30); classification
  statistics for SVHN, Places365 and CelebA-gender
  [#29](https://github.com/ABrain-One/nn-dataset/pull/29). Statistics: 236 pipeline
  configurations and 7,337 per-epoch files (145 and 3,830 in 1.0.4).
- README: `lmur` short alias on PyPI; citation retitled "LEMUR Neural Network Dataset: Towards
  Seamless AutoML" with additional authors.

### Changed
- `check_nn` imports submitted code as a temporary module of the `ab.nn.tmp` package with
  `importlib` instead of `exec` [#27](https://github.com/ABrain-One/nn-dataset/pull/27),
  [#28](https://github.com/ABrain-One/nn-dataset/pull/28).

### Fixed
- `check_nn(save_to_db=True)`: `train_n_eval` no longer writes statistics files for submitted
  code; `train_new` saves the result itself [#36](https://github.com/ABrain-One/nn-dataset/pull/36).
- Detection networks (`FCOS`, `FasterRCNN`, `RetinaNet`, `SSDFull`) set their trainable layers
  according to `pretrained`, and layer freezing is fixed for non-pretrained models
  [#35](https://github.com/ABrain-One/nn-dataset/pull/35).

## [1.0.4] - 2025-02-09

### Changed
- `ab` is an extendable namespace package (`pkgutil.extend_path` in `ab/__init__.py`), so
  `ab.*` packages from other installed distributions can be imported alongside it.
- Distribution name in `setup.py` is `nn-dataset` again (1.0.3 used `NN-Dataset`).
- README installs `nn-stat` from GitHub and runs the export with `python -m ab.plot.export`.
- Statistics: 3,830 per-epoch files (3,695 in 1.0.3).

## [1.0.3] - 2025-02-08

### Added
- `data(cast_prm=True)`: with `False`, hyperparameter values are returned as stored instead of
  being cast to their recorded type.
- `stat` extra (`pip install nn-dataset[stat]` installs `nn-stat`), replacing the `nn-plots`
  extra; README describes `nn-stat` for Excel export and plots.

### Changed
- PyTorch requirement raised to `torch~=2.6.0`, `torchvision~=0.21.0`.
- Project root is found by walking up from the current directory to a folder containing `ab/`
  and `README.md`; the root is printed on import.
- Package metadata: name `NN-Dataset`, homepage `https://ABrain.one`.
- License files renamed without the `.md` extension (`LICENSE`, `Doc/Licenses/LICENSE-*`).
- Statistics: 3,695 per-epoch files (3,335 in 1.0.2).

## [1.0.2] - 2025-02-06

### Added
- README: PyPI badge and PyPI install instructions, including the optional `nn-plots` extra
  (`pip install nn-dataset[nn-plots]`); license links point to absolute GitHub URLs.
- Source distributions include `LICENSE.md` and `Doc/Licenses/*.md` (`MANIFEST.in`).

### Removed
- `pyproject.toml`; packaging is defined in `setup.py` only.

## [1.0.1] - 2025-02-05

### Changed
- `setup.py` no longer runs `pip install torch~=2.5.1` from the CUDA 12.4 index during
  installation (`install_pytorch()` removed); the `[tool.pip_install]` torch pin in
  `pyproject.toml` is removed too.

### Fixed
- `ICNet` read the learning rate from `prm['ls']`; it now uses `prm['lr']`.

## [1.0.0] - 2025-02-05

### Added
- Python API `ab.nn.api`: `data()` returns models, code and statistics as a pandas `DataFrame`,
  filterable by `task`, `dataset`, `metric`, `nn`, `epoch` and `only_best_accuracy`
  [#26](https://github.com/ABrain-One/nn-dataset/pull/26); `check_nn(nn_code, task, dataset,
  metric, prm, save_to_db=True)` trains a model from source code and stores it if it passes.
- SQLite database `db/ab.nn.db` (`ab/nn/util/db/`), built on first use from the JSON
  statistics in `ab/nn/stat/` [#10](https://github.com/ABrain-One/nn-dataset/pull/10),
  [#12](https://github.com/ABrain-One/nn-dataset/pull/12).
- `ab/nn/util/CodeEval.py`: checks model source with Pylint, Radon complexity and a smoke run,
  giving a 0 to 100 score; used by `check_nn`.
- Object detection on COCO with an mAP metric (`metric/map.py`)
  [#24](https://github.com/ABrain-One/nn-dataset/pull/24); classification loaders for
  CIFAR-100, MNIST [#23](https://github.com/ABrain-One/nn-dataset/pull/23), SVHN, ImageNette,
  Places365 and CelebA-gender; COCO split into `coco_/Detection.py` and `coco_/Segmentation.py`,
  which caches its filtered image list [#16](https://github.com/ABrain-One/nn-dataset/pull/16).
- Transforms: `norm`, `complex` and `echo` families at sizes 32 to 512 with `_flip` variants,
  plus `soft_pad` (43 in total); Optuna picks the transform unless `-f/--transform` fixes it.
- CLI flags for search ranges and robustness: `--min_batch_binary_power` and
  `-b/--max_batch_binary_power`, `--min_learning_rate` and `-l/--max_learning_rate`,
  `--min_momentum` and `-m/--max_momentum`, `-a/--nn_fail_attempts`,
  `-r/--random_config_order`, `-w/--workers`.
- Models: 49 architectures in `ab/nn/nn/` (32 in 0.1.2), including AirNet, AirNext, BagNet,
  DarkNet, DPN68/107/131, ICNet [#18](https://github.com/ABrain-One/nn-dataset/pull/18) and
  the detectors FasterRCNN, RetinaNet, FCOS, SSDLite, SSDFull
  [#24](https://github.com/ABrain-One/nn-dataset/pull/24). Statistics: 145 pipeline
  configurations and 3,335 per-epoch files [#25](https://github.com/ABrain-One/nn-dataset/pull/25).

### Changed
- Model contract: `ab/nn/nn/<Name>.py` defines `Net(in_shape, out_shape, prm, device)` with
  `train_setup(prm)` and `learn(train_data)`, plus `supported_hyperparameters()`; Optuna
  searches each listed hyperparameter and the batch size. The README introduces the LEMUR
  name and documents the contract.
- Configuration names use `_` and drop the transform (`img-classification_cifar-10_acc_AlexNet`,
  was `img_classification-cifar10-acc-cifar10_norm-AlexNet`); results are stored as
  `ab/nn/stat/<config>/<epoch>.json`.
- Defaults: one epoch per run (was `(1, 2, 5)`) and one additional trial (`-t -1`; negative
  values add trials).
- Training stops a trial whose accuracy is not above 1.2 times the loader's minimum accuracy or
  whose epoch is estimated to exceed 30 minutes; on CUDA out-of-memory the maximum batch size
  is lowered and the study restarts.

### Removed
- `ab/nn/dataset/` (models moved to `ab/nn/nn/`), `ab/nn/util/Stat.py` with the
  `trials.json`/`best_trial.json` layout, `loader/cifar10.py` (now `cifar-10.py`) and the
  `cifar10_*` transforms.

## [0.1.2] - 2024-12-22

### Added
- `run.py` reads `-c/--config`, `-e/--epochs`, `-t/--trials` and `-b/--max_batch_binary_power`
  from the command line.
- A full configuration name passed with `-c` (`task-dataset-metric-transform-model`) gets its
  statistics directory created automatically; the README no longer asks to create it by hand.

### Changed
- Paths are resolved automatically: statistics and models are found under `ab/nn/` in the
  current directory or in `.venv/Lib/site-packages`, and the CIFAR-10 and COCO loaders store
  data in `data/`.
- COCO loader signature is `loader(resize, **kwargs)`; the `class_list` and `path` parameters
  are removed.
- `setup.py` version set to `0.1.2` (the 0.1.1 tag still declared `0.1.0`).

### Removed
- `-r/--from_root` flag and `default_from_root`.
- `ab/nn/stat/README.md`.

## [0.1.1] - 2024-12-21

Initial release.

### Added
- Training command `python -m ab.nn.train` (or `run.py`): runs an Optuna search over `lr`,
  `momentum` and `batch_size` for every pipeline configuration under `ab/nn/stat/` named
  `task-dataset-metric-transform-model`, by default for 1, 2 and 5 epochs and 100 trials.
- Resumable runs: finished trials are counted from `trials.json`; a negative or `"+N"` trial
  count adds N trials. Results are kept as `trials.json` and `best_trial.json` per epoch count.
- Tasks: image classification on CIFAR-10, semantic segmentation on COCO and character-level
  text generation on WikiText; metrics `acc` and `iou`; transforms `cifar10_norm`,
  `cifar10_norm_32`, `cifar10_complex`, `echo`.
- Models: 32 architectures in `ab/nn/dataset/`, including segmentation networks
  [#2](https://github.com/ABrain-One/nn-dataset/pull/2), ComplexNet
  [#3](https://github.com/ABrain-One/nn-dataset/pull/3) and BayesianNet variants
  [#5](https://github.com/ABrain-One/nn-dataset/pull/5),
  [#9](https://github.com/ABrain-One/nn-dataset/pull/9). Statistics: 28 pipeline
  configurations (CIFAR-10 classification and COCO segmentation).
- Packaging: `setup.py` (Python 3.10 or newer, requirements including `torch~=2.5.1` and
  `optuna~=4.1.0`) and `MANIFEST.in` [#7](https://github.com/ABrain-One/nn-dataset/pull/7),
  [#8](https://github.com/ABrain-One/nn-dataset/pull/8); MIT `LICENSE.md` and third-party
  licenses in `Doc/Licenses/`.
