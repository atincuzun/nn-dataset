# Changelog

All notable changes to nn-dataset (LEMUR) are documented in this file.

The project follows [Semantic Versioning](https://semver.org/) and releases are
tagged on GitHub and published on PyPI as `nn-dataset`. Entries are derived from
the code diff between consecutive release tags and explain why each change was
made; bulk model and training-statistic additions are summarised as counts.

## [Unreleased] (2.2.13)

### Added
- JOSS submission support: continuous integration, contribution guidelines, and
  this changelog.

## [2.2.12] - 2026-08-07

### Added
- Top-level `info.py`, a read-only query tool (`cmd_list_tasks`, `cmd_task_summary`, `cmd_extract_code`, `cmd_best_accuracy`), so users can explore and export the dataset without SQL; it imports `ab.nn.api.data` lazily and picks `.xlsx`/CSV/TXT by file extension.
- `phase2_ensure_db()` and `phase3_ingest_stats()` in `syncDB2HF.py`, so archival runs on a fresh checkout: the DB is fetched from Hugging Face when absent, and local stat JSON is ingested before verification, so newly trained runs verify instead of being held back by the deletion gate.
- `_layer_snapshot_to_table()`/`_save_layer_stat()` in `ab/nn/util/db/Write.py`, so the training-JSON path and historical ingest share one set of table-creation and insert statements.
- A self-contained `util/sh/run_layer_analysis.sh` that freezes the highest-numbered numeric JSON, replays `records[0]`'s exact hyperparameters/batch/transform and retrains for 50 epochs, so layer stats stay comparable to the archived run; it backs up SQLite and rolls back JSON before writeback because that update is expensive and destructive.
- `test.py` moved to `test/test.py`, plus new `test/test_nonnull.py` (gated by `RUN_SLOW_TESTS=1`) and `test-ext/test_utkface.py`, so slow and extended suites stay out of the default run, and `LEMUR_PIPELINE_README.md`/`LEMUR_QUERY_README.md`, so both entry points document their 10-phase flow and safety guarantees.

### Changed
- Layer stats now travel with the DB payload: `Train.py` builds a `layer_stat_group` under `prm['layer_stat']` and moves `epoch_max` top-level, so one prm row reconstructs the run; `layer_stat` is re-inserted after `train_stat` to keep the saved JSON readable.
- `Inference.py` treats an output dir as existing only if `model_name` and, when given, `args.dataset` match, so a cached run is not reused for another dataset sharing the same model.
- `checkpoints_to_hugging_face.py` uses `unique_nn_cls(...)` and `dt['prm']`, relying on the DB's canonical best-parameter rows instead of its own fallback search.

### Fixed
- `save_layer_stat` no longer collides across epochs (epoch in the row id plus `@_serialized_db_write`); `save_stat` pops `layer_stat` so it is not nested in prm.
- `json_train_to_db()` warns and skips non-list `layers` or nameless rows instead of aborting, and `syncDB2HF.py` verifies generated variants via `SELECT 1 FROM stat WHERE nn = ?` when `nn_code` is NULL.
- `Train.py` creates the summary directory before writing `training_summary.json`, so a missing output dir no longer drops the summary.

### Removed
- `ab/nn/transform/norm_160_flip.py`, `k8s/submit-imagenet100-checkpoints.sh`, the top-level `run_layer_analysis.sh`, and `test-nonnull.py`.

## [2.2.11] - 2026-07-10

### Added
- **ImageNet-1k loader** (`ab/nn/loader/imagenet.py`): streams `ILSVRC/imagenet-1k` via `datasets`, but only after checking `HF_TOKEN`. The token gate exists because the dataset is licence-gated, so failing immediately with setup instructions is far cheaper than a run that dies mid-download; streaming avoids shipping ~150 GB of images in the repo.
- **ImageNet-100 loader** (`ab/nn/loader/imagenet100.py`): downloads `ambityga/imagenet100` with `kagglehub` and symlink-merges the `train.X*` shards into one `ImageFolder`. The shards are merged with symlinks, not copies, so the loader adds no duplicate disk use, and 100 classes keep training cheap enough to fan out over the cluster while remaining ImageNet-like.
- **Super-resolution transform `complex_192`**: paired `RandomCrop(size=192, scale=3)` with flips and 90° rotation. The HR crop is derived from the LR crop at the same location (`hr = lr * scale`), because independently cropped pairs do not correspond and would train the model on unrelated pixels.
- **Classification transforms `norm_160_flip`, `norm_256_224_flip`**: two standard recipes (resized-crop 160; resize-256/center-crop-224) so ImageNet-100 runs have sensible defaults at both reduced and full resolution.
- **LEMUR archival/cleanup pipeline** (`lemur_archival_pipeline.py`, phases 0–7): inventories models/stats/transforms, maps their dependencies, and labels each SAFE or KEEP before touching anything.
- **Kubernetes checkpoint fan-out** `k8s/submit-imagenet100-checkpoints.sh`: one Job per classifier (38 models). Deletion in the pipeline is multi-gated — SHA-256 of `nn_code`/`transform_code`, `(task,dataset,nn)` lookups, Excel audit, zstd upload, and only then removal — because local stats and models are irreplaceable and cannot be recovered once deleted. Splitting training per model likewise isolates each run in its own GPU/memory budget so one failure is retryable rather than fatal to the batch.
- Deps: `kagglehub` (ImageNet-100 download), `matplotlib` and `openpyxl` (Excel audit output).

### Changed
- `Loader.get_obj` gains a `loader` branch that imports `ab.nn.loader.<name>` directly, so loaders are resolved as ordinary modules instead of through the DB attribute helper.
- **Inference consolidated into `run_test_drive`** (`ab/nn/util/Inference.py`): one argparse CLI doing warmup then 20 CPU/20 GPU timings and writing JSON under `stat_run_dir/pt/fp32`. Previously an uppercase `Inference.py` profiler and a lowercase `inference.py` benchmark duplicated each other; merging them removes the divergent copies and the case collision that breaks checkouts on case-insensitive (Windows) filesystems.
- `checkpoints_to_hugging_face.py` builds a per-model training plan with per-dataset defaults (`imagenet100` → `norm_160_flip`, batch 256). Defaults are used only when the DB lacks a complete parameter set, so every model still yields checkpoints, and `param_source` keeps DB-derived and defaulted runs distinguishable.

### Fixed
- Migrated the TFLite exporter from `ai-edge-torch` to `litert-torch` (its renamed successor) and pinned `torch==2.9.1`, `torchvision==0.24.1`. The converter builds against a specific torch ABI, so the old `>=2.4,<2.9` range could resolve to a version it cannot convert; exact pins keep export reproducible.

### Removed
- Standalone `ab/nn/util/inference.py` (folded into `Inference.py`); `ab/nn/util/db/Query_CLI.py`; reverted the Blip2FastOpt captioning-stat change.

## [2.2.10] - 2026-06-25

### Added
- **Inference benchmarking harness** (`ab/nn/util/inference.py`): training accuracy says nothing about how fast or how memory-hungry a network is on real hardware, so the harness fetches the best-accuracy row for a model directly from the DB and rebuilds it with the stored hyperparameters and transform. CPU passes are timed with `perf_counter`, but GPU passes must be timed with CUDA events, because kernel launches are asynchronous and wall-clock timing would measure the enqueue rather than the compute. Ten warm-up passes are discarded so allocator/JIT/clock ramp-up does not inflate the result, and mean/min/max/std-dev plus RAM and peak RSS/GPU MB are written as JSON under `stat/run/pt/fp32/...` keyed by device, so numbers from different machines can be compared. `--batch`/`--limit`/`--skip` exist so the whole ~15K-model database can be swept unattended.

### Changed
- **Multi-config training CLI** — `-c/--config` becomes `action='append'`, because a layer-analysis sweep previously meant editing the shell script and invoking the trainer once per model, re-paying interpreter, dataset and DB-startup cost every time; one invocation now queues several configs. `train.py` correspondingly accumulates `last_accuracy` and returns it only after the loop, since the old in-loop `return` abandoned every config after the first.
- **Idempotent stat writes + code override** — `save_stat` uses `INSERT OR IGNORE`, because a stat row's id is a content hash of the run, so re-running the same `(config, epoch)` would otherwise collide on the primary key and abort a resumed or repeated sweep. `save_results(..., nn_code=...)` routes through `code_to_db`, because generated and in-memory models have no registered name to look up, and storing the code that actually trained the model keeps the DB honest about what produced each result. `json_train_to_db` re-attaches the nested `train_stat` so training telemetry survives the flatten-to-columns step.
- **Unified stat JSON** — `Train.py` builds one `prm_json` containing `task/dataset/metric/model` promoted to the top level and `layer_stat` nested inside, so the file and the DB record describe the same experiment without a separate join step. `util/sh/run_layer_analysis.sh` is rewritten in the appendable `-c` style so several models share a single run.

### Removed
- **Per-epoch layer dumps** — `Train._save_layer_analysis`/`_save_layer_table` wrote a separate JSON file per epoch, scattering thousands of files that later had to be matched back to their run. Layer data now travels inside `prm_json["layer_stat"]` (`summary`, `layers`, `raw_analysis`), so the payload is self-contained.

### Fixed
- `util/sh/run_layer_analysis.sh` mis-cased the model name `rlfn` → `RLFN`; model names resolve case-sensitively, so the old spelling silently matched nothing.

## [2.2.9] - 2026-06-20

### Added
- ONNX export pipeline (`ab/nn/imp/onnx_pipeline.py`): trained checkpoints existed only as `nn`/`transform` code plus `.pth` weights, so the platform could not ship a framework-neutral artifact. The pipeline rebuilds each `Net` via `importlib`, exports it with `torch.onnx.export(opset_version=18)`, and re-scores 1000 test images through `onnxruntime`, because an export that raises no error can still be wrong. The input resolution is read from the transform file by regex instead of being assumed to be 32, successes and skips are written to separate `all_models.json`/`skipped_models.json` so one bad model cannot abort the sweep, and existing ONNX files are re-evaluated rather than re-exported.
- Age TFLite validation with MAE (`ab/nn/imp/run_tflite_validation_age.py`, `ab/nn/metric/mae.py`): age is a regression task, so classification accuracy is meaningless; a normalised MAE score of `max(0, 1 - MAE/15)` lets it pass the same `good()`/minimum-accuracy gate as every other task while `get_mae()` still reports the raw error in years. Because the deployment target is a phone, the INT8 TFLite conversion is validated rather than merely produced, and the report is kept on disk so a rerun only redoes the models that are not already `success`.
- Strict ONNX→TFLite converter (`util/py/age/convert_onnx_to_tflite_age.py`): it infers layout and size and refuses any TFLite input that is not exactly `[1, H, W, 3]`, instead of silently resizing the interpreter during evaluation, which previously hid export bugs.
- Layer analysis (`ab/nn/util/LayerAnalysis.py`): pruning and architecture search need to know which layers matter, so `LayerAnalyzer` measures them with WeightWatcher `alpha`, gradient/activation hooks, Taylor importance, `_linear_cka`, effective rank and sensitivity. Missing/NaN values are mapped to SQL `NULL` so stored statistics stay queryable.
- New transforms: `Resize_ColorJit_Flip_Blur.py` (224 face crops for age, with jitter/flip/blur/erasing), `norm_224.py` (224 input backbones), `sr.py`, and `cached_blip2fast_processor.py`, which returns data precomputed by BLIP-2/Q-Former so caption training does not re-run the vision encoder every epoch.
- Mobile/HF tooling (`util/py/age/{upload_tflite_for_mobile,upload_age_checkpoint_to_hf,monitor_mobile_testing,process_age_mobile_results}.py`).

### Changed
- Serialized DB writes (`ab/nn/util/db/Write.py`): many training jobs write the same SQLite file, so writers are wrapped in a re-entrant `_db_write_lock` (`flock` + `RLock`) to avoid corruption and self-deadlock, and `save_stat` now returns the `stat_id` that the new layer tables link to. `Init.py` sets `PRAGMA busy_timeout=300000` so a competing writer waits instead of failing.
- Telemetry schema: per-epoch diagnostics were accumulating inside the free-form `prm`, so they are normalised into `train_stat` (one row per `stat(id)`) and `layer_stat`/`per_layer_stat`, which makes CPU/GPU/RAM, loss and per-layer values addressable columns.
- Pretrained resume (`Util.save_if_best`/`load_pretrained_weights`, `train.py --layer_analysis`): the monotonic-best rule previously forgot the best score between runs, so a resumed run could overwrite a better checkpoint with a worse one; a `best_model.json` sidecar now persists it, and loading tolerates a missing file or an architecture change.
- Loaders: `utkface.py` gains an age-stratified split, `div2k.py` scale 3, `denoise.py` a 256 crop with augmentation, `imdb_wiki.py` caching, and `coco_/Caption.py` a cached-feature branch plus `gpt2_collate_fn` for decoder compatibility.
- Query rework (`db/Query.py`): `SIM_BANDS` is split into six similarity bands with anchor fallback and chunked `prm` loading, so curriculum selection stays responsive as the table grows.

### Fixed
- `layer_stat` epoch collisions — the row id now includes the epoch (`uuid4([stat_id, layer_name, str(epoch)])`), and `save_stat` removes `layer_stat` before the `prm` insert so it is not stored twice.
- Invalid layer rows are skipped with a warning instead of aborting the import when `layers` is not a list or a row has no name.

### Removed
- `ab/nn/imp/backup/*` (superseded quantisation scripts and JSON dumps); `ab/nn/metric/__init__.py` registry emptied.

## [2.2.8] - 2026-03-25

### Added
- `bleu1.py`–`bleu4.py` each expose a `BLEUMetric` that scores one n-gram order, so the pipeline can persist all four as separate `metric_*` columns while Optuna still ranks by BLEU-1, preserving the earlier 41–44% baseline instead of collapsing n-gram degradation into a single blended score.
- `ab/nn/imp/prHF.py` prunes checkpoints with `torch_pruning` (structured L1, layer-wise, `PRUNING_RATIO=0.30`) and uploads the `.pt` files with resumable history, so pruned variants are stored and re-runnable on HF rather than recomputed from scratch.
- `ab/nn/imp/upload2HF.py` converts to FP32/INT8, evaluates TFLite accuracy in a subprocess and retries on rate limits, so mobile artifacts and their measured accuracy stay together.
- `ab/nn/util/Inference.py` (`build_dataset_loader`, `build_model`, `run_inference`) moves the profiling logic into a reusable module.
- `ab/nn/transform/base/BatchTransform.py` (CutMix/MixUp/CutOut scheduled by epoch progress) plus `base/norm_256.py` and `norm_cc_256.py`, so augmentations follow a fixed order within an epoch rather than being chosen per batch.
- `test-nonnull.py` exercises the new non-NULL query path; new `tflite` and `prun` analytics tables and a BLOB `nn_minhash`.

### Changed
- `ab/nn/run.py` becomes a thin CLI (366 → ~24 lines) that delegates to `Inference.run_inference`, so the profiler has one implementation instead of two that drift.
- `Const.py` adds `tflite_table`, `prun_table`, `stat_run_tflite_{fp32,int8}_dir` and `stat_run_pt_dir`, giving the new artifacts typed locations.
- `Train.py` unpacks `model.learn()` into `(train_accuracy, train_loss)` and stores every metric as `metric_<k>`, so secondary metrics are persisted instead of discarded.
- `build_nn_similarity.py` drops `datasketch`/LSH for 128-perm `struct` BLOB packing and a `jaccard_blobs` SQLite UDF, so Jaccard is computed in SQL over compact 512-byte signatures without keeping an LSH dependency and its Python-side state.
- The read API gains `data_withnonnullvalue`, `tflite_data`, `prun_data` and `precision_type` in `run_data`, so callers can require rows whose `nn_stat`/`prm` values are actually non-NULL instead of filtering them out by hand.

### Fixed
- `db/Calc.py` gains `sanitize_for_json` and `_safe_json_load`, because NaN/Inf accuracies and their bare JSON tokens previously produced files that could not be loaded again.
- `Train.py` re-raises validation exceptions and `Write.py` logs and skips bad files, so a failed epoch no longer masquerades as a zero-accuracy result.
- `db/Util.py` resolves the package location through `importlib.metadata`, as `pkg_resources` is no longer reliable.

### Removed
- Legacy loader `ab/nn/loader/lemur-v2.py`, the datasketch-based similarity builder and the `nn_code_minhash` schema (replaced by `nn_minhash` with BLOB signatures).

## [2.2.7] - 2026-02-12

### Added
- INT8/TFLite export (`ab/nn/imp/quant8.py`): `convert_int8()` converts a checkpoint with a representative CIFAR-10 dataset and `TFLITE_BUILTINS_INT8`, so the model zoo can be measured on the hardware it actually runs on instead of only on the workstation. Accuracy is evaluated in a subprocess because the TensorFlow Lite interpreter is native code that can crash the whole process; isolating it lets a resumable run over hundreds of models record the failure and carry on rather than die. Latency comes from the LiteRT `benchmark_model` binary over `adb` on CPU, GPU and NNAPI, which is what feeds the mobile `run` analytics table; writes are atomic and resumable because Android benchmarking is slow and fragile.
- The first converted artifact (~306 KB, under `ab/nn/stat/run/...`) is committed so one known-good model can be benchmarked before the whole zoo is converted.
- CIDEr (`ab/nn/metric/cider.py`, `CiderScorer`/`CiderMetric`): captions were only scored by BLEU/METEOR, whose plain n-gram overlap tracks human judgement poorly; CIDEr's TF-IDF-weighted n-gram consensus is the standard captioning score, and the comma-separated metric support added below lets `bleu,meteor,cider` be evaluated together.
- Curriculum query tooling (`ab/nn/util/db/Query_CLI.py`, `build_nn_similarity.py`): the similarity edges become usable, so the next architectures to train can be chosen by a similarity band around a good anchor rather than at random.
- HF processor transforms (`blip2_processor.py`, `git_processor.py`, `sr_transforms.py`) and new networks `GIT-SOTA.py` / rewritten `rlfn.py` to cover the captioning and super-resolution tasks.

### Changed
- NN-statistics joins: static architecture properties (params, FLOPs, depth, residual/attention flags, code length) were already written to per-model JSON but were not queryable next to accuracy. Persisting them in `nn_stat` (keyed on `nn_name`+`prm_id`, via `save_nn_stat()` / `json_nn_to_db()`) and LEFT JOINing it behind `Read.data(..., include_nn_stats=...)` avoids a second lookup and lets a single query correlate architecture with accuracy and duration; `Read.nn_stat_data()` exposes the raw table.
- Code-similarity rework: similarity used to be recomputed in memory from the whole `data()` DataFrame, holding `nn_code` for every row and merging one large `nn.json`. Instead each model's stat JSON now stores a portable 128-perm/7-gram `code_minhash_signature()`, and similarity is streamed from those files as `.jsonl(.gz)`, keeping memory bounded and the result consumable piece by piece; `nn_code_minhash`/`nn_similarity` support the same data DB-side.
- `ab/nn/run.py` now delegates to `Train.Train.eval` instead of reimplementing a private accuracy class, so profiling scores with the same metric stack as training; durations are written in nanoseconds and the report is named `windows_<host>.json` under `stat/run/<config>_<arch>-<ts>/` so desktop and Android numbers share one unit and runs from several machines do not overwrite each other.
- Multi-metric training: `metric` is split on `,` into `self.metric_fns` with the first as `primary_metric`, and `eval()` returns `(primary_accuracy, all_results)`, so a model can be scored against several captioning metrics at once while ranking stays on one of them.
- SR loader and metrics: `div2k.py` pairs real LR/HR at X4 (the earlier X2 URLs and expected filenames did not match the downloaded set, and it now falls back to synthetic pairs when the data is absent); `psnr.py`/`ssim.py` compute on the Y-channel, the standard super-resolution convention, because RGB over-weights chroma.
- Repo layout: `cmd/` → `util/` groups all repo-level helper scripts under a name matching the library's `ab/nn/util`, and the added `__init__.py` files make them importable as modules instead of only runnable from disk, keeping tooling distinct from the shipped `ab` package.

### Fixed
- `HF.py` cache-cleanup `Path`/`str` bug; `Train.py` `accuracy = test_accuracy[0]` after eval began returning a tuple; `div2k.py` X4 scale and HR/LR pairing; `Const.get_version()` now resolves the root explicitly and prefers installed metadata, so version lookup no longer depends on cwd.

### Removed
- `SimpleAccuracy` and the inline profiler aggregation (superseded by `Train.eval`), and the DataFrame-based similarity merge (superseded by the streamed per-model signatures).

## [2.2.6] - 2026-01-11

This release consolidates six untagged intermediate versions (2.1.3 → 2.2.6), so the "why" matters more than the raw diff: the goal was to turn LEMUR from a CPU/GPU benchmark into a pipeline whose models can actually be deployed and whose results can be trusted and filtered.

### Added
- **FX-graph static quantisation** `ab/nn/imp/quantize_fx_static.py`. Static INT8 is what edge runtimes execute fastest, and FX tracing quantises the whole graph rather than individual modules; CIFAR-10 calibration supplies the activation ranges static quantisation needs, and `fbgemm` targets x86 CPUs. The comparison report exists so the accuracy cost of INT8 is measured per model instead of assumed.
- **Hybrid repair pipeline** `ab/nn/imp/quantize_hybrid_fallback.py`. Some architectures contain ops FX cannot statically quantise, and failing them outright would leave gaps in the model zoo, so when the static drop exceeds `ACCURACY_DROP_LIMIT=0.10` the pipeline retries with dynamic `qint8`. `QuantWrapper` provides the quant/dequant stubs eager mode requires, so every model still yields a deployable artifact.
- **TFLite export + LiteRT validation** `ab/nn/imp/run_tflite_validation.py`. Mobile is the real target, and export alone is not enough: the script converts through `ai_edge_torch` with the XNNPACK quantizer, then re-evaluates the flatbuffer in `ai_edge_litert` to confirm the artifact behaves like the PyTorch original.
- **Four loaders** (`denoise.py`, `div2k.py`, `imdb_wiki.py`, `utkface.py`) and **metrics** (`mai_psnr.py`, `meteor.py`, `ssim.py`, `ssim_module.py` + a `METRICS` registry). Classification alone could not validate quantisation across tasks, so denoising, super-resolution and age regression were added, each with the metric it needs: PSNR/SSIM on the Y-channel to match perceived brightness for restoration, and METEOR to complement BLEU for captioning.
- **Hugging Face DB sync** (`ZST.py`, `util/hf/{DB2HF,DB_from_HF,HF}.py`), with `init_population()` bootstrapping from HF. A fresh checkout cannot afford to regenerate thousands of stats, so the SQLite DB is zstd-compressed at level 16 and uploaded to HF, then downloaded transparently when no local `stat/` exists — keeping clones small and the DB reproducible.
- **Maintenance tooling** `cmd/py/checkpoints_to_hugging_face.py`, `cmd/py/log_nn_stat.py`, `ab/nn/util/db/Util.py`; **code-novelty analysis** `NNAnalysisSimilarity.py`. With thousands of generated models, near-duplicate code masks genuine novelty, so MinHash+LSH over 7-gram shingles (the same tokenisation as nn-gpt) scales similarity search instead of comparing every pair.
- **New models/transforms** — `DeepSeek-Coder-V2-Lite.py`, `DenoiseUNet2.py`, `rlfn.py`, seven `bf-v1-*_256.py` transforms and `ab/nn/transform/denoise.py`.

### Changed
- **Training telemetry** — `EpochMetrics`, `compute_gradient_norm`, throughput and `training_summary.json`. Persisting only a final accuracy made divergence and LR problems impossible to diagnose after the fact, so per-epoch loss/accuracy/LR/gradient-norm/throughput are now recorded as a learning curve.
- **Config paths** — `get_version`/`add_version`, `ckpt_dir`, `zst_db_file`, `stat_nn_dir`, `code_folders`/`gen_folders`, `core_nn`/`core_nn_cls`. Version-stamped artifact names stop one release overwriting another's DB and checkpoints, and the allow-lists separate hand-written architectures from generated variants.
- **DB/API filtering** — `unique_nn` (`ROW_NUMBER() OVER (PARTITION BY nn ORDER BY accuracy DESC)`) and tuple-valued `nn`, so repeated trials collapse to the best run per model instead of polluting queries.
- **PSNR normalisation** — raw dB is unbounded and not comparable across datasets, so the score is clipped to `[0, 48]` and min–max scaled to `[0, 1]`, which is what the optimizer and DB expect.
- **Packaging** — `packages.find` restricted to `["ab", "ab.*"]` (the old root-wide search mis-collected files), the `hf` extra dropped because DB sync is now core, and `ai-edge-torch`, `einops`, `datasketch`, `zstandard` added.

### Fixed
- The SQLite run-analytics schema is extended and nested stat directories are created with `mkdir(parents=True, exist_ok=True)`, so writes to new paths no longer fail; empty or infinite PSNR returns a neutral `0.5` rather than `NaN`/`0`.

### Removed
- `ab/nn/imp/auto_train.py`, superseded by the explicit training plus `checkpoints_to_hugging_face.py` upload flow.

### Documentation
- README: Optuna stat requirement lowered from 100 to 10 trials so a new contributor can reproduce results quickly; "run all tests before a PR"; LEMUR HF checkpoint link.

## [2.1.3] - 2025-12-11

### Added
- **Inference profiler** `ab/nn/run.py`: `run_inference()` builds a model from a dotted class path and scores it with a standalone `SimpleAccuracy`, so an architecture can be compared on inference cost the same way training runs are compared on accuracy, without dragging in the training metric registry. It samples `nvidia-smi` and process/system memory around each batch, folds `torch.profiler` key-averages into a top-40 op timeline, and writes `report.json` (device, timeline, peak RSS/GPU, metric) under `ab/nn/stat/run/<name>-<ts>/`, giving the `run` table added in 2.1.1 a producer of real device-analytics records.
- **Automatic train-and-upload loop** `ab/nn/imp/auto_train.py`: closes the gap between a recorded result and a downloadable artifact by reading each model's best existing hyperparameters from `data()`, retraining with `n_optuna_trials=-1` to force a fresh run, deriving `metadata.json` from the newest training stat, and pushing weights plus metadata to the `NN-Dataset/checkpoints` Hugging Face repo. Batch size is clamped to 32 and `num_workers=0` so a long unattended sweep survives OOM, and clearly unsuitable families are filtered out up front rather than crashing midway.
- **LEMUR v2 denoising task**: `ab/nn/loader/lemur-v2.py` downloads and extracts the noisy/clean train and validation archives on first use and pairs `denoised`→`original` at `(3,256,256)`, while `ab/nn/metric/psnr.py` supplies the matching restoration metric. PSNR is used because denoising has a clean reference, so pixel fidelity is directly measurable; the metric accumulates squared error across the epoch and returns a neutral 100.0 when the error is zero, avoiding `log(0)` infinities.
- **Static model analysis** `ab/nn/util/NNAnalysis.py`: instantiates each `Net` from its stored `nn_code` and writes cheap, deterministic structural descriptors (parameter counts, depth, conv/linear/activation/norm/pool statistics, estimated FLOPs, detected architecture patterns) to `stat_dir/nn.json`, so models can be grouped and filtered by structure without paying for a training or profiling run.
- **Supporting tooling/models**: `resize-lemur-256.py` transform; caption inference `cmd/py/infer_caption.py`; training-curve plotting under `simple_stat_viz/`; `DenoiseUNet` and the `C10C`/`C5C-RESNETLSTM` captioning variants; and a `pyproject.toml` `hf = ['nn-hf']` extra so the Hugging Face helpers stay an optional install.

### Changed
- **`epoch_max` replaces `n_epochs`/`num_epochs`** across `train.py`, `Train.optuna_objective` and `Train.train_n_eval`, and is now injected as `prms['epoch_max']`. The rename makes the epoch count an explicit upper bound (alongside the existing wall-clock `epoch_limit_minutes`) and, by recording it as an ordinary hyperparameter, lets every stored stat and Optuna trial state how long it was allowed to train.
- `Util.accuracy_to_time_metric` parameter `training_duration` → `duration`, since the same accuracy-per-time score is now computed for inference as well as training.

### Fixed
- `optuna_objective()` re-raises the original exception (`raise e`) instead of swallowing it in the empty `NNException`; the real traceback and message survive a failure, making retries debuggable. `LemurDataset.__getitem__` logs the offending file paths and returns zero tensors for a corrupt image, preserving the fixed `(3,256,256)` shape so one bad file cannot abort a long run.

### Removed
- Unused `ab/nn/util/Exception.NNException`, superseded by the `raise e` fix.

## [2.1.2] - 2025-11-15

### Added
- A catalogue of 155 `bf-v1-*.py` transforms under `ab/nn/transform/`, each one operation or a fixed chain ending in `Resize((64, 64))`, `ToTensor` and `Normalize`. The dataset compares architectures, so augmentation must be varied deliberately and held equal when it is not; naming each file after the operations it composes makes every regime addressable and reproducible, while numeric suffixes (`_4`, `_57`) separate parameter draws.
- A `cmd/` toolkit for repairing statistics the pipelines had already produced: `add_prm.py` merges new `gpu`/`cpu` fields into each recorded trial, `uid_gen.py` recomputes `uid` after the identity rule changed, and `duration_checker.py` drops entries with no duration so crashed runs stop counting as results. They rewrite local JSON in place, so they stay out of the importable package under `cmd/`, with a `README.md` marking them as operator tools.
- 44 generated `ab/nn/nn/rl-init1-*.py` models, growing the architecture axis alongside the transform axis, plus `evaluate` in `requirements.txt`.

### Changed
- Packaging ships what it claims: `data-files` is scoped to `ab` (`'ab' = ['**/*.json']`) instead of `'*'`, which had scattered the dataset's JSON across the install tree, and the `doc` package-data key picks up `licenses/*` so the third-party texts ship with a build.
- Licences moved `Doc/Licenses/` → `doc/licenses/`, with README links updated, because the old spelling never matched the lowercase `doc` glob and mixed case does not survive the case-sensitive filesystems that build and install the package; `sh/rename-dir.sh` moved to `cmd/sh/rename-dir.sh` for the same reason, grouping maintenance scripts in one place.
- `version` 2.1.1 → 2.1.2; README cites LEMUR 2.

### Removed
- The old `Doc/Licenses/` and `sh/` locations, and the large CIFAR-10 `ga-*` stat set, which duplicated pipeline output and only bloated the package.

## [2.1.1] - 2025-10-27

### Added
- **Mobile runtime analytics** — a new `run` table (device, OS, validity, emulator, error, JSON analytics) indexed by model and device, so on-device results can be compared beside training stats instead of living outside the database; runs accumulate far faster than those stats, which is why both lookup axes are indexed up front.
- **`api.run_data(model_name, device_type, max_rows)`** exposes those records publicly, so analysts can pull device telemetry without hand-writing SQL, with the analytics JSON parsed.
- **`join_nn_query`/`JoinConf`** (`db/Query.py`) — a frozen, validating config that compares related models in one query (`same_columns`, `diff_columns`, optional accuracy gain) and returns the variant as `nn_2`/`accuracy_2`/`prm_2`; comparing a seed with its variant previously meant two queries plus manual joining, so the comparison is now pushed into SQL.
- **`sql=` (`JoinConf`) and `nn_prefixes` on `api.data`** let a family of generated models (e.g. `rag-`) be filtered and compared in one pass instead of materialising the whole DataFrame and filtering afterwards; covered by `test_data_join`.
- **`save_onnx_weights` toggle** through `Const`/`Util.args`/`Train.train_n_eval`/`save_if_best` — mobile/non-PyTorch runtimes need ONNX, and exporting inside the best-model path means the exported graph is the checkpoint whose accuracy is reported, not whatever model finished last.
- **Heterogeneous `MoE-hetero4-Alex-Dense-Air-Bag`** — four structurally different experts behind one gate (load-balance loss, routing diagnostics), because homogeneous MoE variants had not shown whether cross-architecture routing actually helps.

### Changed
- Checkpointing centralized in `Util.save_if_best`, which owns `.pth` and `.onnx` and honours the save toggles; one owner keeps "best" consistent across models and demos and finally gates the `.pth` write, which was previously unconditional.
- `json_n_code_to_db` → `json_train_to_db` (+ `json_run_to_db`), so `init_population` also ingests runtime analytics; a malformed JSON is logged instead of aborting the entire import.

### Fixed
- Hyperparameter assembly uses one bulk `SELECT uid, name, value FROM prm`, replacing chunked `IN (...)` fetches that only dodged SQLite's variable limit.
- `str_not_none` prefixes a leading space so SQL fragments like `LIMIT` cannot glue onto the previous token, and checkpoint/ONNX dirs are created via `makedirs(dirname(path))`.

### Removed
- `ab/nn/metric/utils/` and per-epoch checkpoint bookkeeping in `ConditionalGAN`, both replaced by the centralized save path.

## [2.1.0] - 2025-10-15

### Added
- The public `data()` query accepts an optional `sql` string that is spliced into the read query as a CTE over a `WITH all_data AS (...)` wrapper. The wrapper always exposes the fully joined dataset, so callers can aggregate, filter or reshape results with their own SQL while keeping the same API signature and column contract; when `sql` is omitted the query falls back to `SELECT * FROM all_data`.
- Training can save the best PyTorch weights (`save_if_best`, `export_torch_weights`), enabled with `--save_pth_weights`. It is off by default so ordinary benchmark runs stay lightweight and the repository does not fill with checkpoints; only users who need to reuse a model pay the disk cost, and only the best-scoring epoch is kept.
- A second, CVAE-based text-to-image demo (`demo/Text-ImageCVAE.py`) beside the GAN one, a bulk directory renamer (`sh/rename-dir.sh`), and a `demo/README.md`.

### Changed
- Training artifacts move from `stat/` to `stat/train/` (`stat_train_dir`), and `model_stat_dir()` and the DB importer scan the new location. This frees the top-level `stat/` for other artifact kinds, so training statistics no longer collide with run or inference output.
- The demo now lives at the repository root (`ab/nn/demo/` → `demo/`) and fetches generator weights on demand from the Hugging Face hub, so it runs from a clean checkout instead of depending on a pre-existing local checkpoint.
- The COCO text-to-image loader reads the image size from the framework transform and adds an explicit `CenterCrop`, so every sample is square, and skips corrupt images instead of crashing training.
- `Train` records `self.model_name`, which the checkpoint saver uses to name its directory.

### Fixed
- Evaluation moves both inputs and labels to the device unconditionally, so tasks with non-tensor or CPU-resident labels no longer fail.

### Removed
- The `ab.nn.api.accuracy_to_time_metric` wrapper; the metric remains available as `Util.accuracy_to_time_metric`.

## [2.0.6] - 2025-10-03

### Added
- Text-to-image demo (`demo/app.py`, `demo/index.html`, `demo/requirements.txt`): a standalone FastAPI app serves `POST /generate_image`, which runs the trained `ConditionalGAN` generator on a `CLIPTokenizer`-encoded prompt and returns the saved PNG. It exists to make the trained generator usable from a browser without dragging in the training framework, and its file route rejects `..` paths because images are served back by name from a writable directory.
- MoE variants (`MoEv2.py`–`MoEv9-AlexNetv4.py`, 12 files): top-2 gated experts, `SEBlock` attention experts, class-biased expert initialisation, `mixup_data()` and `ExpertUtilizationTracker`. These are deliberate probes into the MoE failure modes seen earlier: top-k gating with load-balancing/importance/entropy aux losses guards against expert collapse (where the gate routes everything to a few experts and the rest never train), `SEBlock` adds channel attention where a plain expert overfits, and the class prior gives each expert a starting specialisation so the gate has something to differentiate.
- Regenerated `ab/nn/nn/rag-*.py`/`alt-nn*.py` candidates.

### Changed
- `ab/nn/util/Train.py` calls `release_memory()` after the model is deleted: `del` only drops the Python reference, and CPython/CUDA do not reliably return the memory at once, so a sequence of training runs kept growing in RAM/VRAM until it failed; `gc.collect()` plus `torch.cuda.empty_cache()` (`Util.release_memory`) actually reclaims it.
- `db/Write.py` `json_n_code_to_db()` wraps its `for sub_config_str in sub_configs` loop in `tqdm` and prints the target directory, so the long JSON import shows progress and a stall can be told apart from a hang.
- `version` 2.0.5 → 2.0.6, with the `pyproject.toml` authors table collapsed to a single line (no behaviour change).

### Fixed
- Memory accumulation across runs is mitigated by the `release_memory()` call above.
- Expert collapse in the MoE probes is addressed in `MoEv8`/`MoEv9-AlexNetv3` through the stronger balancing losses and class-biased initialisation described above.

## [2.0.5] - 2025-09-25

### Added
- **Transforms can be loaded from any directory** — `Loader.load_dataset` now takes `transform_dir`; when `<transform_dir>/<transform_name>.py` exists it is imported by file path and its `transform` function is used, and otherwise the loader falls back to the built-in `ab.nn.transform` registry. External and one-off transforms can be tried without adding them to the package, and existing configs keep working because the registry remains the default.
- **`transform_dir` is exposed through the public API** — `check_nn(..., transform_dir=...)` forwards to `Train.train_new` and then to `load_dataset`. The directory has to reach the training entry point, not only the loader, or a custom transform could not be selected programmatically.
- **Resumable batch runner `run_all_models.sh`** — iterates over a model glob, runs `python -m ab.nn.train` for each, and appends each success to `processed_models.log`; a re-run skips the finished models. A long sweep can resume after a failure instead of repeating work that already succeeded.
- **COCO reports real shape metadata** — the text-to-image loader returns `((in_shape, out_shape, class_names), 0.0, train, val)` with the output image size read from the transform and a `vocab_size` input shape, instead of `(None,)`; the framework and metric need those shapes to build and evaluate the model correctly.

### Changed
- **COCO trusts the transform it is handed** — the loader no longer rebuilds its own `Resize/CenterCrop/ToTensor/Normalize` pipeline but uses `transform_fn((NORM_MEAN, NORM_DEV))`. Rebuilding it silently discarded the transform requested by every caller.
- **CLIP uses the model's own similarity** — scoring reads `outputs.logits_per_image.diag()` and divides the average by 100, following CLIP's convention, rather than a hand-rolled embedding dot product.
- **`setuptools` unbounded** — the `setuptools<80.0.0` pin blocked installs on newer setuptools builds for no functional reason.

### Fixed
- **Non-tensor labels no longer break evaluation** — the eval loop moved `labels` to the device unconditionally, which fails for image-generation tasks whose labels are raw text/prompt lists; tensors are still moved and everything else is passed through untouched.
- **CLIP survives malformed or empty batches** — unexpected `preds` shapes and empty lists are ignored instead of raising.
- **COCO image-size detection is guarded** — if no `Resize` step can be found the loader warns and falls back to 256 instead of raising.

### Removed
- **`ftfy` hard dependency** — the stable CLIP processor no longer needs it.
- Obsolete COCO transform-rebuild path.

## [2.0.4] - 2025-08-26

### Added
- **Fixed hyperparameters via CLI** — `-p/--nn_prm` (parsed with `json.loads`, default `default_nn_hyperparameters={}`) accepts e.g. `{"lr": 0.0061, "momentum": 0.7549, "batch": 4}`, so a known-good run can be reproduced with the exact values that produced it instead of being re-searched differently each time.
- New `C1C-RESNETLSTM.py` and 51 RAG-generated `rag-*.py` candidates.

### Changed
- `train.main` wraps `nn_prm` in a read-only `MappingProxyType` and forwards it to `optuna_objective`, which starts from `dict(nn_prm)` and suggests a hyperparameter only when it is absent or falsy: pinned values are kept, everything else is still searched.
- Pinned `batch`/`transform` are honoured by the new `Util.add_categorical_if_absent(...)`, which calls `trial.suggest_categorical` only for unset names and builds the candidate list lazily (`default or fn()`).
- `version` 2.0.4 → 2.0.5; README adds "Reproducing Results with Fixed Training Parameters" and `pip install nn-dataset[stat]`.

### Fixed
- BagNet declared a `SupportedHyperparameters` class with `check_hyperparameters` instead of the `supported_hyperparameters()` the trainer calls, so its `{'lr','momentum','dropout'}` set was never found; it now exposes the function the framework expects.
- `Train.py` warning text corrected (`staticis`→`statics`, `save_Path`→`save_path`) so it names the real field.

### Removed
- Duplicate imports and a second, identical `supported_hyperparameters()` in `ConvNeXtTransformer.py`, a dead shadow of the real declaration that could silently drift out of sync.
- The commented-out beam-search `forward`/`generate` block in `ResNetTransformer.py`, superseded by `beam_search_generate`/`batch_beam_search`, so the file no longer carries two decoding paths.

## [2.0.3] - 2025-08-20

### Added
- **CLIP text-to-image metric** `ab/nn/metric/clip.py` (`CLIPMetric`, `openai/clip-vit-base-patch32`) — generated images have no ground-truth label to score against, so instead of comparing to a target the metric embeds each generated image and its prompt with CLIP and reports the mean cosine similarity of the two normalised embeddings; the model/processor are cached per device (the backbone is loaded once per run, not per batch) and the small `base-patch32` variant keeps scoring cheap enough to run on CPU.
- **Conditional VAE** `ab/nn/nn/ConditionalVAE.py` (encoder→`fc_mu`/`fc_log_var`, `SelfAttention` decoder, `PerceptualLoss` on VGG16) — a latent-variable counterpart to the GAN/diffusion text-to-image models, so the benchmark can compare generative families rather than one. The VGG perceptual term is added to the L1 reconstruction loss because pixel loss alone yields blurry samples; `SelfAttention` lets distant regions (e.g. the car and its background) influence each other, which matters when condensing one category; and the `kld_warmup_epochs` schedule ramps the KL weight so the posterior is not collapsed before reconstruction is learnt.
- **20 classifier backbones** `ab/nn/nn/rag-*.py` (MBConv/EfficientNet, GhostNet, MobileNet inverted-residual, CSP, YOLO/RT-DETR blocks) — efficient, well-proven designs added as a batch so the architecture search gains strong low-FLOPs baselines instead of relying only on the hand-written models.

### Changed
- **Conditional diffusion** `ConditionalDiffusion.py`: `supported_hyperparameters()` → `{'lr','beta1','beta2','steps_per_epoch'}`, AMP + `GradScaler`, xformers memory-efficient attention, frozen VAE. Training a UNet alongside a VAE and a text encoder exceeds a single GPU's memory, so the VAE is frozen (it only supplies latents), xformers attention and gradient checkpointing shrink the activation footprint, and AMP/`GradScaler` halves the compute and memory of the forward/backward pass; the hyperparameter set is narrowed to the values the training loop actually consumes.
- **COCO Text2Image loader** filters to one category, emits eval labels, deterministic square transform. Fitting a small conditional model to all 80 COCO categories from scratch is infeasible and would make the CLIP score hard to interpret, so images are restricted to `TARGET_CATEGORIES=['car']` to give the model one consistent text→image mapping; the transform is rebuilt as `Resize` + `CenterCrop` because the VAE/UNet assume square inputs, and the val split returns a dummy tensor label since evaluation prompts are supplied by the model.
- Deps: `transformers`, `accelerate`, `diffusers`, `ftfy`, `scikit-learn` — required by the CLIP metric, the DistilBERT text encoders, the diffusers VAE/UNet/scheduler, CLIP text cleanup, and metric math; 1015 `ga-alexnet-*.py` renamed to `ga-*.py` to drop the redundant `alexnet` prefix from these generated variants; `train.sh` made executable.

### Fixed
- Image-load errors print a warning and advance, so one unreadable COCO file cannot abort a whole epoch; CLIP uses the stable slow processor, whose tokenisation is deterministic, instead of the fast processor that scored unstably.

### Removed
- Standalone inference CLI in `ConditionalDiffusion.py`; generation is now driven through the framework's eval loop, so there is a single code path for training and inference.

## [2.0.2] - 2025-08-12

### Changed
- Built distributions now carry the JSON dataset. `package-data` keys are matched per package directory with non-recursive globs, so `'ab' = ['*.json']` only ever matched JSON sitting directly in `ab/`; the deeply nested `ab/nn/stat/**` stat/config files were silently dropped from wheels and sdists. Moving JSON to `[tool.setuptools.data-files] '*' = ['**/*.json']` uses a recursive glob over the whole tree, so one pattern ships every JSON artifact at any depth.
- The `ab` package-data entry is narrowed from `['*.py', '*.json', '*.md']` to `['*.md']`, because `.py` sources are already installed as package modules and `*.json` is now owned by `data-files`; keeping either would only double-ship files, so only the package-local markdown docs still need to travel with `ab`.
- `version` 2.0.1 → 2.0.2, a patch release that changes packaging only.

### Fixed
- Missing nested JSON in built distributions: the non-recursive `*.json` glob was replaced by the recursive `**/*.json` one.

## [2.0.1] - 2025-08-12

### Changed
- Packaging moved from `setup.py` to a declarative `pyproject.toml` (PEP 621): the metadata now lives under `[project]`, so builds no longer import or execute project code, and any PEP 517/518 frontend can read the same fields instead of relying on setuptools' imperative API.
- `version` and `dependencies` are declared dynamic (`version = { file = 'version' }`, `dependencies = { file = 'requirements.txt' }`), so the `version` file and `requirements.txt` remain the single source of truth; previously `setup.py` re-read both by hand, which let the installed metadata drift from what the repository actually stated.
- Package discovery and shipped data are delegated to `[tool.setuptools.packages.find]`, `include-package-data`, and `[tool.setuptools.package-data]`, replacing the hardcoded includes; the contents of a wheel are now described in one place rather than split across `packages=`, `include_package_data`, and `MANIFEST.in`.

### Removed
- `setup.py`, because everything it did (reading `README.md`, `requirements.txt`, and `version`, and listing packages) is now expressed directly in `pyproject.toml`; keeping both would duplicate the metadata and let the two copies disagree.
- `MANIFEST.in`, because its `recursive-include` rules duplicate what `package-data` already ships; maintaining both invites the sdist and the wheel to package different files.
- The empty top-level `__init__.py`, which otherwise made the repository root look like an importable package.

## [2.0.0] - 2025-08-11

### Added
- Text-to-image task and COCO loader (`ab/nn/loader/coco_/Text2Image.py`): the framework could benchmark models that classify, detect, segment or caption, but not a model that *consumes* text to synthesise an image, so the newly added conditional generators had no benchmark. The loader draws one random caption per COCO image (auto-downloading annotations and images), registers `txt-image` in the COCO task dispatcher, and steps to the next index on unreadable images so one corrupt file cannot abort a training epoch.
- FID metric (`ab/nn/metric/fid.py`): generated images are judged by distributional distance to real ones, which the framework's accuracy-style maximisation cannot express. FID is lower-is-better, so the result is returned negated; the InceptionV3 feature extractor is cached globally to avoid reloading it on every batch, and the matrix square root is regularised and de-complexified so degenerate covariances cannot crash a run.
- Shared metric helpers (`ab/nn/metric/utils/utils.py`, `metric/base/__init__.py`): `flatten_logits_and_labels` folds `[B,S,V]`/`[B,S]` sequence outputs to `[B*S,V]`/`[B*S]`, so the same classification and cross-entropy logic serves both sequence and single-label models instead of each metric re-implementing the reshape.

### Changed
- Perplexity is min-max normalised in log-space to `[0,1]` (`PPL_MIN=1`, `PPL_MAX=1_000`): raw perplexity is unbounded and lower-is-better, so it was neither comparable across datasets nor usable by the maximisation loop. Log scaling keeps the whole 1–1000 range meaningful, mapping a perfect model to 1 and anything at or beyond the worst threshold to 0.
- Accuracy `update()` now runs through `flatten_logits_and_labels` and `__call__` returns the accumulated totals, so sequence models are scored on every token and batch statistics have a single source of truth instead of being computed twice.
- `acc.py`, `ppl.py`, `iou.py` and `map.py` import `BaseMetric` relatively (`from .base import ...`): absolute `ab.nn.metric...` imports pin metrics to the installed package root, which breaks under the namespace-package install or when metrics are loaded from a different checkout; relative imports keep the metric package portable.
- `accuracy_to_time_metric` and `good` treat a missing accuracy or threshold as `0.0`, so runs without a score (or a dataset without a minimum) are ranked as worst-case rather than raising `TypeError` during result ranking.
- `version` 1.2.6 → 2.0.0: metric score semantics changed (the FID sign and the perplexity scale), so new statistics are intentionally not comparable with older ones.

### Fixed
- `None`-safety in `Util.py`, and `Text2Image.__getitem__` skips unreadable images and advances instead of failing the whole epoch.

## [1.2.6] - 2025-07-17

### Added
- **Perplexity metric** (`ab/nn/metric/ppl.py`): character-level language models had no usable score, so WikiText could be trained but not ranked. Perplexity is the exponential of the mean per-token cross-entropy, min-max normalised in log space over `PPL_MIN=1`–`PPL_MAX=1000` into `[0,1]`, which makes it comparable with the other accuracy-like metrics and lets the shared threshold/ranking logic treat it uniformly. An optional `ignore_index` excludes padding so batches of unequal length do not skew the score.
- **New models** `ConvNeXtTransformer.py`, `ResNetTransformer.py`, `MoE-0707.py`.
- **New transforms** (`center_crop.py`, `five_crop.py`, `random_horizontal_flip.py`, `random_vertical_flip.py`): flips and crops previously existed only inside larger composed recipes, so the search could not isolate a single augmentation's effect. Each file is a minimal, standard `transform(norm)` recipe (one operation, then resize to 64×64 and normalise), giving the pipeline catalogue independent, attributable variants.
- **Shell entry points** `train.sh`, `train-loop.sh`, `train-loop-50.sh`.

### Changed
- **WikiText loader rewrite**: the loader now exposes `out_shape=(vocab_size,)` instead of `None`, because the perplexity metric and cross-entropy loss need the model to emit `[B,S,V]` logits, which require the vocabulary size up front. The dataset switched from non-overlapping blocks to a stride-1 sliding window, so every character position becomes a training example (~`seq_length`× more samples from the same small corpus) and next-character prediction is learned densely rather than only on block boundaries; `seq_length` rose 100→128 for a longer context. Targets are now `torch.long` (cross-entropy) rather than float, indices are precomputed once, and `__getitem__` bounds-checks.
- `minimum_accuracy` → `MINIMUM_ACCURACY` across all loaders, with `min_accuracy()` reading the new name; the old lowercase spelling was easy to miss and inconsistent between files.
- COCO Caption NLP now uses `nltk.word_tokenize`, `threshold=5`, and a process-wide `GLOBAL_CAPTION_VOCAB`, so captions with punctuation and numbers split correctly and rare words collapse to `<UNK>` instead of bloating the vocabulary.
- **Per-epoch time limit expressed in minutes** (`default_epoch_limit_minutes=30`): the old `Const.max_epoch_seconds` was a globally-mutated constant in a different unit from the duration that is reported and stored. The limit is now an explicit parameter threaded `train.main → optuna_objective → Train.train_n_eval → DataRoll`, so it is per-call and shares the unit of the training-time report.
- `check_nn()` returns a 4-tuple `(name, accuracy, accuracy_to_time, score)` and `train_n_eval()` returns `(accuracy, accuracy_to_time, duration)`: the old return dropped the accuracy-to-time metric and let the time-penalised value be reported and compared as accuracy, so the objective and the `minimum_accuracy` gate used the wrong number.
- `check_nn()`/`data()` drop `cast_prm`, and DB joins/indexes move to `(uid, name, value)`: hyperparameters have been stored in native Python types since 1.2.2, so the runtime cast flag was dead weight, and the covering index turns the bulk `WHERE uid IN (...)` hyperparameter fetch into an index-only lookup.
- Sequence models modernized (`LSTM.py`, `RNN.py`, `RESNETLSTM.py`, now with AMP).

### Fixed
- Duplicate `trial.suggest_float` for `lr` removed: the learning rate was sampled twice, wasting an Optuna sample and distorting parameter-importance reporting.
- `uid` is computed from `only_prm` (hyperparameters alone, excluding `duration`/`accuracy`/`epoch`): hashing the full dictionary gave the same configuration a different id every epoch, defeating dedup. The stat id is likewise now `uuid4(configuration + transform)` and `populate_prm_table` uses `INSERT OR IGNORE`, so re-saving is idempotent.
- `Read.data` bulk-loads hyperparameters by `stat_prm` rather than `stat_id` (the parameter tables are keyed by the stat's `uid`, so the previous key returned empty `prm` dictionaries), and now selects `metric_id`/`nn_id`/`transform_id` alongside the code columns.
- BLEU skips empty references.

### Removed
- `run.py`, `run.sh`: `run.py` duplicated `ab/nn/train.py`'s argparse setup and kept a hand-maintained argument list that drifted whenever a flag was added. `train.sh` now activates `.venv` if present and calls `python -m ab.nn.train "$@"`, so the CLI has one source of truth and shell callers forward flags transparently.
- `ab/nn/nn/LSTMK.py`; `test_cast_flag_equivalence` and `cast_prm`.

## [1.2.5] - 2025-06-06

### Changed
- Relaxed the lower bounds on `tqdm`, `optuna`, `numpy`, `pandas`, `scipy` and `pillow` from `>=` pins to plain requirement names. Those floors were never real runtime requirements; they only forced pip to upgrade already-satisfied dependencies and made resolution fail when another package — notably `torch` — constrained the same library to an older release. Dropping the floors lets pip reuse whatever compatible version the target environment already has, so installs stop failing for no functional reason.
- Changed the `setuptools` cap from `<=80.0.0` to `<80.0.0`, so version 80.0.0 itself is now excluded rather than allowed. The inclusive bound was an off-by-one against the intended "stay below setuptools 80" limit: that release dropped build behaviour the package's `setup.py`-era build still relies on, so a range admitting it could install a setuptools that breaks the build. The exclusive bound restores the intended ceiling.
- Bumped `version` 1.2.4 → 1.2.5 for the release.

## [1.2.4] - 2025-06-06

### Added
- Model and run identity is now the content hash of what is recorded (`uuid4(obj)` = MD5 of the whitespace-stripped string): equal code or equal hyperparameters map to the same name and `uid`, so re-runs update one row instead of duplicating it.
- `test_nn_train` trains CIFAR-10 for an epoch, because the tests all passed while training was broken.

### Changed
- Task key `image-captioning` → `img-captioning`, matching the `img-*` prefix of every other task (`img-classification`, `img-segmentation`); `coco.py`'s dispatch and the loader's guard changed together so they cannot drift.
- The captioning pipeline was reworked end to end: the vocabulary now lives on the dataset, and `collate_fn(batch, word2idx)` wraps captions in `<SOS>`/`<EOS>`, pads to `[B, caps, T]` and returns token ids, with the embedding moved into `LSTMDecoder` so the pipeline is device-free and the model owns its parameters. `RESNETLSTM` trains by teacher forcing and decodes greedily at inference, with `Adam` instead of `SGD` for its pretrained, frozen ResNet-50 backbone, and `minimum_bleu` fell 0.05 → 0.001 so early training is no longer discarded as "too low".
- BLEU now scores BLEU-1..4 separately and returns BLEU-4, ignoring padding and empty references and accepting logits or ids; the old metric averaged one opaque score that counted pad tokens.
- `requirements.txt` relaxed `~=` → `>=` so installs can take newer compatible releases, added `nltk` and `gdown` for captioning, and bounded `setuptools<=80.0.0` for build compatibility.

### Fixed
- Caption images load in a bounded two-attempt loop (local, then a 10 s download) that reports the failing path, and `validate_prm` guards each range only when both bounds are set.

### Removed
- `BaseMetric` inheritance from `bleu.py`, the local `uuid4` in `Write.py`, and the loader-built shared embedding.

## [1.2.3] - 2025-05-19

### Added
- Image captioning as a first-class COCO task (`ab/nn/loader/coco_/Caption.py`): detection and segmentation targets are per-box or per-pixel and a caption model cannot consume them, so `COCOCaptionDataset` returns an image with its raw captions and `build_vocab` builds a shared vocabulary with `<PAD>/<SOS>/<EOS>/<UNK>`. Captions vary in length and count, so `collate_fn` pads both to the batch maximum, and the loader returns `(vocab_size,)` so the framework can size the decoder head.
- BLEU (`ab/nn/metric/bleu.py`) because token accuracy is meaningless for free-form text: it measures n-gram overlap with the references while keeping the `BaseMetric` interface, and captioning networks `RESNETLSTM.py`/`LSTMK.py`, which emit a logit per vocabulary token instead of one class.
- `ab/nn/util/DeleteNN.py`, because deleting a model by hand left its `<name>`-suffixed stat directories and its DB rows, so it kept appearing in queries; the tool removes file, stat dirs and DB, which is rebuilt clean.

### Changed
- `coco.py` dispatches `image-captioning` to `Caption`, keeping task-name selection in one entry point, and `Const.model_script(name)` centralises the `<nn_dir>/<name>.py` path shared by the tool and the default model.
- `Read.query_rows()` returns plain tuples, since callers reached `sqlite3.Row` objects that broke the set/dedup comparisons used to collect configs; `test.py` runs a real `check_nn` pass (as well as the mocked path) so the caption wiring is covered by the suite.

### Fixed
- `train_new()` wraps each cleanup `del` in `try/except NameError`: when loading failed before `train_set`/`test_set` existed, cleanup raised a misleading `NameError` that hid the real error, letting the original exception surface.

### Removed
- 27 generated `*-<uuid>.py` models and their stats, plus the unused `gdown` dependency.

## [1.2.2] - 2025-05-02

### Changed
- Hyper-parameters are now stored in the `prm` tables in their native type (`(uid, name, value OBJECT)`) instead of `str(value)` plus a `type` column that reads cast back with `getattr(builtins, ...)`. The stringify-and-cast round-trip rejected values whose type is not a `builtins` name and paid a Python-level cast per row, so writing `int`/`float`/`str` natively keeps type fidelity and turns `cast_prm` into a no-op (`test_cast_flag_equivalence`, `test_native_param_types`).
- `data()` reconstructs hyper-parameters with a single chunked `SELECT uid, name, value FROM prm WHERE uid IN (...)` (900 ids per chunk, below SQLite's 999-parameter limit) into a `defaultdict`, replacing the old one-query-per-stat-row `WHERE uid = ?` loop. That loop had no index and full-scanned `prm` for every row, so read cost scaled with rows × columns; the new `idx_<table>_uid` indexes plus the bulk fetch keep it proportional to rows, which `test_fetch_time_budget` guards at ≈0.25 ms/row.
- The connection uses `row_factory = sqlite3.Row`, and every `data()` query runs inside `try/finally: close_conn`, so column-name access replaces manual tuple zipping and the connection is closed even when a query raises.
- `ab.nn.api.data()` is wrapped in `lru_cache(maxsize=10)`, so repeated identical queries (tests, CLI, `only_best_accuracy`) reuse the cached frame instead of re-reading the database.
- `crate_file` is renamed `create_file` and removes an existing file before writing; the old version skipped the write when the path already existed, leaving stale content or failing on repeated writes.
- ONNX export moved inside the `train_new` `try` block (with `trainer = None` initialised first), and `train_set`/`test_set`/`trainer.model` are deleted in `finally`, so the model and datasets are no longer leaked on the error/early-return paths.

### Added
- A rewritten `test.py` that checks the new data contract instead of running a real train: `test_native_param_types` round-trips float/int/str (and bool as 0/1) through write→read, `test_cast_flag_equivalence` proves the removed cast is neutral, `test_fetch_time_budget` fails if bulk loading regresses, `test_remaining_trials`/`test_supported_transformers`/`test_unique_configs`/`test_fetch_all_data_filters` cover the read helpers, `test_check_nn` stubs `train_new`, and `setUpClass`/`tearDownClass` insert then delete a synthetic row and assert it is gone.

### Removed
- The `type` column on the `prm` tables and the runtime casting path it fed.

## [1.2.1] - 2025-04-08

### Added
- Configurable dropout search range (`--min_dropout`/`--max_dropout`, threaded through `train.main` and `optuna_objective`). Optuna had hard-coded `0.0–0.5` with the note that high dropout prevents learning, so every model was searched against the same assumed ceiling; exposing the bounds lets the range be narrowed or widened per experiment instead of editing `Train.py`.
- `Util.torch_device()`, `Util.create_file()`, `Util.str_not_none()` and the DataLoader factories `train_loader_f()`/`test_loader_f()`, plus `onnx_dir`/`onnx_file` constants and the `onnx` dependency.
- `max_rows` on `data()` (API and `db/Read`), rendered with `str_not_none('LIMIT ', …)`, so callers can cap how many rows a query materialises instead of always loading every matching record.
- `check_nn(epoch_duration_limit_sec=...)`. The 30-minute `max_epoch_seconds` was a module constant imported *by value* in `DataRoll`, so a caller could not change it; `api.check_nn` now rebinds `Const.max_epoch_seconds` and `DataRoll` reads it at call time, so long CPU runs can raise the limit.

### Changed
- Device selection and loader construction are centralised in `Util.torch_device()` and the loader factories. The MPS-aware device check was duplicated, and the test loader incorrectly reused the *train* dataset's `num_workers`/`collate_fn`; one helper now yields a consistent device and each loader is built from its own dataset.
- `Train.__init__` now takes a dotted `nn_module` path, `get_attr` no longer prepends `ab.nn`, and `get_ab_nn_attr` preserves the old behaviour for dataset/model helpers. This decouples training from the `ab.nn.nn` package, which is what makes dynamically generated models loadable by name.
- `train_new` writes generated code to `out/nn/tmp/<uuid>.py` (an importable package seeded with `__init__.py`) rather than a `NamedTemporaryFile`, because the module must be importable by dotted path while `Train` loads it; the file is deleted in a `finally` so failed runs do not leak it.
- `export_model_to_onnx(model, dummy_input)` is simplified: it writes to the fixed `onnx_dir`/`onnx_file`, runs `model.eval()` under `torch.no_grad()`, and exports a real training sample instead of `torch.randn(in_shape)`, so the graph matches the actual input and omits training state.
- Inline RNN/LSTM construction for `txt-generation` is dropped from `optuna_objective`; those models flow through the same `Net` path as everything else.

### Fixed
- The test loader derives `num_workers`/`collate_fn` from `test_dataset`; `validate_prm` rejects `dropout_min > dropout_max`; `CodeEval.dynamic_evaluation` builds the model with an explicit device and the `train_setup(prm)` contract, and the code-quality score is normalised to `[0,1]`.

### Removed
- Ad-hoc temp-module loading and the `txt-generation` special case in `optuna_objective`/`train_new`.

## [1.2.0] - 2025-03-31

### Added
- Unified metric interface `ab/nn/metric/base/base.py`: `BaseMetric` fixes one contract — `reset()`, a per-batch `__call__()`/`update()` and `result()`. Previously `Train.load_metric_function()` hard-coded `if metric_name == "iou" … elif "map" … else compute` and `eval()` guessed the shape of the object with `hasattr(..., "update")`/`hasattr(..., "get")`, so every new metric meant editing the trainer and every metric had a slightly different lifecycle. Each metric module now just exposes `create_metric(out_shape)`, so `Train` builds a metric with `module.create_metric(self.out_shape)` and trains through the fixed sequence; metrics became plug-ins and the `get`/`compute` split disappeared. `Accuracy`, `MIoU` and `MAPMetric` were reimplemented on the base.
- `Util.export_model_to_onnx(model, model_name, dummy_input)` and `check_nn(..., export_onnx=...)`: trained networks can now leave the PyTorch training stack as a runtime-neutral ONNX graph, which is what later mobile/quantisation work consumes. The exporter checks for non-trivial `AdaptiveAvgPool2d`/`AdaptiveMaxPool2d` and only marks the spatial axes dynamic when they are actually free, because `torch.onnx.export` cannot combine free height/width with a fixed adaptive-pool target and would otherwise fail or emit a wrong graph.
- `default_pretrained=None` with a `--pretrained {1,0}` flag and `prms['pretrained']` handling in `optuna_objective`: many models can initialise from pretrained weights, but whether that helps depends on task and dataset, so the default now lets Optuna search `pretrained` like any other hyperparameter while 1/0 forces the decision for reproducible runs.
- `Util.min_accuracy(dataset)` and public `ab.nn.api.accuracy_to_time_metric(accuracy, training_duration, dataset)`: speed-aware benchmarking ranks models by the accuracy they gain per unit time, so the wrapper resolves the dataset's floor via `min_accuracy()` and exposes the metric to API users. It is deliberately not stored in the database, since it can be recomputed at any time from the stored accuracy and duration.
- New models `Diffuser.py`, `FractalNet.py` and ~81 NNGPT-generated UUID-suffixed models; the `ab/nn/tmp/` package and a `version` file.

### Changed
- The `Loader` class became the module functions `load_dataset()` and `get_obj()`. The static wrapper added nothing, and routing loaders/transforms through the same dynamic name lookup makes dataset loading consistent with how metrics and transforms are resolved.
- Loaders now expose `minimum_accuracy` as part of the dataset contract, so the acceptance threshold travels with the data instead of being a trainer-side constant.
- `train_new()` writes the submitted `nn_code` into the new `ab/nn/tmp` package and imports it from there, so dynamically submitted architectures are real importable modules — needed for correct model naming and ONNX export.
- `check_nn()` now returns `(name, accuracy, score)`, surfacing the static code-quality score next to accuracy; `train_new()` gained `prefix`/`save_path` to control the generated model name and where statistics are written.
- `setup.py` reads the version from the `version` file instead of a hard-coded string, removing a second place that had to be bumped per release.

### Fixed
- `LearnTimeException` is reported as a bounded ratio and no longer masks the duration; `Calc.save_results` creates the missing stat directory instead of failing on first write; the eval loop reads `metric_function.result()` under the new interface.

### Removed
- The `Loader` class and the per-metric `compute()`/`get()` methods, along with the hard-coded metric dispatch they required; commented-out AP printing.

### Documentation
- README rebranded to LEMUR (with the `lmur` PyPI alias); citation updated (including Tolgay Atinc Uzun, Zofia Antonina Bentyn); NNGPT acknowledged as backer.

## [1.0.4] - 2025-02-09

### Added
- **Namespace-package support** — `ab/__init__.py` sets `__path__ = pkgutil.extend_path(__path__, __name__)` (before `from ab.nn import *`). This is what allows several distributions that each own a subtree of `ab/` to be importable together: previously every such wheel shipped its own `ab/__init__.py`, so the last one installed replaced the others and made `import ab.nn` or the sibling package fail depending on install order. `extend_path` merges those directories into one search path instead of letting one shadow the rest.

### Changed
- `Const.nn_path` is re-anchored from `ab/nn/__init__.py` to `ab/nn/util/__init__.py` (`Path(init_file.__file__).parent.parent` instead of `.parent`). Both resolve to the same `ab/nn` directory, but the old anchor assumed `ab/nn/__init__.py` sits at a fixed place in the installed tree, which namespace packaging no longer guarantees. Anchoring on `ab/nn/util`, a file this distribution always ships itself, keeps `nn_path` correct regardless of how the `ab` namespace is assembled from several packages.
- `setup.py` distribution name `NN-Dataset` → `nn-dataset` (version `1.0.3` → `1.0.4`), so the installed name matches the PyPI project users actually type rather than a differently-cased alias.
- Startup banner reworded from "LEMUR operates within the root directory" to "LEMUR uses the root directory", which reports the resolved root without implying the tool is installed there.

### Fixed
- Path resolution no longer depends on `ab.nn.__init__` being present at a fixed location, the assumption the namespace-package change invalidates.

### Removed
- The PyPI-only `nn-stat` install instruction, replaced by the Git install flow; a standalone PyPI `nn-stat` does not by itself join the `ab` namespace, so the old step misled users.

## [1.0.3] - 2025-02-08

### Added
- `data(..., cast_prm=True)` on `ab/nn/api.py`, forwarded to `Read.data`. The `prm` table stores every hyperparameter as a string next to a type tag, and `data()` always rebuilt it with `getattr(builtins, pr_type)(pr_value)`. That rebuild assumes the tag names a Python builtin that can re-parse the stored text, which is not guaranteed for every row. The flag keeps the typed result as the default but lets a caller request the raw column value, so code that compares or hashes stored hyperparameters sees exactly what is in the database rather than a reconstruction of it.
- `default_nn_name = 'AlexNet'` in `Const.py`, with `default_nn_path` derived from it, so the default architecture is named once instead of a separate literal filename.

### Changed
- `torch~=2.5.1`/`torchvision~=0.20.1` → `torch~=2.6.0`/`torchvision~=0.21.0`. torchvision releases are pinned to a matching torch release, so the two must move together; the bump advances the pair to the then-current matched set instead of leaving torch and torchvision on a stale combination.
- Project-root discovery rewritten. The old `__project_root_path()` checked only `Path().absolute().parent` and returned `'..' * len(to_nn)`, so it assumed the process ran at one exact depth inside `ab/` and resolved the wrong root from any other working directory. The new version walks up from the cwd until it finds a directory containing both `ab/` and `README.md`, returns that absolute path, and stops at the filesystem root so it cannot loop forever. It also prints the root it selected, so a wrong checkout fails visibly instead of silently writing stats into the wrong tree.
- Package metadata: name `nn-dataset` → `NN-Dataset`, version 1.0.2 → 1.0.3, and the project URL points at `https://ABrain.one`. The `nn-plots` extra becomes `stat = ['nn-stat']`, the name that matches the statistics package the extra actually installs.

### Fixed
- `test.py` passes `cast_prm=False` and filters by `default_nn_name`, so it checks the raw stored hyperparameters and the default model instead of depending on the casting path and a hard-coded name.

### Removed
- `LICENSE.md` in favour of `LICENSE`.

### Documentation
- Install/update instructions rewritten; license files renamed extension-less.

## [1.0.2] - 2025-02-06

### Added
- PyPI version badge and PyPI-first install instructions, because `pip install nn-dataset` is now the supported distribution channel and users need a visible version reference and the CUDA wheel index (`--extra-index-url`); the GitHub install flow is kept below as the alternative, and `rm -rf db` is called out because a stale local database silently shadows the freshly installed data. The new `nn-plots` extra makes the plotting dependencies opt-in, so analysis tooling no longer forces matplotlib's install cost on every user.
- License documents are now shipped inside the package: `MANIFEST.in` adds `LICENSE.md` and `Doc/Licenses/*.md`, so the MIT, BSD-3-Clause and DeepSeek-LLM-V2 notices covering models adopted from other projects travel with every sdist/wheel and are not lost when the source is redistributed.

### Changed
- README license links point at absolute GitHub URLs instead of repo-relative paths, because relative links do not resolve on the PyPI project page.
- Version 1.0.1 → 1.0.2.

### Removed
- `pyproject.toml`, reverting to `setup.py`-only packaging: after the 1.0.1 torch-pin removal the file had shrunk to a three-line `[build-system]` block whose only effect was to declare `pip` itself as a build requirement, which is redundant and can interfere with isolated PEP 517 builds, so keeping a second packaging manifest only invited drift from the real metadata in `setup.py`.

## [1.0.1] - 2025-02-05

### Changed
- `setup.py` no longer installs PyTorch itself, and the `[tool.pip_install]` torch pin in `pyproject.toml` is gone with it. The removed `install_pytorch()` ran `pip install torch~=2.5.1` from the CUDA 12.4 index on every build or install, which reached the network as a side effect of packaging and forced one cu124 build onto every user — breaking or mis-installing on CPU-only hosts and on other CUDA versions. PyTorch is now resolved through normal dependency metadata, so the target environment chooses the correct build, which is the only reliable way to support the range of machines the package is installed on.
- Version 1.0.0 → 1.0.1, so the release carrying the installation fix can be distinguished on PyPI.

### Fixed
- ICNet.py built its SGD optimizer with `lr=prm['ls']`, a typo for `prm['lr']`. The `ls` key is never supplied — `supported_hyperparameters()` advertises `lr` — so model construction raised `KeyError` and ICNet could not be trained at all; reading `lr` restores the intended learning-rate search.

### Added
- README links to the public LEMUR repository, so the source and the released package can be found from each other.

### Removed
- The unused module-level `args = []` in `AirNet.py`, which nothing read and which only risked shadowing the framework's own argument handling.

## [1.0.0] - 2025-02-05

Major release that turns the benchmark from a CLI-only, file-scattered experiment runner into a library other programs can query and extend.

### Added
- Programmatic API (`ab/nn/api.py`): `data(...)` returns the database contents as a pandas DataFrame (model source, transform source, hyperparameters and stats), and `check_nn(nn_code, task, dataset, metric, prm, save_to_db=True)` trains a model supplied as a source string and stores it. Before this, results could only be produced through the CLI, so no external code could read the benchmark or contribute a model without knowing the on-disk layout; one documented function for reading and one for contributing is now the public surface.
- SQLite persistence layer (`ab/nn/util/db/`: `Init.py`, `Write.py`, `Read.py`, `Calc.py`), replacing the per-config `trials.json` files that had to be discovered by listing `ab/nn/stat/*`. Model, metric and transform source live in their own code tables so identical code is stored once and joined back onto each stat row, and hyperparameters are stored as name/value/type triples so native Python types survive the round trip instead of being flattened to strings.
- `data(only_best_accuracy=True)` computes the highest-accuracy row per `(task, dataset, metric, nn, epoch)` in SQL, so callers no longer load every trial and filter in Python; indexes on `accuracy` and the join columns keep those queries cheap.
- Task-specialised COCO loaders (`coco_/Detection.py`, `coco_/Segmentation.py`) and additional classification loaders (CelebA-gender, CIFAR-10/100, MNIST, SVHN, ImageNette, Places365): a single `coco.py` had grown to do two unrelated jobs, and every loader now follows the same `(out_shape, minimum_accuracy, train_set, test_set)` return shape that `Train` expects.
- mAP metric (`ab/nn/metric/map.py`) so the newly added detection models can be scored.
- Model source checker (`ab/nn/util/CodeEval.py`): runs Pylint, Radon cyclomatic complexity and a dynamic smoke test (instantiate `Net`, call `train_setup`, run `learn` on random tensors) and folds the results into a 0–100 score. Because contributed models are arbitrary user code, the project needed an automatic way to reject broken or unmaintainable submissions rather than relying on manual review.
- `pyproject.toml`, a `test.py` suite exercising `api.data()`/`api.check_nn()`, and helpers `Classes.py` (`DataRoll`), `Exception.py`, `soft_pad.py`.

### Changed
- Architectures relocated `ab/nn/dataset/` → `ab/nn/nn/` and made to obey one contract — `Net(in_shape, out_shape, prm, device)`, `train_setup(prm)`, `learn(train_data)` and `supported_hyperparameters()`. Model classes previously lived beside dataset code and each task was trained its own way, so the trainer had to special-case architectures (e.g. inline RNN/LSTM construction). With the uniform interface `Train`/`check_nn` can build, train and evaluate any model generically, and `supported_hyperparameters()` lets Optuna search only the parameters a model actually declares.
- `Train.py` rewritten around `optuna_objective`, `train_n_eval` and persistence of `duration`/`accuracy`, with retry/backoff, so one failing trial no longer loses the whole run.
- CLI extended with many flags and the config separator changed `-` → `_` to match the names stored in the database.

### Fixed
- `MIoU` accepts `out_shape`; the `minimum_accuracy`/`AccuracyException` gate stops wasted training; CUDA OOM triggers a batch rollback; `release_memory()` frees GPU memory between trials.

### Removed
- The old `ab/nn/dataset/` tree, `ab/nn/util/Stat.py` (directory-scanning config discovery, superseded by the SQL queries in `db/Read.py`), the single-file `coco.py` bodies, `cifar10.py` and `transform/cifar10_*.py`.

### Documentation
- README rewritten with the LEMUR overview, CLI usage, contribution rules and citation, plus MIT attributions for the imported models.

## [0.1.2] - 2024-12-22

### Added
- Config discovery as a reusable util (`ab/nn/util/Stat.py`: `extract_all_configs`, `provide_all_configs`), because it had lived inside `train.py`, where only `train.py`'s own CLI could reach it; extracting it lets any entry point discover configs, and `extract_all_configs` now also creates a missing `stat/<config>` directory for a complete config (via the new `Util.is_full_config`: five `-`-separated parts and the model module present), so a config that has not been trained yet is queued instead of silently skipped.

### Changed
- Path resolution became automatic (`Util.define_global_paths()`), and the `-r/--from_root` flag with the `Const.from_root_g` global is gone: previously every module had to consult that flag to know whether a path was relative to the project root or to `train.py`, which made the caller decide something the program can decide itself. `define_global_paths()` now probes the cwd for `stat/`, recognising the source layout (`ab/nn/stat`), the installed layout (`.venv/Lib/site-packages/ab/nn/stat`) and a bare run directory, and writes the result once to `Const.stat_dir_global` / `dataset_dir_global` / `data_dir_global`; loaders thus find `data/` and the model/dataset dirs the same way wherever the process starts.
- Loaders honour that resolved directory instead of hardcoding it: `cifar10.py` uses `Const.data_dir_global` instead of a literal `root='data'`, and `coco.py` derives its `coco/` root from it and drops the `class_list`/`path` parameters, so `train.py` no longer imports `coco` specially to pass `img_segmentation` settings and the loader works on its own.
- `run.py` and the CLI are driven by `argparse` (`--config`, `--epochs`, `--trials`, `--max_batch_binary_power`) rather than in-file constants, so a run is configured from the command line without editing the script; config parsing is centralised in `Util.conf_to_names`, and `train.main()` loses the removed `from_root` argument.
- `setup.py` version 0.1.0 → 0.1.2.

### Fixed
- Config banner wording and a COCO comment typo ("doesn't exists" → "doesn't exist").

### Removed
- `-r/--from_root` and `default_from_root`; unused imports in `train.py`; `ab/nn/stat/README.md`.

## [0.1.1] - 2024-12-21

Initial release. A framework for benchmarking neural-network architectures by combining datasets, transforms, and metrics, driven by Optuna search — so a model's quality is reported as a comparable metric value from many small training runs instead of one hand-tuned result. There is no `ab/nn/api.py` yet; the public surface is `ab/nn/train.py`, `ab/nn/util/*`, `ab/nn/loader/*`, `ab/nn/metric/*`, `ab/nn/transform/*`, and 32 `Net` model modules.

### Added
- **Single CLI entry point** `ab/nn/train.py` (`main(...)`, Optuna `study.optimize(objective)`, evaluation at 1/2/5 epochs): one command sweeps a whole (task, dataset, transform, model) grid, so recipes can be rerun and compared without bespoke scripts; the short epoch ladder gives cheap indicative scores before any expensive full run.
- **Config discovery** `extract_all_configs()`/`provide_all_configs()` scanning `ab/nn/stat/*`: the existing result directory *is* the experiment registry, so a run resumes the grid a previous run left behind and nothing must be declared in a separate manifest. At this tag the helpers live in `train.py`; they move to `util/Stat.py` in 0.1.2.
- **Resumable trials** `save_results()` appending to `trials.json`, with `count_trials_left()`: Optuna progress survives restarts, so a crashed or interrupted sweep continues instead of re-paying completed trials, and trials can be topped up with `+N`.
- **Universal training harness** `ab/nn/util/Train.py` (`Train` with SGD + `CrossEntropyLoss`, grad-norm clipping, per-group LRs for segmentation): every architecture trains through one loop, so results reflect the model rather than a per-model recipe; the segmentation LR boost lets backbone-based nets train their heads without a separate script.
- **Dynamic loading utilities** `ab/nn/util/Loader.py` (`load_dataset`) and `ab/nn/util/Util.py` (`nn_mod`, `get_attr`, `args`): datasets, transforms, metrics and models are imported by dotted name at call time, so the grid is data-driven and a new component appears simply by adding a file.
- **Loaders** `cifar10.py`, `coco.py` (`COCOSegDataset`), `wikitext.py` (`TextDatasetPreparation`): classification, segmentation and character-level text generation are covered from the start, spanning the task families the benchmark is meant to compare.
- **Metrics** `acc.py`, `iou.py`: a classification and a segmentation score share one call contract, so different tasks aggregate into the same comparison table.
- **Transforms** `cifar10_norm.py`, `cifar10_norm_32.py`, `cifar10_complex.py`, `echo.py`: inputs are normalised for pretrained backbones, `cifar10_complex` feeds complex-valued nets, and `echo` is the identity baseline that separates model accuracy from transform effects.
- **32 architectures** under `ab/nn/dataset/` (ResNet, VGG, ConvNeXt, SwinTransformer, VisionTransformer, MaxVit, FCN, UNet, DeepLabV3, LRASPP, RNN, LSTM, ComplexNet, BayesianNet, …): a broad, pre-implemented library is what makes the benchmark useful — a contributor adds a model, not a whole pipeline.
- **Packaging** `setup.py`, `setup.cfg`, `MANIFEST.in`, `requirements.txt`, `run.py`: pip-installability plus a runnable example let external contributors reproduce results on their own hardware.

### Documentation
- README (install, Docker, usage, model-contribution criteria, citation); loader/metric/transform READMEs; licensing split in `LICENSE.md` and `Doc/Licenses/`: the contribution criteria fix the contract a new model must satisfy, while the license split keeps third-party model attributions explicit.
