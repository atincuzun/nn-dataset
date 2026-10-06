---
title: 'LEMUR: a framework for standardized training, evaluation, and querying of neural-network programs'
tags:
  - Python
  - deep learning
  - neural architecture search
  - AutoML
  - benchmarking
  - reproducibility
  - PyTorch
authors:
  - name: Tolgay Atinc Uzun
    affiliation: 1
  - name: Waleed Khalid
    affiliation: 1
  - name: Saif U Din
    affiliation: 1
  - name: Sai Revanth Mulukuledu
    affiliation: 1
  - name: Akashdeep Singh
    affiliation: 1
  - name: Chandini Vysyaraju
    affiliation: 1
  - name: Raghuvir Duvvuri
    affiliation: 1
  - name: Avi Goyal
    affiliation: 1
  - name: Yashkumar Rajeshbhai Lukhi
    affiliation: 1
  - name: Muhammad A. Hussain
    affiliation: 1
  - name: Krunal Jesani
    affiliation: 1
  - name: Usha Shrestha
    affiliation: 1
  - name: Yash Mittal
    affiliation: 1
  - name: Roman Kochnev
    affiliation: 1
  - name: Pritam Kadam
    affiliation: 1
  - name: Mohsin Ikram
    affiliation: 1
  - name: Harsh Rameshbhai Moradiya
    affiliation: 1
  - name: Alice Arslanian
    affiliation: 1
  - name: Dmitry Ignatov
    affiliation: 1
  - name: Radu Timofte
    affiliation: 1
affiliations:
  - name: Computer Vision Lab, CAIDAS & IFI, University of Würzburg, Germany
    index: 1
date: 3 October 2026
bibliography: paper.bib
---

# Summary

LEMUR (distributed as the Python package `nn-dataset`, also published on PyPI as
`lmur`) is an open-source framework that turns neural-network programs into
standardized, executable, and queryable records. It is the software layer of the
LEMUR project. A user writes a model file that implements a small contract, or submits model
source code as a string; the framework trains it under a common orchestration
protocol, records code, hyperparameters, per-epoch metrics, resource usage, and
architecture statistics in a local SQLite archive, and exposes the results
through a pandas-based query API. Models can be exported to ONNX (TensorFlow Lite
for selected models), and the archive can be synchronized with Hugging Face.

The contribution is the **software infrastructure** to automatically generate model corpus statistics for any compliant model in an extensible manner. The package provides a program contract that makes a
conforming PyTorch model executable, a training and evaluation pipeline with
hyperparameter search, a queryable archive, and APIs for reading past experiments
and submitting new code. A separate subsystem derives a structural identity from
each model's source code for deduplication and similarity; the pipeline and its
APIs turn model validation into a reusable, programmatic service.

`nn-dataset` is intended for researchers who generate, search over, or compare
neural-network programs — for example, neural architecture search, AutoML, and
LLM-based code generation — and for anyone who needs to run and record many
model/hyperparameter combinations. It is written in Python and installable from
PyPI; training runs on an NVIDIA GPU with CUDA, while the database, API, and
analysis tools run on CPU.

# Statement of need

Work that generates or searches over neural-network code needs three capabilities
that are rarely available together: a way to execute conforming model code under
a common orchestration protocol, a way to compare results across architectures,
hyperparameters, and tasks, and a way to retrieve past results programmatically so
that later runs can build on them.

`nn-dataset` addresses this need directly. The imposed executability makes a model assessable by the same
harness; its training pipeline applies one orchestration protocol to every
submission, while the model itself supplies its optimizer and training step; its
archive stores code and measurements together under a single schema; and its
API lets external programs — statistical models, search loops, or LLM agents —
read validated results and submit new candidates. This is what allows a generator
such as the companion NNGPT framework [@kochnev2026nngpt] to close the loop from
generation to training to improved generation.

# State of the field

Model repositories and zoos (`torchvision`, `timm` [@wightman2019timm])
provide implementations and weights, not a standardized training-and-query
service with a queryable store of per-experiment results. NAS benchmarks [@ying2019nasbench101; @dong2020nasbench201;
@dong2021natsbench] offer strong reproducibility over a fixed, enumerable search
space, but do not accept arbitrary new programs; benchmark suites
[@mattson2020mlperf] standardize a fixed evaluation set.

Experiment-tracking and HPO infrastructure is the nearest neighbor: MLflow
[@zaharia2018mlflow], Weights and Biases
[@biewald2020wandb], and Ray Tune [@liaw2018tune] log runs and manage
experiments, and Optuna [@akiba2019optuna] provides search. Ray Tune executes user-supplied training functions, and MLflow associates each
run's parameters, metrics, and artifacts. `nn-dataset` builds on that idea for
neural-network programs: a model-class contract, a submission endpoint that
evaluates arbitrary code under a fixed policy, and a queryable store that keeps
each program's source, hyperparameters, and measurements together with a
source-derived structural identity.

`nn-dataset` is built around an API and a queryable
history: it introduces a small interface rather than a search space; `data()` returns
the stored code, hyperparameters, and measured accuracy of past configurations;
and `check_nn()` trains new code and returns comparable measurements. This
submission-and-measurement integration of source code, evaluation records, and
structural identity is what `nn-dataset` adds. Its
architecture-identity subsystem also differs from default `torch.fx` tracing of an
instantiated `nn.Module` [@reed2022torchfx], which captures the forward graph of a
constructed object, whereas identity here is derived from source text without
execution. A related project, NNGPT [@kochnev2026nngpt], is a generator that
uses the LEMUR pipeline, where it benefited from the separation of automatic assessment and the data store of the executable network codes, as these had been employed extensively due to their modularized nature. 

# Software design

`nn-dataset` is a Python package (installable as `nn-dataset`, Python >= 3.10).
The `ab.nn` package is organized around a small number of components.

* **Program contract.** Every model implements one interface. A model is a single
  file that declares its supported hyperparameters and provides a `Net` class
  [@paszke2019pytorch] implementing the training step. Because the model supplies its own optimizer and training loop, the
  execution is standardized around measurement and logging rather than the
  training algorithm itself. The same contract applies whether a model is added as
  a file or submitted as a string, so generated and hand-written models are
  treated identically, therefore allowing different investigational networks to be manually added.
* **Registries.** Loaders, metrics, transformations, and models are resolved by
  name at runtime. The package ships 15 dataset loaders, 18 metrics, and more
  than 200 transformation algorithms spanning vision and text tasks; new
  components are added by placing a file in the corresponding directory, so the
  framework extends uniformly to new tasks.
* **Training pipeline.** `ab.nn.train` and `ab.nn.util.Train` run a model for a
  task/dataset/metric configuration. Hyperparameters are searched with Optuna
  [@akiba2019optuna] over ranges supplied by the run configuration, while the
  model declares which hyperparameters it supports; training uses a configurable
  per-epoch time threshold, defaulting to 30 minutes, checked through runtime
  estimates during data iteration; per-epoch metrics, learning rate, gradient norm,
  throughput, and system and GPU usage are recorded, and failures are captured
  rather than aborting the batch.
* **Archive and schema.** Results are stored in a local SQLite database that keeps
  code, hyperparameters, and measurements as related records. The database can be
  initialized from the file system or downloaded from Hugging Face, and writes are
  serialized with a file lock on Unix-like systems (with in-process locking
  elsewhere) so that concurrent training jobs do not corrupt it.
* **Query API.** `ab.nn.api.data()` returns recorded experiments as a
  `pandas.DataFrame`, so earlier results can be analysed and used to select
  candidates for the next run.
* **Validation API.** `ab.nn.api.check_nn()` is the submission endpoint: it
  accepts model source code, a hyperparameter dictionary, and the task context,
  runs the full training pipeline, archives the result when it passes the
  acceptance check, and returns the measured metrics.
* **Statistics, quality, and identity.** Architecture statistics (parameter
  counts, FLOPs estimate, model size, depth, and heuristic
  architecture-family flags) and layer-level analyses are stored. A separate
  subsystem derives an architectural similarity from each model's source using an AST
  def-use graph and Weisfeiler--Lehman refinement, without executing it, for
  deduplication. A code-quality score uses Pylint and Radon
  complexity. Models can be exported to ONNX, and the archive can be synchronized
  with Hugging Face.
* **Contribution protocol and tests.** The project documents a review workflow
  in which a new model is accompanied by statistics from ten Optuna trials
  (three for very large models) over short training runs; a test suite in `test/`
  exercises the database and API.

Several trade-offs are deliberate. The program contract enables uniform
evaluation and direct comparison of generated and hand-written models, but every
model must follow the interface. A single SQLite file keeps the archive
portable, but concurrent writers must be serialized by the write lock. Storing
code and measurements in one schema enables query-driven search at the cost of a
growing repository, so the archive is also distributed through Hugging Face.
Resolving components by name keeps extension and search cheap. The archive records each
run's task/dataset/metric configuration, model-declared hyperparameters, and
measurements alongside the code, so results are compared across models under a
shared, recorded protocol; the database, API, and analysis tools run on CPU,
while training runs on an NVIDIA GPU with CUDA.

# Research impact statement

`nn-dataset` is used by the NNGPT project [@kochnev2026nngpt], a separate
framework whose loop calls `data()` and
`check_nn()`. Several studies run the framework and record their results in it:
the learning-rate-scheduling study [@mateen2026scheduling] evaluated 3,938 variants
and contributed its accuracy landscape to LEMUR; the loss-optimizer study
[@abramochkin2026lossoptimizer] generated 594 variants by source-level injection;
delta-based NAS [@adhikari2026delta] fine-tuned code LLMs on LEMUR architectures;
and NN-RAG [@khalid2025nnrag] contributed about 72% of the novel architectures in
LEMUR. The LEMUR papers [@goodarzi2025lemur; @uzun2026lemur2] describe the corpus
itself. The corpus and checkpoints are released through Hugging Face
(https://huggingface.co/NN-Dataset). The repository (https://github.com/ABrain-One/nn-dataset) has been public since
October 2024 with about sixty contributors (see its public contributor graph),
published on PyPI (currently 2.2.11, with 2.2.13 in the repository), and
documented for the public `abrainone/ai-linux` Docker image.

# AI usage disclosure

Generative AI is used as a component of this software: models and
transformations in the archive may be produced by LLM-based generators such as
NNGPT. Separately, DeepSeek v4.1 Flash was used during development of the
software, its documentation, and the drafting of this paper such as the documentation and the copy-editing. Human authors reviewed, edited, and validated all
AI-assisted output, tested the software, and made the core design decisions.

# Acknowledgements

We thank the contributors to the `nn-dataset` project and the developers of the
LEMUR dataset and its ecosystem. This work was partially supported by the
Alexander von Humboldt Foundation; the foundation had no role in the design,
implementation, or reporting of the work.

# References
