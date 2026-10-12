---
title: 'nn-dataset: the experiment harness of the LEMUR project for training and measuring submitted neural-network code'
tags:
  - Python
  - deep learning
  - neural architecture search
  - AutoML
  - experiment management
  - reproducibility
  - PyTorch
authors:
  - name: Tolgay Atinc Uzun
    corresponding: true
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

`nn-dataset` is a self-contained Python package that turns a neural-network
program submitted as code into a trained, measured, and named record. It is
available at https://github.com/ABrain-One/nn-dataset [@nndataset2026].

The cost of a training run is not only its GPU hours. It is also the
attention it demands, and the trust that research can place in what it leaves
behind. `nn-dataset` is built so none of these costs falls on the researcher:
**Learning:** it trains the model within a per-epoch budget,
forecast from the first batches of each epoch, adapts when the run fails, and
keeps the epochs a
stopped run completed. **Evaluation:** it grades the submission's code with standard
Python linters, refuses runs that do no better than random guessing, and keeps
per-epoch health signals such as gradient stability and processing speed. **Modeling:** it
describes the model from its own code and names it from the source structure, so
renamed or rewritten copies of one architecture are recognized as the same
model. The researcher designs the experiments; the package supervises them.

The package is intended for researchers who generate, search over, or compare
neural-network programs — for example, neural architecture search, AutoML, and
LLM-based code generation. Training runs on an NVIDIA GPU; the database and
analysis tools run on CPU. It is pip-installable under the MIT licence.

# Statement of need

Research that generates or searches over neural-network code produces candidates
faster than they can be trained. Testing candidates one at a time spends GPU time
on runs that fail for reasons the system could have handled, and what was
learned from failed or interrupted runs is usually discarded. Renamed or
reformatted copies of one architecture are hard to recognize, so collections
fill with duplicates. What is missing is a training
procedure that reports the quality of a submission before training, keeps every
run within a time limit, adjusts the run when it fails, and stores enough about
both the run and the code that the work remains usable. The package provides this
procedure — candidates are stopped, repaired, or skipped before wasting GPU
time — and generators such as NNGPT [@kochnev2026nngpt] use it to train and
measure what they write. The dataset papers and the generator work
describe the template, the corpora, and the generation methods
[@goodarzi2025lemur; @uzun2026lemur2; @kochnev2026nngpt]; the time-limited run,
recovery from memory failures, retries, the accuracy floor, code scoring,
per-layer measurements, source-structure naming, and per-machine CPU/GPU timing reports are described
here.

# State of the field

Model repositories and zoos (`torchvision`, `timm` [@wightman2019timm]) provide
implementations and weights, but they do not train or record submitted code; NAS
benchmarks [@ying2019nasbench101; @dong2020nasbench201; @dong2021natsbench]
and benchmark suites [@mattson2020mlperf] evaluate fixed spaces rather than
submitted programs.

Experiment trackers and hyperparameter search are the nearest neighbours: MLflow
[@zaharia2018mlflow] and Weights and Biases [@biewald2020wandb] record runs, Ray
Tune [@liaw2018tune] manages experiments with time budgets and retries, and
Optuna [@akiba2019optuna] searches hyperparameters. The difference here is a
submitted program: the package exercises it before training, projects an epoch's
cost from the first batches and stops it before it overruns, halves the batch
after an out-of-memory failure, and names the code so that a rewritten copy
matches the model it came from. These behaviors sit inside the training loop,
which a tracker leaves to the user's code, so the package was built instead of
extending one. They cannot be patched into a
tracker, because a submission must be exercised, repaired, and named before it
runs.

# Software design

The package is written in Python (>= 3.10) and follows one procedure from
submission to stored result: training, measurement, and description. A submission is one PyTorch [@paszke2019pytorch]
file against the model template of [@goodarzi2025lemur]: a `Net` class
with `train_setup`, `learn`, and a module-level declaration of its supported
hyperparameters. Submitted code runs in the caller's environment; the operator
chooses the container or machine boundary.

**Learning.** Generated and hand-written models follow the same path.
Training runs under a per-epoch time
limit, 30 minutes by default: after a few batches the package estimates the full
epoch from current throughput and stops an epoch that would exceed it.
A stopped run keeps its completed epochs and returns a small positive score,
above a failed trial and below a completed one, so the search keeps
partial-run signal without paying for a restart. Recoverable failures change the run instead of ending it: a
CUDA out-of-memory error halves the batch size by powers of two and restarts
the search; other exceptions are retried up to thirty times per model, the
counter resetting after a good result. The hyperparameter search
remains the Optuna-driven procedure of [@goodarzi2025lemur]; described here is
the bounded, recovering wrapper around it. Saved weights
carry their best score in a sidecar
record, so a later run that resumes from the checkpoint replaces it only when
it beats the recorded best.

**Evaluation.** Before training, the package exercises the submission: it imports
the file, builds `Net`, calls `train_setup`, and runs `learn` on two small
random batches, recording a pass or the error; a
related acceptance check is described with the generator work
[@kochnev2026nngpt]. The package then scores the code with
Pylint [@pylint2024] and Radon [@radon2024], combining lint findings and complexity into one score, and
returns it with the training result; the check is advisory, and a broken
submission fails at its own import. Earlier work logged accuracy, duration,
GPU utilization, and the code revision [@goodarzi2025lemur;
@kochnev2026nngpt]; this package additionally stores gradient norm, throughput,
and CPU, RAM, and GPU-memory use together with the machine's identity. A run whose accuracy does not exceed 1.2 times the task's random-guess
level is refused and not stored, because a chance-level row carries no signal;
the caller still receives the failure and recoverable failures are retried.
Caption tasks, whose
scores move slowly, are exempt from the floor and keep
training. Optionally, it measures per-layer properties — weight
spectra, effective rank, redundancy, dead units, Taylor
importance, and zero-out sensitivity — per layer.
Trained models can be rebuilt from their records and timed without further
training: warm-up passes, repeated CPU and GPU runs, per-machine minimum
(fastest), maximum, and standard-deviation times, in a batch mode that skips
measured models.

**Modeling.** Stored model code can be described automatically: FLOPs and memory
footprint from one forward pass over the main layer types, plus depth, layer
composition, and architecture patterns such as residual or transformer-like
families; parameter counts and layer details are already part of the recorded
model data [@goodarzi2025lemur; @uzun2026lemur2]. The code is named from its
structure: a graph of definitions and uses is refined with Weisfeiler--Lehman
[@weisfeiler1968reduction] and
hashed without executing the file, so renamed variables, reformatting, and folded
constants such as `256*6*6` versus `9216` keep one name, and a second file of the
same architecture attaches its measurements to the existing entry. The name is
computed with a fixed hash over exact parameter values, so one file keeps one
name across machines. Earlier work rejects repeated source text by hashing it
[@uzun2026lemur2; @kochnev2026nngpt]; a renamed copy passes such a check but
not this one. Before training, a candidate is checked
against a name index built from stored code: a repeat costs a lookup
rather than a training run. Different code with similar text can
be found through a MinHash measure [@broder1997syntactic].
This differs from `torch.fx` [@reed2022torchfx], which traces an instantiated
module, whereas the name comes from source text alone. The naming rules are covered by 47 structured tests in
`test/test_arch_uid.py` — 21 edits that must keep one name, 20 that must change
it, and 6 graded cases for the structural similarity measure — supplemented by
a fuzzer that rewrites real
models with both edit kinds.

**Unified research.** If the store is missing, the package rebuilds it from
per-run files or a compressed snapshot. Campaigns resume:
reissuing a command runs only the missing trials.
Runs that prune trained models are
recorded alongside training runs, and old model files keep training, because
the loop takes the loss from wherever the model
provides it. A repository script keeps the store and the published snapshot
in step: it verifies recorded code by hash, writes an audit sheet, and deletes
local files only after compression and upload pass in the same run.

# Research impact statement

The package is the training and
measurement layer for seven publications from our group. Three are built on it
directly: the LEMUR (Learning, Evaluation, and Modeling for Unified Research)
dataset [@goodarzi2025lemur], whose successor corpus LEMUR
2 — over 14,000 architectures [@uzun2026lemur2] — was produced through this
pipeline, and the NNGPT generator [@kochnev2026nngpt], which trains and scores
every generated model through it. Four further studies relied on its time limits,
failure handling, and run recording:
learning-rate scheduling [@mateen2026scheduling], loss--optimizer pairing
[@abramochkin2026lossoptimizer], delta-based NAS [@adhikari2026delta], and
retrieval-augmented generation from model code [@khalid2025nnrag]. Companion
packages build on it: `nn-gpt` depends
on `nn-dataset`, and `nn-stat` ships as an optional extra. Development has been
public since October 2024 under the MIT licence, with more than thirty tagged
releases, a thousand-plus commits, a changelog, a contribution guide, and tests
on Python 3.10 and 3.11. The software is
released on PyPI as `nn-dataset` (alias `lmur`; ~190 PyPI downloads in
September 2026); snapshots and checkpoints are shared on Hugging
Face (https://huggingface.co/NN-Dataset), and the container image
`abrainone/ai-linux` documents a reproducible environment. Releases are
archived on GitHub; a citable DOI is minted on acceptance.
All confirmed uses so
far are inside our group; the package is public, installable, and documented for
others to build on.

# AI usage disclosure

Generative AI is used as a component of this software: models and transformations
in the store may be produced by LLM-based generators such as NNGPT
[@kochnev2026nngpt]. Separately, DeepSeek v4.1 Flash was used during the
development of this software, its documentation, and the draft of this paper.
Human authors reviewed, edited, and validated all AI-assisted output, tested the
software, and made the design decisions.

# Acknowledgements

We thank the contributors to the `nn-dataset` project and its community. This work was partially
supported by the Alexander von Humboldt Foundation; the foundation had no role in
the design, implementation, or reporting of the work.

# References
