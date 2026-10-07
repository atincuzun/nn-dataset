---
title: 'Learning, Evaluation, and Modeling for Unified Research: a Python package for training and measuring submitted neural-network code'
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

`nn-dataset` is the reference implementation of the framework named Learning,
Evaluation, and Modeling for Unified Research [@goodarzi2025lemur]: it turns a
neural-network program submitted as code into a trained, measured, and named
record. The neural-network dataset [@uzun2026lemur2] and the NNGPT generator
[@kochnev2026nngpt] were built on it; those works describe the model template,
the collected and generated networks, and the generation methods. This paper describes the software itself: how a submitted
model is scored before training, trained within a time limit, recovered when it
fails, described from its own code, and stored with the machine it used and the
measurements it produced.

Three questions decide whether a submitted model is useful for research: can it
be trained, how well does it perform, and what exactly is it? The package answers
all three and keeps the answers together. **Learning:** it trains the program
under a per-epoch time limit, adjusts the run when it fails, and can stop and
continue later. **Evaluation:** it scores the submission's code with Pylint and
Radon, refuses runs that stay near the task's random-guess level, and keeps
per-epoch signals such as gradient norm and throughput. **Modeling:** it
describes the model from its own code and names it from the source structure, so
renamed or rewritten copies of one architecture are recognized as the same
model. A researcher can start a campaign and leave it: failed runs are adjusted
or retried, stopped runs keep the epochs they finished, and the store can be
rebuilt or moved to another machine, so the results outlast a single run.

The package is intended for researchers who generate, search over, or compare
neural-network programs — for example, neural architecture search, AutoML, and
LLM-based code generation. It is installable from PyPI and released under the MIT
licence; training runs on an NVIDIA GPU, while the database and analysis tools
run on CPU.

# Statement of need

Research that generates or searches over neural-network code produces candidates
faster than they can be trained. Testing candidates one at a time spends GPU time
on runs that fail for reasons the system could have handled, and what was learned
from failed or interrupted runs is usually discarded. Repeated architectures are
also hard to recognize once the code has been renamed or reformatted, so
collections fill with copies of the same model. What is missing is a training
procedure that reports the quality of a submission before training, keeps every
run within a time limit, adjusts the run when it fails, and stores enough about
both the run and the code that the work remains usable. The package provides this
procedure, and what used to be discarded — failed runs, stopped runs, repeated
architectures — stays usable; generators such as NNGPT [@kochnev2026nngpt] use it
to train and measure what they write. The dataset papers and the generator work
describe the template, the corpora, and the generation methods
[@goodarzi2025lemur; @uzun2026lemur2; @kochnev2026nngpt]; the time-limited run,
recovery from memory failures, retries, the accuracy floor, code scoring,
per-layer measurements, source-structure naming, and per-machine CPU/GPU timing reports are described
here.

# State of the field

Model repositories and zoos (`torchvision`, `timm` [@wightman2019timm]) provide
implementations and weights, but they do not train or record submitted code, and
NAS benchmarks [@ying2019nasbench101; @dong2020nasbench201; @dong2021natsbench]
and benchmark suites [@mattson2020mlperf] evaluate fixed spaces and fixed test
sets rather than submitted programs.

Experiment trackers and hyperparameter search are the nearest neighbours: MLflow
[@zaharia2018mlflow] and Weights and Biases [@biewald2020wandb] record runs, Ray
Tune [@liaw2018tune] manages experiments with time budgets and retries, and
Optuna [@akiba2019optuna] searches hyperparameters. The difference here is a
submitted program: the package exercises it before training, projects an epoch's
cost from the first batches and stops it before it overruns, halves the batch
after an out-of-memory failure, and names the code so that a rewritten copy
matches the model it came from. These behaviors sit inside the training loop,
which a tracker leaves to the user's code, so the package was built instead of
extending one.

# Software design

The package is written in Python (>= 3.10) and follows one procedure from
submission to stored result: training, measurement, and description, matching the
three parts of its name. A submission is a single PyTorch [@paszke2019pytorch]
file written against the model template of [@goodarzi2025lemur]: a `Net` class
with `train_setup`, `learn`, and a module-level declaration of its supported
hyperparameters, in its current form. Submitted code runs in the caller's own
environment; the operator chooses the container or machine boundary.

**Learning.** Files and strings follow the same path, so generated and
hand-written models are treated alike. Training runs under a per-epoch time
limit, 30 minutes by default: after a few batches the package estimates the full
epoch from the current throughput and stops an epoch that would exceed the limit.
A stopped run is not discarded: it keeps the epochs already completed and
returns a small positive score, ranking above a failed trial and below a
completed one. Recoverable failures change the run instead of ending it. A
CUDA out-of-memory error lowers the batch size by powers of two and restarts the
search with the new limit; other exceptions are retried up to thirty times per
model, and the counter resets after a good result. Saved weights carry their
best score beside them, so a later run replaces a checkpoint only when it beats
the recorded best.

**Evaluation.** Before training, the package exercises the submission: it imports
the file, builds `Net` with a dummy input and output shape, calls `train_setup`,
and runs `learn` on two small random batches, recording a pass or the error. A related acceptance check is described with the generator work
[@kochnev2026nngpt]. The package then scores the code with
Pylint and Radon, combining lint findings and complexity into one score, and
returns it with the training result; the check is advisory, and a broken
submission fails at its own import. Earlier work recorded different fields [@goodarzi2025lemur; @kochnev2026nngpt]; this
package stores per-epoch train and test loss, gradient norm, throughput, and
CPU, RAM, and GPU-memory use with machine details. A run whose accuracy does not exceed 1.2 times the task's random-guess
level is refused and not stored, because a chance-level row carries no signal;
the caller still receives the failure, time-stopped runs keep the epochs
completed, and recoverable failures are retried. Caption tasks, whose early scores move slowly, are exempt from the floor and keep
training. Optionally, during the first epochs and
periodically after that, the package also measures per-layer properties — weight
spectra, effective rank, redundancy between layers, dead units, Taylor
importance, and sensitivity when a layer is zeroed — and records them per layer.
Trained models can be rebuilt from their records and timed without further
training: warm-up passes, repeated CPU and GPU runs, per-machine mean, minimum,
maximum, and standard deviation, and a batch mode that skips models already
measured.

**Modeling.** Stored model code can be described automatically: FLOPs and memory
footprint from one forward pass over the main layer types, plus depth, layer
composition, and architecture patterns such as residual or transformer-like
families; parameter counts and layer details are already part of the recorded
model data [@goodarzi2025lemur; @uzun2026lemur2]. The code is named from its
structure: a graph of definitions and uses is refined with Weisfeiler--Lehman and
hashed without executing the file, so renamed variables, reformatting, and folded
constants such as `256*6*6` versus `9216` keep one name, and a second file of the
same architecture attaches its measurements to the existing entry. The name is
computed with a fixed hash over exact parameter values, so one file keeps one
name on every machine. Earlier work rejects repeated source text by hashing it
[@uzun2026lemur2; @kochnev2026nngpt]. Before training, a candidate can be checked
against a name index built from stored code: a collision is reported as an
earlier stored entry or a repeat within the same run, so a repeat costs a lookup
rather than a training run. Different code with similar text can
be found through a MinHash similarity measure. This differs from `torch.fx` [@reed2022torchfx], which traces an instantiated
module, whereas the name comes from source text alone. The naming rules are covered by 47 tests in
`test/test_arch_uid.py`: 21 edits that must keep one name, 20 that must change
it, and 6 graded similarity cases. Other tests cover the identity in file names, queries, and the write path; the
repository's tests workflow is configured for Python 3.10 and 3.11. A fuzzer
supplements the suite: it rewrites real models with edits that must keep the name
and edits that must change it, so both failure kinds surface.

**Unified research.** If the store is missing, the package rebuilds it from the
stored per-run files or downloads a compressed snapshot, and snapshots can be
uploaded for use on another machine. Campaigns resume: the store counts the
trials already done, so reissuing a command runs only the missing trials, and
asking for more trials is a normal input. Runs that prune trained models are
recorded alongside training runs, so a reduced model stays comparable with its
source. Old model files continue to train as
the package evolves, because the training loop takes the loss from wherever the
model provides it. A repository script keeps the store and the published snapshot
in step: it verifies recorded code by hash, writes an audit sheet, compresses and
uploads the database, and deletes local files only after every check passes in
the same run.

# Research impact statement

The package is the shared training and measurement layer for six research efforts
in our group: the NNGPT generator [@kochnev2026nngpt], the dataset pipelines
[@goodarzi2025lemur; @uzun2026lemur2], and studies on learning-rate scheduling
[@mateen2026scheduling], loss--optimizer pairing [@abramochkin2026lossoptimizer],
delta-based NAS [@adhikari2026delta], and retrieval-augmented generation from
model code [@khalid2025nnrag]. Each runs models through the same pipeline and
relies on the package for time limits, failure handling, and run recording
instead of reimplementing them. Companion packages build on it: `nn-gpt` depends
on `nn-dataset`, and `nn-stat` ships as an optional extra. Development has been
public since October 2024 under the MIT licence, with more than thirty tagged
releases, over a thousand commits, a changelog, a contribution guide, a code of
conduct, and a tests workflow on Python 3.10 and 3.11. The software is
released on PyPI (`nn-dataset`, also `lmur`; about 190 downloads in September
2026 (pypistats.org); snapshots and checkpoints are shared on Hugging
Face (https://huggingface.co/NN-Dataset), and the container image
`abrainone/ai-linux` documents a reproducible environment. All confirmed uses so
far are inside our group; the package is public, installable, and documented for
others to build on.

# AI usage disclosure

Generative AI is used as a component of this software: models and transformations
in the store may be produced by LLM-based generators such as NNGPT
[@kochnev2026nngpt]. Separately, DeepSeek v4.1 Flash was used during the
development of the software and its documentation and in drafting this paper.
Human authors reviewed, edited, and validated all AI-assisted output, tested the
software, and made the design decisions.

# Acknowledgements

We thank the contributors to the `nn-dataset` project and its community. This work was partially
supported by the Alexander von Humboldt Foundation; the foundation had no role in
the design, implementation, or reporting of the work.

# References
