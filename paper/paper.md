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

`nn-dataset` is the Python package behind the framework named Learning,
Evaluation, and Modeling for Unified Research [@goodarzi2025lemur]. It provides
the execution layer used to build the neural-network dataset
[@uzun2026lemur2] and by the NNGPT generator [@kochnev2026nngpt]; those works
describe the model template, the collected and generated networks, and the
generation methods. This paper describes the software itself: how a submitted
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
LLM-based code generation — and for anyone who needs to train and record many
model/hyperparameter combinations. It is installable from PyPI; training runs on
an NVIDIA GPU, while the database and analysis tools run on CPU.

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
to train and measure what they write.

# State of the field

Model repositories and zoos (`torchvision`, `timm` [@wightman2019timm]) provide
implementations and weights, but they do not train or record submitted code. NAS
benchmarks [@ying2019nasbench101; @dong2020nasbench201; @dong2021natsbench]
offer reproducible evaluation over a fixed, enumerable search space, but they do
not accept new programs; benchmark suites [@mattson2020mlperf] standardize a
fixed evaluation set.

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
file whose `Net` class takes `(in_shape, out_shape, prm, device)`, implements
`train_setup(prm)` and `learn(train_data)`, and whose module defines
`supported_hyperparameters()`; this is the current form of the model template
introduced in [@goodarzi2025lemur]. Submitted code runs in the same Python process
as the package, which provides no sandbox; isolation is whatever container or
machine the operator provides.

**Learning.** The same path applies whether the model arrives as a file or as a
string, so generated and hand-written models are treated alike. Training runs under a per-epoch time
limit, 30 minutes by default: after a few batches the package estimates the full
epoch from the current throughput and stops an epoch that would exceed the limit.
A stopped run is not discarded: it keeps the epochs already completed and enters
the hyperparameter search with a reduced score, ranking below completed runs but
above failed ones. Recoverable failures change the run instead of ending it. A
CUDA out-of-memory error lowers the batch size by powers of two and restarts the
search with the new limit; other exceptions are retried up to thirty times per
model, and the counter resets after a good result. Hyperparameters are searched
with Optuna [@akiba2019optuna].

**Evaluation.** Before training, the package exercises the submission: it imports
the file, builds `Net` with a dummy input and output shape, calls `train_setup`,
and runs `learn` on two small random batches, recording a pass or the error. The
NNGPT generator describes a related check: compilation, forward correctness, and
a short CIFAR-10 run [@kochnev2026nngpt]. The package then scores the code with
Pylint and Radon and returns the score to the caller with the training result; the check does not by itself block training, because the trainer imports
the same file and fails there if the submission is broken. Accuracy and duration
are the fields the dataset work records [@goodarzi2025lemur]; the NNGPT generator
reported loss, accuracy, GPU utilization, and the Git commit of each run
[@kochnev2026nngpt]; this package stores per-epoch train and test loss, gradient
norm, throughput, and CPU, RAM, and GPU-memory use with machine details. A run whose accuracy does not exceed 1.2 times the task's random-guess
level is refused and not stored, because a chance-level row carries no signal;
the outcome still reaches the caller, time-stopped runs keep the epochs
completed, and recoverable failures are retried. Caption tasks continue training
instead of being refused at this level. Optionally, during the first epochs and
periodically after that, the package also measures per-layer properties — weight
spectra, effective rank, redundancy between layers, dead units, Taylor
importance, and sensitivity when a layer is zeroed — and records them per layer.
Trained models can be reloaded and timed without retraining: warm-up passes,
repeated CPU and GPU runs, and mean, minimum, maximum, and standard deviation per
machine written to a report; a batch mode skips models already measured.

**Modeling.** Stored model code can be described automatically: FLOPs and memory
footprint from one forward pass that records each layer's work, plus depth, layer
composition, and architecture patterns such as residual or transformer-like
families; parameter counts and layer details are already part of the recorded
model data [@goodarzi2025lemur; @uzun2026lemur2]. The code is named from its
structure: a graph of definitions and uses is refined with Weisfeiler--Lehman and
hashed without executing the file, so renamed variables, reformatting, and folded
constants such as `256*6*6` versus `9216` keep one name, and a second file of the
same architecture attaches its measurements to the existing entry. Earlier work removes duplicates by
normalizing the source text and hashing it [@uzun2026lemur2; @kochnev2026nngpt]. Different code with similar text can
be found through a MinHash similarity measure. This differs from tracing a
constructed module with `torch.fx` [@reed2022torchfx], which captures the forward
graph of an instantiated object, whereas the name here is derived from source
text alone. The suite covers the naming rules — 47 tests in
`test/test_arch_uid.py`: 21 edits that must keep one name, 20 that must change
it, and 6 graded similarity cases — the identity used in file names, the query
filters, and the write-and-init paths; the repository workflow runs it on Python
3.10 and 3.11.

**Unified research.** If the store is missing, the package rebuilds it from the
stored per-run files or downloads a compressed snapshot, and snapshots can be
uploaded for use on another machine. Campaigns resume: the store counts the
trials already done, so reissuing a command runs only the missing trials, and
asking for more trials is a normal input. Old model files continue to train as
the package evolves, because the training loop takes the loss from wherever the
model provides it. A repository script keeps the store and the published snapshot
in step: it verifies recorded code by hash, writes an audit sheet, compresses and
uploads the database, and deletes local files only after every check passes in
the same run.

# Research impact statement

The package is the shared training and measurement layer for six efforts in our
group: the NNGPT generator [@kochnev2026nngpt], the dataset pipelines
[@goodarzi2025lemur; @uzun2026lemur2], and studies on learning-rate scheduling
[@mateen2026scheduling], loss--optimizer pairing [@abramochkin2026lossoptimizer],
delta-based NAS [@adhikari2026delta], and retrieval-augmented generation from
model code [@khalid2025nnrag]. Each runs models through the same pipeline and
relies on the package for time limits, failure handling, and run recording
instead of reimplementing them. Separate packages build on it: `nn-gpt` depends
on `nn-dataset`, and `nn-stat` ships as an optional extra. The software is
released on PyPI (`nn-dataset`, also `lmur`; latest release 2.2.11, about 190
downloads in the last month) and maintained in the public repository
(https://github.com/ABrain-One/nn-dataset), open since October 2024 with about
forty contributors, twenty stars, and thirty-five forks; database snapshots and
checkpoints are shared on Hugging Face (https://huggingface.co/NN-Dataset), and
the container image `abrainone/ai-linux` documents the package for reproducible
environments. We know of no project outside our group that uses it yet.

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
