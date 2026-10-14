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

The cost of a training run is not only its GPU-hours. It is also the
attention it demands, and the trust that researchers can place in what it
leaves behind. `nn-dataset` is built so none of these costs falls on the researcher:
**Learning:** it trains the model within
a per-epoch budget forecast from the first batches of each epoch, adapts when
the run fails, and keeps the epochs a
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

An experiment with neural-network code starts before the first epoch. The
submission must first prove that it trains, that it does not waste compute, and
that its results stay attached to the code that produced them. Runs that fail
for reasons the system could have handled spend GPU time anyway, and what was
learned from failed or interrupted runs is usually discarded. Renamed or
reformatted copies of one architecture are hard to recognize, so collections
fill with duplicates. None of these problems is solved by training more, and
none is confined to one project. Related problems are documented when
machine-learning runs are reproduced [@henderson2018rl], compared
[@bouthillier2021variance], and maintained [@sculley2015debt]. What is missing
is the procedure that covers the whole life of a run. Before training, it must
report the quality of a submission. During the run, it must keep the run within
a time limit, adjust the run when it fails, and record what happens while it is
running, from per-epoch health signals to the machine it used. After the run,
the store must keep the code and its measurements together, so the work can be
rebuilt, compared across machines, and reused later. The time limit, the
recovery from memory failures, the retries, the accuracy floor, the code check,
the per-layer measurements, the source-structure naming, and the per-machine
timing reports are the parts of that procedure, and a run without them returns
less research for the same GPU time.

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

**One interface for every model.** The package is written in Python (>= 3.10)
and runs every submission through the same three steps, training, measurement,
and description. The model defines only how it trains. The package does the
rest. To connect the two, a submission follows one rule, taken from the model
template of [@goodarzi2025lemur]: a PyTorch [@paszke2019pytorch] file that
declares its supported hyperparameters beside a class named `Net`, gives that
class `train_setup` and `learn` methods, and is constructed as
`Net(in_shape, out_shape, prm, device)`. Those names and that one signature are
the whole interface, and they tell the package what to build, how the model
trains, and which values it may vary. Because the names are the same in every
model, the package can load any file the same way, prepare the device and data
loaders it needs, and write the results into the same store that keeps the
model, all through one call, `check_nn` in `ab/nn/api.py`, with the description
step running as part of the campaign. So adding a model is a matter of adding
a file, not changing the pipeline.

**Bounded, self-repairing runs.** The design bounds cost per epoch rather than
per study, so one slow model cannot hide inside an average. Training runs
under a per-epoch time limit of 30 minutes by default. After the first five
batches the package divides the elapsed time by the batches processed and
projects how long the full epoch will take, and it stops the run once that
projection exceeds the limit. The limit is an input rather than a constant, so
a researcher can tighten it to study how networks behave under a smaller
compute budget. A stopped run keeps its completed epochs and
returns a small positive score, above a failed trial and below a completed
one, so the search keeps partial-run signal without paying for a restart.
Recoverable failures change the run instead of ending it. A CUDA
out-of-memory error steps the batch size down to the next lower power of two
and restarts that model's search. Other exceptions are retried up to thirty
times per model, and the counter resets after a result that passes the
accuracy check. Every completed epoch of a run that stores its results is
written to a JSON file. The file records the losses and accuracy of that
epoch, gradient norm, and throughput, together with the run's maximum number
of epochs, the best accuracy so far and the epoch that reached it, and the
machine's type, capacities, and CPU, RAM, and GPU usage, so each run leaves
the measurements that later statistical analysis needs. The hyperparameter search itself remains the Optuna-driven
procedure of [@goodarzi2025lemur], and what is added here is the bounding and
recovery around each trial. Each checkpoint is accompanied by a small JSON
file that records its best score, so a later run that resumes from the
checkpoint replaces it only when the new score is higher.

**Acceptance and health signals.** Before training, the package exercises the submission: it imports
the file, builds `Net`, calls `train_setup`, and runs `learn` on two small
random batches, recording a pass or the error; a
related acceptance check is described with the generator work
[@kochnev2026nngpt]. The package then scores the code with
Pylint [@pylint2024] and Radon [@radon2024], combining lint findings and complexity into one score, and
returns it with the training result; the check is advisory, and a broken
submission fails at its own import. Earlier work logged accuracy, duration,
GPU utilization, and the code revision [@goodarzi2025lemur;
@kochnev2026nngpt]; this package additionally stores gradient norm, throughput,
and CPU, RAM, and
GPU-memory use together with the machine's identity, so timings stay
attributable to the hardware that made them. A run whose accuracy does not exceed 1.2 times the task's random-guess
level is refused and not stored, because a chance-level row carries no signal;
the caller still receives the failure and recoverable failures are retried.
Caption tasks, whose
scores move slowly, are exempt from the floor and keep
training. Optionally, it measures per-layer properties — weight
spectra, effective rank, redundancy, dead units, Taylor
importance, and zero-out sensitivity.
Trained models can be rebuilt from their records and timed without further
training: warm-up passes, repeated CPU and GPU runs, per-machine minimum
(fastest), maximum, and standard-deviation times, in a batch mode that skips
measured models, so an inventory of finished models can gain measurements at
any later time.

**Description and naming from the source.** Stored model code can be described automatically: FLOPs and memory
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
not this one. Before training, a candidate can be checked
against a name index built from stored code: a repeat costs a lookup
rather than a training run. Different code with similar text can
be found through a MinHash measure [@broder1997syntactic].
This differs from `torch.fx` [@reed2022torchfx], which traces an instantiated
module, whereas the name comes from source text alone. The naming rules are covered by 47 structured tests in
`test/test_arch_uid.py` — 21 edits that must keep one name, 20 that must change
it, and 6 graded cases for the structural similarity measure — supplemented by
a fuzzer that rewrites real
models with both edit kinds.

**A derived store and a checked archive.** The per-run files are the primary
record, and the database and the published snapshot are derived from them, so
a lost database is an inconvenience, not a loss. If it is missing, the package
rebuilds the store from the per-run files or a compressed snapshot. Runs resume:
reissuing a command runs only the missing trials.
Runs that prune trained models are
recorded alongside training runs, and old model files keep training, because
the loop takes the loss from wherever the model
provides it. A repository script keeps the store and the published snapshot
in step: it verifies recorded code by hash, writes an audit sheet, and deletes
local files only after compression and upload pass in the same run, so the
published snapshot never runs ahead of what was verified.

# Research impact statement

The LEMUR papers present the datasets themselves, and the training and
measurement runs behind those datasets came from this package. The LEMUR
dataset [@goodarzi2025lemur] and its successor corpus LEMUR 2 with over 14,000
architectures [@uzun2026lemur2] were trained and measured with it, and the
NNGPT generator [@kochnev2026nngpt] trains and scores every generated model
the same way.

Four further studies have already used the package. A systematic evaluation of
25 learning-rate schedules across 30 architectures found the choice to be
architecture dependent and contributed the resulting accuracy landscape back
to the dataset [@mateen2026scheduling]. A study of all 18 loss and optimizer
pairings across 33 architectures and six datasets showed that no single
pairing is universally optimal [@abramochkin2026lossoptimizer]. A delta-based
generation study trained 1,100 candidates per language model and confirmed
that first-epoch rankings predict the rankings of full training
[@adhikari2026delta]. A retrieval-augmented pipeline validated 941 of 1,289
candidate blocks from 19 repositories, supplying the majority of novel
architectures in the LEMUR dataset [@khalid2025nnrag].

Development has been public since October 2024 under the MIT licence, with
more than thirty tagged releases, a thousand-plus commits, a changelog, a
contribution guide, and tests on Python 3.10 and 3.11. The software is
released on PyPI as `nn-dataset` (alias `lmur`) with around 30,000 downloads
in total. Snapshots and checkpoints are shared on Hugging Face
(https://huggingface.co/NN-Dataset), and the container image
`abrainone/ai-linux` documents a reproducible environment.

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
