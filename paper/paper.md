---
title: 'nn-dataset for training and analysing neural networks'
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

`nn-dataset` is an open-source Python package for training neural networks and
analysing their results. Researchers supply model code and select a task,
dataset, and evaluation measure. The package records evaluation scores, training
losses, hardware information, and resource use. Optional analyses measure
properties of individual layers and compare model source code.

The package supports studies of network architectures and training procedures.
It can evaluate researcher-written models and models produced by code generators.
Users can add models, data loaders, preprocessing methods, and evaluation
measures. The software is distributed under the MIT licence at
https://github.com/ABrain-One/nn-dataset [@nndataset2026].

# Statement of need

Comparing neural networks requires records of their training settings and
behaviour, not only their final scores. Differences in learning procedures,
memory requirements, and hardware affect interpretation. Studies of benchmark
variability [@bouthillier2021variance] and software maintenance
[@sculley2015debt] support the need for consistent recording and shared code.
Researchers studying training procedures need to keep model-specific learning
code while measuring losses, gradients, running time, and layer properties.
Implementing these measurements separately in each study repeats work and makes
the recording procedures harder to compare.

`nn-dataset` provides a common model interface, training and retry procedures,
and a database of measurements. Researchers can evaluate their own model code
without selecting a model from the published collection or using a generator.

The number of runs grows as studies vary models, datasets, and training settings.
Common records and explicit selection criteria support comparisons at this scale
and allow completed experiments to be reused for follow-up questions.

# State of the field

Model libraries such as `timm` [@wightman2019timm] provide implementations and
weights. NAS-Bench-101 [@ying2019nasbench101] and NAS-Bench-201
[@dong2020nasbench201] provide results for defined collections of architectures.
`nn-dataset` accepts model source code that follows its interface, including
code for the learning procedure.

MLflow [@zaharia2018mlflow] and Weights and Biases [@biewald2020wandb] record
parameters, metrics, and model files. Ray Tune [@liaw2018tune] schedules
user-defined trials and handles failures. Optuna [@akiba2019optuna] searches
hyperparameters. PyTorch Lightning [@falcon2019lightning] manages training loops,
logging, and checkpoints through model methods and callbacks.

Using these tools to compare submitted implementations requires additional code
for source comparison, selected layer measurements, and queries that connect
these outputs to experimental settings. Tracking tools organize run outputs,
while search systems optimize reported objectives. Task-specific layer analyses
and the rules for comparing implementations require experiment-specific
definitions. `nn-dataset` provides this combination through
a common model interface, shared analysis routines, and linked database tables.
It reuses PyTorch and Optuna for execution and search. Keeping measurement and
selection code in the package lets different studies reuse the same procedures
while retaining each model's learning method. General-purpose framework hooks
can support equivalent extensions when researchers provide these components.

# Software design

Shared execution and recording support comparisons across many models, datasets,
and training settings. Recorded conditions support rechecking results and
extending completed experiments.

**Model interface.** The model interface and core training framework retain
the original design [@goodarzi2025lemur]. A PyTorch [@paszke2019pytorch] file
declares supported hyperparameters and provides a `Net` class with training
setup and learning methods. Model modules own the loss, optimizer, and learning
steps. The common engine handles data loading, evaluation, and recording. This
separation lets different training procedures reuse measurement code while keeping
their scientific choices in ordinary Python files. Reruns with a study's data
and settings support checks of published results, while a declared common
protocol supports comparisons across implementations. Loaders, transformations,
and evaluation measures are separate modules, so changes to preprocessing can
use the same result format. The `check_nn` API accepts researcher-supplied source.
Model modules run as trusted Python code with the process's permissions.

**Training control.** The API evaluates caller-selected settings. The training
script coordinates Optuna trials, retries, and reductions in the range of batch
sizes searched following CUDA out-of-memory errors. Keeping these decisions in
the trial runner separates search management from each model's learning method.
An iterator estimates epoch duration from early training batches and interrupts
learning when the forecast exceeds the budget. This lets researchers select
configurations by expected training cost. Existing epoch records remain available
after interruptions.
Advisory Pylint [@pylint2024] and Radon [@radon2024] reports and a brief test
using synthetic image batches provide feedback on code issues, initialization,
and learning steps before longer runs.

**Training and layer measurements.** Epochs provide a common recording unit.
With recording enabled, losses, scores, gradient norm, estimated processing rate,
and best score are stored with experiment settings and hardware information.
Host CPU/RAM samples and GPU allocation statistics accompany the scores and
settings. These observations help researchers assess performance together with
computational requirements and choose feasible experiments for available
hardware. Hardware and batch-size information retain the conditions in which
running-time and memory measurements were taken.

Optional layer analysis records gradient norms, the fraction of activations near
zero, layer importance estimated from weights and gradients
[@molchanov2017pruning], effective rank [@roy2007effectiverank], and output
similarity through centered kernel alignment
[@kornblith2019cka]. It also measures loss changes after zeroing layer weights
and collects WeightWatcher statistics [@martin2021weightwatcher] when its optional
dependency is installed. Selected batches and supported convolutional and linear
layers define the analysis scope. Different learning curves and layer properties
can accompany similar final scores. Recording both lets researchers investigate
these differences and design follow-up experiments. Making these calculations
optional lets users choose where to spend additional forward passes, gradient
calculations, and matrix operations. Layer records retain the experiment
identifier, so their measurements can be read alongside the corresponding scores
and settings.
Model summaries add parameter counts, layer types, module nesting depth, and
operation estimates for covered layers. Configuration identifiers connect these
summaries to the settings used to construct the model, supporting analysis of
size and computational requirements under specified conditions.

**Code comparison.** Grouping candidate source matches helps researchers compare
related implementations and examine which code differences accompany changes
in recorded performance. Source comparison uses a Python syntax graph, allowing
fingerprints to be calculated without constructing a model or loading its data.
Graph labels are refined using the Weisfeiler-Leman method
[@weisfeiler1968reduction] and hashed into an identifier. Selected renamings,
formatting changes, and constant expressions are normalized. This source-based
choice differs from tracing a model object with `torch.fx` [@reed2022torchfx]
and permits code screening before training. Fingerprints summarize the supported
structural features and identify matches for source inspection. Comparing model
behaviour also uses the source and runtime settings, while trained weights remain
separate from the code identifier. An index utility lets callers compare new
fingerprints with stored ones. Persisted MinHash signatures
[@broder1997syntactic] provide graded text similarity and retrieve related model
code without recomputing signatures from its full text for each query.

**Record selection.** SQLite tables link source code, configurations, results,
model summaries, and layer measurements through identifiers. Queries return
pandas DataFrames [@mckinney2010pandas], allowing analysis with existing statistical
and plotting tools. A shared table format lets completed runs support new
questions across projects while retaining their experimental conditions.
Pairwise queries can match task, dataset, evaluation measure, and epoch count
while varying the model, making comparison criteria explicit in the retrieval
step.

Additional selection routines require chosen result fields, model descriptors,
or hyperparameters to be present. They also expand parameter dictionaries into
table columns. For example, an analysis can select records with available
parameter counts, scores, and learning rates at a specified epoch. These options
give researchers a stated rule for constructing an analysis table when
measurements differ across tasks or older runs. Returning parameters as columns
supports grouping, regression, and plotting without repeated unpacking of
configuration dictionaries.

**Storage and sharing.** Per-epoch JSON exports support inspection and file-based
exchange. SQLite reuses source and configuration entries across results and
supports queries spanning runs. Retaining the same experimental fields in both
formats lets later studies inspect, query, and extend completed work after the
original training jobs have ended. Database rebuilding uses retained
code and measurement files or a compatible compressed snapshot. Continuing a
search uses completed-trial counts to determine additional work. Reproducing a
particular run also requires its inputs, code, software environment, and random
states.

The archive script inventories models, statistics, and transformations, then
records their associations, covered database checks, and proposed file actions
in an Excel audit report. Compression and upload precede cleanup, which depends
on eligibility flags and successful audit, compression, and upload stages.
Recording the verification method and proposed action makes storage decisions
inspectable and separates file management from model training.

# Research impact statement

The original model interface and training framework were introduced with LEMUR
[@goodarzi2025lemur]. The package also supplied evaluation software for the
LEMUR 2 collection [@uzun2026lemur2] and NNGPT's generated implementations
[@kochnev2026nngpt]. Applications by the contributing research group include
a study of 25 learning-rate schedules across 30 architectures
[@mateen2026scheduling] and 18 loss-optimizer pairings across 33 architectures
and six datasets [@abramochkin2026lossoptimizer]. Studies of model generation
through code changes [@adhikari2026delta] and extraction of network code from
repositories [@khalid2025nnrag] also used the evaluation procedures. These uses
document the reuse of training and query components for different experimental
questions.

PePy reported 31,992 cumulative PyPI downloads of `nn-dataset` as of
15 October 2026, including automated downloads [@pepy2026downloads].

Development history dates to October 2024. Source version 2.2.13 includes the
training and analysis modules. The repository provides installation and extension
instructions, tagged releases, a changelog, examples, and automated tests.
The package requires Python 3.10 or newer. Training uses CUDA, while queries of
stored records and source fingerprinting support CPU-based analysis.
Shared database copies and saved model weights are available at
https://huggingface.co/NN-Dataset.

# AI usage disclosure

DeepSeek V4.1 Flash was used during earlier software development, documentation,
and paper drafting. Human authors reviewed those outputs and made the core
design decisions. Kilo Code, using GPT-6.1 Sol, supported literature comparison,
source inspection, and text revision. Human review and validation are required
before submission, with authors responsible for accuracy, originality, and
licensing. External code generators can supply models for experiments. Their
use as experimental inputs is separate from AI-assisted software and manuscript
preparation.

# Acknowledgements

Contributions to `nn-dataset` are gratefully acknowledged. The Alexander von
Humboldt Foundation provided partial support. The foundation had no role in the
design, implementation, or reporting of the work.

# References
