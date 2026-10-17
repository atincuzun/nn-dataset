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
losses, hardware information, and resource use. Additional analyses measure
properties of individual layers and compare model source code.

The package supports studies of network architectures and training procedures.
It can evaluate researcher-written models and models produced by code generators.
Users can add models, data loaders, preprocessing methods, and evaluation
measures. The software is distributed under the MIT licence at
https://github.com/ABrain-One/nn-dataset [@nndataset2026].

# Statement of need

Hundreds of different neural network structures are produced in the domain of computer science, targeted to specific tasks. This introduces the problem of reimplementing the repetitive code across independent research. Scientist must write the similar loops to train a neural network. Even though this is still a manageable problem, many ideas need to be tried, tested and compared, producing a growing number of results that must be stored together with the 
model code and training settings used to obtain them.

Comparing neural networks requires records of their training settings and
behaviour, not only their final scores. Differences in learning procedures,
memory requirements, and hardware affect interpretation. Studies of benchmark
variability [@bouthillier2021variance] and software maintenance
[@sculley2015debt] support the need for consistent recording and shared code.
Researchers studying training procedures need to keep model-specific learning
code while measuring losses, gradients, running time, and layer properties.
Implementing these measurements separately in each study repeats work and makes
the recording procedures harder to compare.

`nn-dataset` provides a common model interface, training and retry procedures, similarity checking,
and a database for keeping the measurements. Researchers can automatically evaluate their own model code
without manual intervention.

The number of runs grows as studies vary models, datasets, and training settings.
Common records and explicit selection criteria support comparisons at this scale
and allow completed experiments to be reused for follow-up questions.

# State of the field

Model libraries such as `timm` [@wightman2019timm] provide implementations and
weights. NAS-Bench-101 [@ying2019nasbench101] and NAS-Bench-201
[@dong2020nasbench201] provide results for defined collections of architectures. 
MLflow [@zaharia2018mlflow] and Weights and Biases [@biewald2020wandb] record
parameters, metrics, and model files. Ray Tune [@liaw2018tune] schedules
user-defined trials and handles failures. Optuna [@akiba2019optuna] searches
hyperparameters. PyTorch Lightning [@falcon2019lightning] manages training loops,
logging, and checkpoints through model methods and callbacks.

Using these tools for submitted implementations requires additional code
for source comparison, selected layer measurements, and queries that connect
these outputs to experimental settings. Trackng tools organize run outputs,
while search systems optimize reported objectives. Task-specific layer analyses
and the rules for comparing implementations riequire experiment-specific
definitions. `nn-dataset` provides this through shared analysis routines, and linked database tables.
It reuses PyTorch and Optuna for execution and search. Keeping measurement and
selection code in the package lets different studies reuse the same procedures
while retaining each model's learning method. General-purpose framework hooks
can support equivalent extensions when researchers provide these components.

# Software design

`nn-dataset` was designed with modularity in mind. Data interface, processing,  and had been kept
separate. The analytical routines use this foundation to relate the implementations to their
learning behaviour and computational requirements under recorded conditions.

**Data Interface** PyTorch [@paszke2019pytorch] model modules contain their
loss, optimizer, learning steps and hyperparameters while the engine handles data preparation,
evaluation, and recording under a declared common
protocol. Compatible code written under the interface helps the unification of the pipeline while allowing the user defined custom training procedures. Importantly, this permits decoupling networks structure, learning and evaluation loops, allowing researchers to narrow down the scope of their work. Model files are expected to be Python files that provide a small set of functions. These constraints have been described in the LEMUR dataset such as `Net(nn.Module)` naming and `train_setup(self, prm)`, `learn(self, train_data)` functions.

A complete training cycle also requires a dataset, a transformation, and an evaluation measure. Beyond the model file naming, the same principles are applied to these components. For example, a dataset file provides `loader(transform_fn, task)`, a transformation file provides `transform(norm)`, and an evaluation measure file provides `create_metric(out_shape)`. Therefore, researchers can easily write their own datasets, transformations, or measures following the same convention and reuse the full training engine, focusing on particular aspects like structural network-level modifications rather than execution code.


**Dynamic Imports**
Aside from the definitions, the name based dynamic imports were extensively used to compose the system. Experiment configurations select models, dataset loaders, transformations, and
evaluation measures by name. The corresponding Python modules are imported at
runtime, and their expected classes or functions are obtained through the common
interfaces. New compatible modules can therefore be selected for experiments
without rewriting the central execution procedure.

This separates experimental choices from the code that coordinates a run.
Researchers can compare implementations or preprocessing methods while reusing
evaluation and recording logic. For model, metric, and transformation modules,
the loading helper can recover stored source from the database when a module
file is missing. Imported code runs with the Python process's permissions, so
the selected modules and their dependencies must be available and trusted.




Trial retries and batch-size search bounds belong to the runner rather than
the learning method. The training iterator estimates epoch duration from early
batches and stops iteration when the forecast exceeds the budget. Checks occur
between batches, so this is a forecast-based cost rule, not a hard limit on total
process time. Existing epoch records remain available after interruption.
Advisory Pylint [@pylint2024] and Radon [@radon2024] reports and a synthetic-image
execution check provide earlier feedback before spending time on longer runs.

**Learning and execution.**

**Measurement resolution.** Recording at epoch boundaries limits the volume of
stored observations across many trials. It retains learning progress at coarser
resolution than batch-level profiling, with settings linked to each observation.
Hardware descriptions and batch size accompany host CPU/RAM samples and CUDA
allocation statistics, making their measurement context available when comparing
computational requirements.

Similar final scores can conceal different learning curves and layer behaviour.
Optional analysis therefore adds importance estimates from activations and their
gradients [@molchanov2017pruning], effective rank [@roy2007effectiverank],
representation similarity [@kornblith2019cka], relative loss changes after zeroing
layer weights and biases, and WeightWatcher statistics [@martin2021weightwatcher].
Sampling selected epochs and batches limits the additional passes and matrix
operations. The resulting
trade-off is more internal information at additional cost, with coverage defined
by supported convolutional and linear layers. These observations guide further
experiments rather than automatically explaining why one model performs better.

**Code relationships.** Exact-text duplicate checks in earlier pipelines
[@uzun2026lemur2] and [@kochnev2026nngpt] address repeated submissions. Comparing
related implementations additionally requires recognizing selected rewritings.
Source-graph fingerprints normalize supported renamings and constant expressions
using Weisfeiler-Leman refinement [@weisfeiler1968reduction]. Unlike tracing an
instantiated model with `torch.fx` [@reed2022torchfx], source fingerprints can be
computed without model construction or data loading. MinHash signatures
[@broder1997syntactic] provide graded text similarity for retrieving candidate variants.

The choice of source analysis reduces execution requirements but also defines
the interpretation of a match. A fingerprint is a structural summary, not proof
of architectural or behavioural equivalence. Source inspection, runtime settings,
and trained weights remain necessary for interpreting model differences.
The fingerprint and structural-similarity routines have 47 tests covering selected
renamings, constant expressions, structural changes, and similarity cases.

**Persistent comparisons.** SQLite links available source, configurations,
results, and layer observations so that comparisons can select matching
experimental conditions and required measurements. Returning pandas DataFrames
[@mckinney2010pandas] keeps statistical analysis separate from execution, allowing
completed runs to be reused when research questions change.

The inherited dual storage serves different access patterns. JSON exports remain
readable and exchangeable, while SQLite avoids repeating stored source and
configuration entries across results. Keeping both requires synchronization
between files and the query store. Retained sources and measurements or compatible
snapshots can rebuild that store. Search continuation counts completed trials
rather than restoring optimizer or search-sampler state. Reproducing individual
runs additionally requires their retained code, data preparation, software
environment, and random states.

# Research impact statement

The neural network dataset that is introduced with LEMUR
[@goodarzi2025lemur] was gathered using `nn-dataset` package. The package also granted extension and evaluation software for the
LEMUR 2 collection [@uzun2026lemur2] and supplied API to NNGPT
[@kochnev2026nngpt]. Applications of the research angle include
a study of 25 learning-rate schedules across 30 architectures
[@mateen2026scheduling] and 18 loss-optimizer pairings across 33 architectures
and six datasets [@abramochkin2026lossoptimizer]. Studies of model generation
through code changes [@adhikari2026delta] and extraction of network code from
repositories [@khalid2025nnrag] also used the evaluation procedures. These uses
document the reuse of training and query components for different experimental
questions.

PyPI reported 31,992 cumulative PyPI downloads of `nn-dataset` as of
15 October 2026, including automated downloads [@pepy2026downloads].

Development history dates to October 2024. Source version 2.2.13 includes the
training and analysis modules. The repository provides installation and extension
instructions, tagged releases, a changelog, examples, and automated tests.
The package requires Python 3.10 or newer. The engine prefers CUDA with MPS and
CPU fallbacks. Stored-record queries and source fingerprinting also run on CPU.
Shared database copies and saved model weights are available at
https://huggingface.co/NN-Dataset.

# AI usage disclosure

DeepSeek V4.1 Flash was used during earlier software development, documentation,
and paper drafting. Human authors reviewed those outputs and made the core
design decisions. Kilo Code, using GPT-6.1 Sol, assisted literature comparison,
source inspection, restructuring, and drafting. MiMo V2.6 Pro, DeepSeek V4.1
Flash, GLM 5.3, and Step 5 Preview provided informal manuscript feedback and
technical checks. Author-provided feedback from Claude was also considered,
with its model version unspecified. Human review, editing, and validation are
required before submission, with authors responsible for accuracy, originality,
and licensing.
External code generators can supply experimental models. Their use as inputs is
separate from AI-assisted software and manuscript preparation.

# Acknowledgements

Contributions to `nn-dataset` are gratefully acknowledged. The Alexander von
Humboldt Foundation provided partial support. The foundation had no role in the
design, implementation, or reporting of the work.

# References
