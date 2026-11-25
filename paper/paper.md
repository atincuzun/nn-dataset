---
title: 'LEMUR: A Unified Python Framework for Reproducible Neural Network Experiments at Scale'
tags:
  - Python
  - deep learning
  - neural architecture search
  - AutoML
  - experiment management
  - reproducibility
  - PyTorch
authors:
  - name: Dmitry Ignatov
    affiliation: 1
  - name: Tolgay Atinc Uzun
    corresponding: true
    affiliation: 1
  - name: Radu Timofte
    affiliation: 1
affiliations:
  - name: Computer Vision Lab, CAIDAS & IFI, University of Würzburg, Germany
    index: 1
date: 16 November 2026
bibliography: paper.bib
---

# Summary

The LEMUR framework is an open-source Python package for training neural networks
and analyzing their results under a shared protocol. Distributed through PyPI as
`nn-dataset`, it takes its name from the records it accumulates: every run is stored, queryable, and reusable. Researchers provide model code
and select a task, dataset, and evaluation metric; the framework trains the model
and records its scores, losses, hardware context, and resource use at every
epoch. Additional analyses characterize layers and identify architectures from source code,
allowing results across studies to be compared in one database. Models, data loaders, transformations,
and evaluation metrics are added as individual Python files, and models written
by researchers or code generators are evaluated identically. A
programmatic interface accepts a network's source code and returns validated
measurements, allowing external programs to submit candidates and retrieve
results without manual intervention. The software is released under the MIT
license at https://github.com/ABrain-One/nn-dataset [@nndataset2026].

# Statement of need

Studies of network architectures and training procedures repeat the same work: each requires training code, a way to store results together with the
model code and settings that produced them, and measurements of losses,
gradients, running time, and layer properties. Results depend on training
procedures, memory use, and hardware, so they are comparable only when these
conditions are recorded consistently [@bouthillier2021variance]. Per-study code is also hard to maintain and share
[@sculley2015debt].

The LEMUR framework provides a shared implementation of these components: a
common model interface that preserves each model's learning code, shared
training and failure handling, source-level architecture identity, and a
database that stores every measurement with its settings. It serves
three groups: researchers conducting comparative studies across many
architectures, who need a uniform procedure; AutoML and code-generation systems,
which need candidates validated programmatically within a search loop; and
researchers and students who reuse existing measurements rather than recompute
them. As studies expand across models, datasets, and settings, shared records
keep comparisons feasible and make completed experiments reusable.

# State of the field

Model libraries such as `timm` [@wightman2019timm] provide implementations and
weights. NAS-Bench-101 [@ying2019nasbench101] and NAS-Bench-201
[@dong2020nasbench201] provide results for defined collections of architectures,
while MLPerf [@mattson2020mlperf] specifies training procedures for a small set
of reference tasks. MLflow [@zaharia2018mlflow] and Weights and Biases
[@biewald2020wandb] record parameters, metrics, and model files. Ray Tune
[@liaw2018tune] schedules user-defined trials and handles failures. Optuna
[@akiba2019optuna] searches hyperparameters. PyTorch Lightning
[@falcon2019lightning] manages training loops, logging, and checkpoints through
model methods and callbacks.

Using these tools with researchers' model code requires extra code for source
comparison, layer measurements, and queries linking outputs to settings. The LEMUR framework combines them: a model interface that keeps learning code with the model, analysis
routines that work with any model implementing that interface, a submission
endpoint that validates candidate code under recorded conditions, and linked
database tables that associate outputs with settings. It reuses PyTorch and
Optuna for execution and search. Keeping measurement and selection code in the
package lets different studies reuse the same procedures while preserving
each model's learning method.

# Software design

The LEMUR framework separates data interfaces, execution, and record keeping so
that each can change without affecting the others. The analysis routines build
on this separation to characterize each implementation by its learning
behavior and computational cost under recorded conditions. \autoref{fig:overview}
shows how inputs pass through the evaluation engine into the records, and how
the architecture identity and post-training tools connect to them.

![Overview of the LEMUR framework: inputs, evaluation engine and its loops, records, architecture identity, and post-training tools.\label{fig:overview}](figures/lemur_connected_schema.png)

**Component interfaces.** Each PyTorch [@paszke2019pytorch] model module
contains its own loss function, optimizer, training steps, and hyperparameters,
while the engine handles data preparation, evaluation, and recording under a shared protocol. Datasets, transformations, and evaluation
metrics follow the same convention and are each provided as a single Python file implementing the declared interface. The experiment configuration names the task, dataset, metric, model, and transformation, which determine the modules the engine loads dynamically
and combines through their interfaces. For example, an image-classification
experiment combines AlexNet, CIFAR-10, accuracy, and normalization. Researchers
can add a model, data loader, transformation, or metric by writing one file,
selecting it by name, and reusing the shared training, evaluation, and recording
code unchanged.

**Programmatic interface.** Two functions expose the framework to other
programs. A query function returns stored records in a pandas DataFrame
[@mckinney2010pandas], filtered by task, dataset, or model; each record includes
the model source, hyperparameters, measurement, and duration. A
validation function accepts the network's source code as a string, with its task, dataset, metric, and hyperparameters. It runs the full pipeline
under a declared time budget and returns an identifier for the archived model,
its measured quality, a quality-per-time measure, and a structural score for the submitted code. Because a candidate is supplied as source
code rather than as an object in the caller's process, generators
that emit source text can be evaluated without adaptation, and the result is
returned directly to the caller. This lets code-generating systems close the loop between proposal and measurement.

**Robustness and budgets.** Before and during training, the package checks each
candidate to help ensure every trial produces a usable record. A short test builds the network and trains it on two batches of random
inputs to verify that initialization and the first training steps run. If a batch exceeds GPU memory, training restarts with a smaller batch. Each trial runs in isolation. Errors are recorded, and the
study proceeds with the next configuration; a pipeline is set aside only after
a configurable number of failures. Compute costs are bounded in advance: each epoch has a declared time limit (30 minutes by default), enforced
between batches using a running estimate of the total duration. A network whose
accuracy after an epoch falls below 1.2 times the random-guess baseline for its
dataset stops early. A long experiment across many models and settings therefore
has a predictable cost and continues despite individual failures.

**Record identity.** Different architectures must remain distinct in the
database, while the same architecture, even when written differently,
should be recognized as a single entry. The package therefore fingerprints each model's code. It parses the model source as Python syntax,
builds a graph of the layers and their settings, ignores variable names and
formatting, evaluates simple arithmetic expressions such as `256 * 6 * 6`, and
refines the resulting graph with the Weisfeiler–Lehman procedure
[@weisfeiler1968reduction] to produce a structural certificate stored as a
BLAKE2b hash. Computed from source code rather than from a constructed model,
the certificate identifies duplicates before any training and remains unchanged under renaming or reformatting. For architectures
with different fingerprints, MinHash signatures [@broder1997syntactic] over the code text measure similarity and retrieve related variants.
Other values, such as hyperparameter sets, receive a
simple content hash of their text.

**Records and storage.** Each run is recorded once per epoch rather than once
per batch, keeping storage low across many trials while preserving the
learning curve. A record links the achieved measurement to the settings that
produced it—learning rate, momentum, dropout, batch size, and transformation—and
to the machine context sampled during the run, making comparisons across
studies meaningful. Supplying these settings explicitly reruns a recorded
configuration, enabling published results to be reproduced. Optional per-layer
analysis stores one row per layer for selected epochs and records each layer's
contribution to the loss [@molchanov2017pruning], the information carried by
its representation [@roy2007effectiverank], the similarity of its output to
those of other layers [@kornblith2019cka], and a statistic of its weight matrix
[@martin2021weightwatcher]. A database layer stores code, settings, scores, and
layer measurements under the identities above and a schema shared
across experiments, so new runs reuse existing tables. Queries return
pandas DataFrames, so analysis can happen outside the training environment.
Each experiment also writes its statistics to JSON, allowing the database to be
rebuilt at any time.

**Post-training tools.** Saved checkpoints support further analyses. Latency
profiling measures CPU and GPU inference times after warm-up runs; structured
pruning removes the filters with the lowest L1 norm in each layer; and TFLite
conversion produces models at several precisions that can be benchmarked on
phones. Their results are stored in separate database tables
alongside the training records. ONNX export writes model files and checks their
accuracy with ONNX Runtime. The package installs from PyPI, with
optional extras for spreadsheet and plot export and for the mutually
incompatible mobile conversion toolchains. It also runs unchanged in the
provided container image.

# Research impact statement

The framework serves as the shared execution layer for a research program on
LLM-driven model design. Work supported by the framework has been presented at
workshops at CVPR, ICCV, ICPR, ECCV, and BMVC, and published in *Sensors*. At
scale, the framework has produced two public collections of trained networks
[@ABrain.NN-Dataset; @ABrain.LEMUR2] and provides the evaluation interface for
the NNGPT AutoML system [@ABrain.NNGPT] and LLM-based hyperparameter tuning
[@ABrain.HPGPT].

Its dense accuracy and resource measurements, comparable across architectures and studies, have enabled theoretical
analyses that ad hoc, per-model evaluation cannot support: a frontier-based
account of quality–resource trade-offs in model selection and its differential
extension to the multiresource frontier [@ABrain.Rank; @ABrain.Multi-Rank], and a convergence analysis of iterative LLM-based architecture search
[@ABrain.Converg].

Empirically, the framework supports controlled studies of training protocols
across thirty or more architectures [@ABrain.LR_Scheduling;
@ABrain.Loss-Optimizer; @ABrain.Transform2]; provides the fitness signal for
architecture generation and search [@ABrain.Prompt; @ABrain.Architect;
@ABrain.NN-Captioning_2025; @ABrain.Delta; @ABrain.Feedback_Memory;
@ABrain.Similarity; @ABrain.CV_Channel; @ABrain.CV_Channel_WS; @ABrain.MoE];
trains surrogate models of accuracy and latency [@ABrain.NN-Stat;
@ABrain.NN-Perf]; validates networks recovered from repositories or curated
into families [@ABrain.NN-RAG; @ABrain.Analog]; and supports mobile deployment
pipelines [@ABrain.NN-Lite; @ABrain.NN-Lite-Mobile; @ABrain.Device-First].

Development is collaborative and externally driven: since October 2024, the
repository has merged several hundred pull requests from more than 40
contributors, and the package has accumulated tens of thousands of PyPI
downloads [@pepy2026downloads], including automated retrievals. The project
provides installation and extension documentation, tagged releases, a
changelog, examples, and a test suite.

# AI usage disclosure

DeepSeek V4.1 Flash and Claude Opus 5.5 assisted with drafting and editing this
manuscript, code inspection, documentation, and checking claims against the
source code. The authors reviewed and validated all AI-assisted output, made all
design and scientific decisions, and take full responsibility for the work.

# Acknowledgements

We thank the many contributors to the LEMUR framework. Waleed
Khalid, Roman Kochnev, Akashdeep Singh, Sai Revanth Mulukuledu, Chandini
Vysyaraju, Raghuvir Duvvuri, Arash Torabi Goodarzi, and Muhammad A. Hussain
contributed to the software implementation. This work was partially
supported by the Alexander von Humboldt Foundation, which had no role in designing the software or preparing this manuscript. The authors
declare no competing interests.

# References
