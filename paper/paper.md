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

LEMUR introduced the original model collection and evaluation framework
[@goodarzi2025lemur]. LEMUR 2 described a larger collection and deployment
applications [@uzun2026lemur2]. NNGPT studied model generation and optimization
using large language models [@kochnev2026nngpt]. This article describes the
current software package, its measurements, and its tools for comparing code
and stored results. The original interface, hyperparameter search, and database
are not presented as new methods. The article does not repeat the procedures
for collecting and generating models or the comparisons of model performance
reported in the earlier papers.

# State of the field

Model libraries such as `timm` [@wightman2019timm] provide implementations and
weights. NAS-Bench-101 [@ying2019nasbench101] and NAS-Bench-201
[@dong2020nasbench201] provide results for defined collections of architectures.
`nn-dataset` accepts model source code that follows its interface, including
code for the learning procedure.

MLflow [@zaharia2018mlflow] and Weights and Biases [@biewald2020wandb] record
experiments. Ray Tune [@liaw2018tune] schedules trials and handles failures.
Optuna [@akiba2019optuna] searches hyperparameters. PyTorch Lightning
[@falcon2019lightning] manages training loops, logging, and checkpoints.
`nn-dataset` uses PyTorch and Optuna and adds measurements and a database for
comparing implementations. Equivalent procedures could be built with these
other tools. They would need code for the model interface, analyses, and
database structure. Keeping these functions in one package allows different
studies to use the same measurement code without changing each model's learning
procedure.

# Software design

**Model interface.** A model uses PyTorch [@paszke2019pytorch] and follows the
interface introduced in LEMUR. Its Python file declares supported
hyperparameters and provides a `Net` class with training setup and learning
methods. Model code defines the loss, optimizer, and learning steps. The package
prepares data and devices, calls the model's methods, evaluates predictions,
and saves results. Other model implementations must be adapted to this interface.
New loaders, evaluation measures, and transformations are added as Python
modules. The `check_nn` function accepts supplied source code rather than only
models from the collection. Submitted code runs with the Python process's
permissions and must be trusted. The package requires Python 3.10 or newer.
Training requires CUDA, but code comparison and database queries do not.

**Training and failure handling.** Direct evaluation runs the settings supplied
by the caller. The training script also searches hyperparameters with Optuna,
retries failures, and reduces the range of batch sizes searched after CUDA
out-of-memory errors. These retry rules do not apply to every evaluation call.
During learning, the data iterator estimates epoch duration from elapsed time
and the number of batches processed. It raises an exception when the estimate
exceeds the chosen limit. Saved epochs remain available. This estimate does not
strictly bound total running time. Evaluation and layer analysis take additional
time, and model code controls the learning loop. Source checks use Pylint
[@pylint2024] and Radon [@radon2024] to report code issues and complexity.
A short image-classification test checks whether the model can initialize and
run learning steps. These checks are advisory and do not establish correctness
for all tasks.

**Training and layer measurements.** When epoch recording is enabled, the
package saves training and evaluation losses, evaluation scores, gradient norm,
an estimate of samples processed per second, and the best score and its epoch.
It also records hardware information and samples CPU, RAM, and GPU-memory use.
These records do not provide continuous resource monitoring or guarantee the
same results on different machines.

Optional layer analysis records gradient norms and the fraction of activations
near zero. It also estimates layer importance from weights and gradients
[@molchanov2017pruning], effective rank [@roy2007effectiverank], and similarity
between layer outputs using centered kernel alignment [@kornblith2019cka].
Further measurements include the change in loss after setting layer weights
to zero and weight-matrix statistics calculated with WeightWatcher
[@martin2021weightwatcher].
These are established measures added to the experiment record, not new methods
introduced by this package. Analysis covers supported convolutional and linear
layers. Several estimates use selected batches, and WeightWatcher requires an
optional dependency. These analyses are disabled by default because they require
additional calculations. Model summaries include parameter counts, layer types,
depth, and approximate operation counts for supported layers. Network-type labels
inferred from code are estimates, not a complete classification of architectures.

**Code comparison and stored results.** The package constructs a graph from
Python source and updates its labels using the Weisfeiler-Leman method
[@weisfeiler1968reduction]. It hashes the result to produce an identifier called
a fingerprint. Selected variable renamings, formatting changes, and equivalent
constant expressions preserve this identifier. The calculation uses source code
without executing the model. This differs from tracing a model object
with `torch.fx` [@reed2022torchfx]. Programs with different behaviour can receive
the same fingerprint. A match therefore does not establish that two programs or
architectures are equivalent. Settings passed at runtime and trained weights
are separate from this identifier. Merging or deleting model files requires
inspection. Tests cover selected renamings and other code changes.
A separate index can compare the fingerprint of new code with stored ones
before training. This check is not automatically part of the training search.
MinHash [@broder1997syntactic] measures similarity between source files.

JSON files and a SQLite database store results for later analysis. Queries
combine available source code, settings, scores, model summaries, and layer
measurements. Researchers can select records with the same task, dataset,
evaluation measure, and epoch count. They can compare training losses and layer
properties as well as final scores. Rebuilding a database requires
retained code and measurement files or a compatible saved database. Continuing
a training search counts completed trials but does not restore optimizer,
random-number, or Optuna search state. Compression and upload tools support
sharing files. Uploading them does not establish that the archive contains
everything needed to reproduce a run. Reproduction also requires the relevant
code, data preparation, software environment, and random states.

# Research impact statement

The package has been used for studies beyond constructing the LEMUR collections.
A learning-rate study evaluated 25 schedules across 30 architectures
[@mateen2026scheduling]. A loss-optimizer study evaluated 18 pairings across
33 architectures and six datasets [@abramochkin2026lossoptimizer]. Studies of
model generation through code changes [@adhikari2026delta] and extraction of
network code from repositories [@khalid2025nnrag] also used the evaluation
procedures. These papers report their own scientific findings. They demonstrate
use within the collaborating research group, not independent external adoption.

Development history dates to October 2024. The repository contains tagged
releases, a changelog, contribution instructions, examples, and automated tests.
This article describes the version 2.2.13 source code. Earlier PyPI releases may
contain fewer analysis tools. The repository provides installation and extension
instructions. Shared database copies and saved model weights are available at
https://huggingface.co/NN-Dataset.

# AI usage disclosure

DeepSeek V4.1 Flash was used during earlier software development, documentation,
and paper drafting. Human authors reviewed those outputs and made the core
design decisions. Kilo Code, using GPT-6.1 Sol, was used to compare papers,
inspect code, and revise the manuscript. Human authors must review and validate
this revision before submission. They remain responsible for its accuracy,
originality, and licensing. Separately, the package can evaluate models produced
by generators such as NNGPT [@kochnev2026nngpt]. Using generated models in
experiments is distinct from using AI to prepare software and a paper.

# Acknowledgements

We thank the contributors to `nn-dataset`. This work was partially supported
by the Alexander von Humboldt Foundation. The foundation had no role in the
design, implementation, or reporting of the work.

# References
