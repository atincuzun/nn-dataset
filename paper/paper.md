---
title: 'LEMUR: Learning, Evaluation, and Modeling for Unified Research, for training and analysing neural networks'
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
date: 3 October 2026
bibliography: paper.bib
---

# Summary

The LEMUR framework, distributed as `nn-dataset`, is an open-source Python package for training neural networks and
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

Researchers produce many task-specific network architectures, and studies often rewrite similar training code. Many ideas must be tried, tested, and compared, and the results must be stored with the 
model code and training settings used to obtain them. Differences in learning procedures, memory requirements, and hardware affect interpretation. Studies of benchmark
variability [@bouthillier2021variance] and software maintenance
[@sculley2015debt] support the need for consistent recording and shared code.
Researchers studying training procedures need to keep model-specific learning
code while measuring losses, gradients, running time, and layer properties.
Implementing these measurements separately in each study repeats work and complicates comparison.

The LEMUR framework provides a common model interface, training and retry procedures, similarity checking,
and a database for keeping the measurements. Researchers in deep learning, neural architecture
search, and AutoML can evaluate their own model code
without manual intervention. Runs multiply as studies vary models, datasets, and settings,
so common records and explicit selection criteria keep comparisons feasible and completed
experiments reusable.

# State of the field

Model libraries such as `timm` [@wightman2019timm] provide implementations and
weights. NAS-Bench-101 [@ying2019nasbench101] and NAS-Bench-201
[@dong2020nasbench201] provide results for defined collections of architectures. 
MLflow [@zaharia2018mlflow] and Weights and Biases [@biewald2020wandb] record
parameters, metrics, and model files. Ray Tune [@liaw2018tune] schedules
user-defined trials and handles failures. Optuna [@akiba2019optuna] searches
hyperparameters. PyTorch Lightning [@falcon2019lightning] manages training loops,
logging, and checkpoints through model methods and callbacks.

Using these tools with submitted implementations requires extra code for
source comparison, layer measurements, and queries connecting these outputs to
settings. The contribution of the LEMUR framework is to supply these components as one
combination, a model interface that keeps learning code with the model,
analysis routines that work with any model following it, and linked database
tables that tie outputs to settings, while reusing PyTorch and Optuna for
execution and search. Keeping measurement and selection code in the package lets different
studies reuse the same procedures while retaining each model's learning method.

# Software design

The LEMUR framework was designed with modularity in mind. Data interfaces, processing, and recording are kept separate. The analytical routines use this foundation to relate the implementations to their
learning behaviour and computational requirements under recorded conditions.

![Overview of the LEMUR framework workflow, recorded information, and the source-analysis path.](diagram.png)

**Component interfaces.** In the package, PyTorch [@paszke2019pytorch] model modules contain their loss, optimizer, learning steps, and hyperparameters, while the engine handles data preparation, evaluation, and recording under a declared common protocol. Compatible code written to the interface unifies the pipeline while allowing user-defined custom training procedures. This separates model-specific learning from shared evaluation and recording, allowing researchers to narrow down their scope to individual components while reusing the engine. These conventions were described with LEMUR [@goodarzi2025lemur].

A complete training cycle also requires a dataset, a transformation, and an evaluation measure. As with model modules, the same principles are applied to these components. Each dataset, transformation, and evaluation measure is provided as a single Python file that follows its declared interface. Therefore, researchers can write their own datasets, transformations, or measures following the same convention and reuse the full training engine, focusing on particular aspects such as structural network-level modifications rather than execution code.

**Dynamic imports.** In addition to these interface conventions, name-based dynamic imports are used to compose the system. The training engine loads the selected Python modules dynamically and combines them through their declared interfaces. Component names identify the implementations, while the interfaces define how they interact with the shared engine.

The experiment configuration identifies the task, dataset, metric, and model. Together with the transformation name, these selections determine which modules the engine loads and combines. For example, an image classification experiment can combine AlexNet, CIFAR-10, accuracy, and normalization. This design allows researchers to introduce custom models, dataset loaders, transformations, or metrics through the same interfaces and select them by name for supported tasks. Researchers can therefore extend the system with new components while reusing the shared training, evaluation, and recording code.

**Failure handling and recovery.** The execution of the neural network can fail due to various issues. The code may contain mistakes, the GPU may run out of memory, an epoch may exceed its time budget, or accuracy may stay below the dataset minimum. The package handles these conditions to prevent individual failed trials from stopping the whole experiment. Static code analysis is performed before training. Pylint [@pylint2024] reports errors, unused variables, and style problems, and Radon [@radon2024] reports the complexity of each function. A quick test then builds the network and trains two batches of random images to check that initialization and initial learning steps execute successfully. During runtime, OOM errors are handled by lowering the maximum batch size until a suitable small batch is found.

Crashes can still occur due to the transient dynamics of the environment. The common causes are download failures, fragmentation errors, resource contention, unlucky sampling and many more. To handle these external influences, retries are introduced so the experiments can continue without disruption.

**Budget allocation.** Without a time limit, a slow model could consume time intended for other models, and the cost would become known only after the run. The limit is therefore set before training, so the cost is decided in advance. The time limit is declared by the researcher and is 30 minutes per epoch by default. For each epoch, an estimator predicts the total duration from the first batches, and the epoch is stopped between batches when the prediction exceeds the limit. For performance, a network that stays below 1.2 times the random guess baseline for its dataset is stopped early.

**Record identity.** Different architectures must stay distinct in the database, and the same architecture written in different ways should be recognized as a single entry. The package therefore gives model code its own fingerprint. It reads the model source as Python syntax, follows the layers and their settings, ignores variable names and formatting, folds simple arithmetic such as `256 * 6 * 6` into one number, and refines the resulting graph with the Weisfeiler–Lehman procedure [@weisfeiler1968reduction] into a structural certificate stored as a BLAKE2b hash. Computing the identity from a trained model would require building every candidate first, which is the cost that the check is intended to avoid, and exact comparison of source files would require comparing candidates in pairs. The certificate is computed from the source, so a duplicate is identified before training. Renaming and formatting changes preserve the same identity, and MinHash similarity identifies related models. For architectures with different fingerprints, the package also uses MinHash signatures [@broder1997syntactic] over the code text to measure similarity and retrieve related variants. Values that are not model code, such as hyperparameter sets, receive a plain content hash of their text instead.

**Record structure.** Every run stores the primary metric, the duration, the
batch size, and the learning rate, momentum, dropout, and transformation
settings. Records are written once per epoch rather than once per batch, which keeps the stored data small across many trials while still showing the learning curve. For reproducibility,
the record adds the training and test loss, the training accuracy, the
gradient norm after the epoch, samples per second, and the sampled machine
context, which covers processor load, total and occupied memory, and the
graphics device with its memory. Layer analysis stores one row per analysed
layer for selected epochs, including how much the layer contributes to the
loss [@molchanov2017pruning], how much of its representation carries
information [@roy2007effectiverank], how similar its output is to that of
other layers [@kornblith2019cka], how much the loss changes when its weights
and biases are zeroed, the size of its gradients, the fraction of its near-zero outputs, and a statistic of its weight matrix [@martin2021weightwatcher]. 

**Persistent storage.** A database layer holds the stored code, settings,
scores, and layer measurements under the identities described above, so runs
can be found and compared across experiments. The schema is shared across experiments, allowing new runs to reuse the existing tables. If the
database is lost or damaged, it can be rebuilt from this material, so the
records stay recoverable. Queries return pandas DataFrames
[@mckinney2010pandas], so analysis happens outside the training environment. Each experiment also writes its statistics
as JSON files into its own folder, where they can be read directly.

# Research impact statement

The neural network dataset that is introduced with LEMUR
[@goodarzi2025lemur] was gathered with this framework. The package also provided extension and evaluation software for the
LEMUR 2 collection [@uzun2026lemur2] and an API for NNGPT
[@kochnev2026nngpt]. Applications include
a study of 25 learning-rate schedules across 30 architectures
[@mateen2026scheduling] and 18 loss-optimizer pairings across 33 architectures
and six datasets [@abramochkin2026lossoptimizer]. Studies of model generation
through code changes [@adhikari2026delta] and extraction of network code from
repositories [@khalid2025nnrag] also used the evaluation procedures. These uses
document the reuse of training and query components for different experimental
questions.

PePy reported 31,992 cumulative downloads of the LEMUR framework as of
15 October 2026, including automated downloads [@pepy2026downloads].

Development history dates to October 2024. Source version 2.2.13 includes the
training and analysis modules. The repository provides installation and extension
instructions, tagged releases, a changelog, examples, and automated tests.
The package requires Python 3.10 or newer. 
Shared database copies and saved model weights are available at
https://huggingface.co/NN-Dataset.

# AI usage disclosure

DeepSeek V4.1 Flash, used through the Kilo Code assistant, supported drafting,
restructuring, and editing of this manuscript, code inspection, and
documentation. The human authors reviewed, edited, and validated all AI-assisted
output, checked technical claims against the source code, made the core design
and scientific decisions, and remain responsible for accuracy, originality, and
licensing.

# Acknowledgements

Contributions to the LEMUR framework are gratefully acknowledged. Waleed Khalid,
Saif U Din, Sai Revanth Mulukuledu, Akashdeep Singh, Chandini Vysyaraju,
Raghuvir Duvvuri, Avi Goyal, Yashkumar Rajeshbhai Lukhi, Muhammad A. Hussain,
Krunal Jesani, Usha Shrestha, Yash Mittal, Roman Kochnev, Pritam Kadam,
Mohsin Ikram, Harsh Rameshbhai Moradiya, and Alice Arslanian contributed to
the dataset and software. The Alexander von
Humboldt Foundation provided partial support.

# References
