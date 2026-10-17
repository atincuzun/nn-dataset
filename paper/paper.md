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

Hundreds of different neural network structures are produced in the domain of computer science, targeted to specific tasks. This introduces the problem of reimplementing the repetitive code across independent research. Scientist must write the similar loops to train a neural network. Many ideas need to be tried, tested and compared, producing a growing number of results that must be stored together with the 
model code and training settings used to obtain them, including the differences in learning procedures,
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

**Component interfaces.** In the package, PyTorch [@paszke2019pytorch] model modules contain their
loss, optimizer, learning steps, and hyperparameters while the engine handles data preparation,
evaluation, and recording under a declared common
protocol. Compatible code written to the interface unifies the pipeline while allowing user-defined custom training procedures. Importantly, this separates model-specific learning from shared evaluation and recording, allowing researchers to narrow down the scope of their work. Model files are expected to be Python files that provide a `Net` class with the required methods. These conventions were described with the LEMUR dataset [@goodarzi2025lemur], including the `Net(nn.Module)` class and its `train_setup(self, prm)` and `learn(self, train_data)` methods.

A complete training cycle also requires a dataset, a transformation, and an evaluation measure. As with model modules, the same principles are applied to these components. For example, a dataset file provides `loader(transform_fn, task)`, a transformation file provides `transform(norm)`, and an evaluation measure file provides `create_metric(out_shape)`. Therefore, researchers can easily write their own datasets, transformations, or measures following the same convention and reuse the full training engine, focusing on particular aspects like structural network-level modifications rather than execution code.

**Dynamic Imports** Aside from the definitions, the name based dynamic imports were extensively used to compose the system. The components described above are loaded dynamically and combined in the
training engine in `ab/nn/util/Train.py`. Loading helpers import each component's
Python module by name and get the class or function required by its interface.

The command-line entry point in `ab/nn/train.py` lets users specify these names.
For example,
`python -m ab.nn.train --config img-classification_cifar-10_acc_AlexNet --transform norm`
selects AlexNet, CIFAR-10, accuracy, and normalization for image classification.
The entry point reads the configuration as task, dataset, metric, and model and
passes these names to `Train.py` with the transformation name. The engine then
loads the selected modules and combines them through the interfaces above.
Researchers can add a custom model, dataset loader, transformation, or metric in
the corresponding package directory and select it by name. The new file must
follow the required interface and work with a supported task. It can then be used
without changing the shared training, evaluation, or recording code.

**Failure handling and recovery.** The execution of the neural network can fail due to various issues. The code may contain mistakes, the GPU may run out of memory, an epoch may exceed its time budget, or accuracy may stay below the dataset minimum. The package handles each case and returns a recorded result instead of stopping the whole experiment. Mistakes in the code are caught before training. Pylint [@pylint2024] reports errors, unused variables, and style problems, and Radon [@radon2024] reports the complexity of each function. A quick test then builds the network and trains two batches of random images to confirm that the code works. During runtime, OOM errors are handled by lowering the maximum batch size until a suitable small batch is found.

Crashes can still occur due to the transient dynamics of the environment. The common causes are download failures, fragmentation errors, resource contentions, unlucky sampling and many more. To handle these external influences, retrials are introduced so the experiments can continue without disruption.

**Budget Allocation** While training a neural network, certain criteria can be
taken into account such as training time and performance. The time limit is
declared by the researcher and is 30 minutes per epoch by default. For each
epoch, an estimator predicts the total duration from the first batches, and
the epoch is stopped between batches when the prediction exceeds the limit.
For performance, a network that stays below the random guess baseline for its
dataset is stopped early.

**Record identity.** Different architectures must stay distinct in the database, and the same architecture written in a different way must stay one entry. The package therefore gives model code its own fingerprint. It reads the model source as Python syntax, follows the layers and their settings, ignores variable names and formatting, folds simple arithmetic such as `256 * 6 * 6` into one number, and refines the resulting graph with the Weisfeiler-Leman procedure [@weisfeiler1968reduction] into a single certificate stored as a blake2b hash. The same network written differently receives the same certificate, and this certificate keys the recorded results of that architecture. For graded comparison rather than exact identity, the package also measures similarity between two architectures and stores MinHash signatures [@broder1997syntactic] over the code text for retrieving close variants. Values that are not model code, such as hyperparameter sets, receive a plain content hash of their text instead.

**Layer statistics.** Optional analysis adds an inside view of the model beyond its score. When enabled, every selected epoch stores measurements for each convolutional and linear layer, including how much the layer contributes to the loss [@molchanov2017pruning], how much of its representation carries information [@roy2007effectiverank], how similar its output is to that of other layers [@kornblith2019cka], how much the loss changes when its weights and biases are zeroed, the size of its gradients, the fraction of its dead neurons, and a statistic of its weight matrix [@martin2021weightwatcher]. The measurements need extra forward and backward passes, so they are computed only in the first five epochs and every fifth epoch afterwards, on a few batches. They point to where the next experiment should look and do not explain on their own why one model scores higher.

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

PePy reported 31,992 cumulative downloads of `nn-dataset` as of
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

Contributions to `nn-dataset` are gratefully acknowledged. Waleed Khalid,
Saif U Din, Sai Revanth Mulukuledu, Akashdeep Singh, Chandini Vysyaraju,
Raghuvir Duvvuri, Avi Goyal, Yashkumar Rajeshbhai Lukhi, Muhammad A. Hussain,
Krunal Jesani, Usha Shrestha, Yash Mittal, Roman Kochnev, Pritam Kadam,
Mohsin Ikram, Harsh Rameshbhai Moradiya, and Alice Arslanian contributed to
the dataset and software. The Alexander von
Humboldt Foundation provided partial support.

# References
