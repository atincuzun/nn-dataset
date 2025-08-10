# Generated Architectures

This folder contains the Python code for unique neural network architectures evaluated during the Genetic Algorithm (GA) search process.

*   **File Naming:** Each architecture is saved as `ga-alexnet-X.py`, where `X` is a sequential number.
*   **Content:** The file contains the complete PyTorch code for the `Net` class, configured according to the evolved parameters (chromosome) for that specific architecture.
*   **Statistics:** Evaluation statistics (like per-epoch accuracy) for each architecture saved here can be found in the `../stats` directory. The corresponding stats are located in a subfolder named `img-classification_cifar-10_acc_ga-alexnet-X`, where `X` matches the architecture file number.
