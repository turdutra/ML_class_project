# Machine Learning Project for P_MT571A_2024S2

This repository contains the class project for the Machine Learning course (P_MT571A_2024S2) at UNICAMP.

## Project Structure

### 1. Data Generation

In the `generate_dataset` folder, you’ll find Julia functions used to create the dataset for machine learning. This section includes the following files:

- **`generate_dataset_mp.jl`**: The main script for creating dataset entries.
- **`functions.jl`**: Contains auxiliary functions used by `generate_dataset_mp.jl`.
- **Polytope Files**: These files relate to the polytope hyperparameter required in the steering certification algorithm.

### 2. Model Training and Analysis

- **`ml_training_disciplina.py`**: The primary script for hyperparameter search and model training.
- **`ml_func.py` and `ml_stat.py`**: These files contain functions and statistical tools for dataset analysis.

### 3. Best Models

In the `best_models` folder, you’ll find:

- Code for further training of the top 10 models.
- Results of the top 10 models, including performance metrics and any additional analysis.

---

This structure should help you navigate the project files and understand their purposes.
