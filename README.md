# Physics-Informed MOGP for HEA Screening

This repository contains the Jupyter Notebooks and analysis code for the manuscript: **"A multi-output Gaussian process framework for physics-embedded transfer learning in data-scarce high-entropy alloy screening"**.

## Repository Content

| Notebook | Description |
| :--- | :--- |
| `Digital_twin_files.ipynb` | Data cleaning, feature engineering, and physics-informed embedding generation. |
| `MOGP_asset_creation.ipynb` | Implementation of the Multi-Output Gaussian Process (MOGP) architecture. |
| `digital_twin_files.py` | python file of the Digital_twin_files.ipynb used for auditing the codes. |
| `Figure_2.ipynb` | Visualizes the 3D physics embedding space by plotting material properties (density, conductivity, specific heat) relative to an interpolation centroid. |
| `Figure_5.ipynb` | generate a comprehensive "Data Validity & Sensitivity Report" |
| `Figure_6.ipynb` | implements a Multi-Output Gaussian Process (MOGP) using GPyTorch to model manufacturing outputs with uncertainty quantification |
| `Figure_8.ipynb` | performs a statistical correlation analysis on the raw experimental outputs and visualizes their inter-dependencies using a labeled Pearson correlation heatmap. |
| `Figure_9.ipynb` | executes a rigorous 5-fold cross-validation benchmark to compare the predictive accuracy ($R^2$) of Multi-Output Gaussian Processes and Stacking Ensembles against standard ML models. |
| `Figure_10.ipynb` | performs a Sobol sensitivity analysis on a trained Multi-Output Gaussian Process (MOGP) surrogate to quantify how much each parameter (contributes to the variance in Maximum Temperature. |
| `Figure_11 to 15.ipynb` | assesses the MOGP model's zero-shot generalization capability by training on a subset of alloys and validating its ability to accurately predict the multiphysics outputs of a completely unseen material (AlCoCrFeNi). |
| `Figure_16.ipynb` | evaluates the Multi-Output Gaussian Process (MOGP) model by calculating and visualizing key performance metrics (RMSE, R-square, and Negative Log Likelihood) across training and validation sets to ensure the model isn't overfitting. |
| `Figure_17.ipynb` | generates a series of high-resolution contour plots (Figure 17a–e) visualizing how the thermal process window shifts for five different alloys under varying laser beam radii using the pre-trained Stacking Ensemble model. |
| `Figure_18.ipynb` | leverages the trained MOGP surrogate to perform multi-objective Pareto optimization, identifying the optimal "knee point" trade-off between minimizing stress and maximizing heat flux across 10,000 virtual experiments. |
| `Figure_19.ipynb` | utilizes the trained MOGP surrogate to execute multi-objective Pareto optimization, specifically identifying the optimal "knee point" trade-off between minimizing Max Principal Stress and maximizing heat flux. |
| `digital_twin_asset.pkl` | This binary file is a serialized Python dictionary (Joblib/Pickle) containing the pre-trained "Digital Twin" machine learning pipeline, feature scalers, and material property metadata required to simulate and predict the multiphysics outputs of the alloy manufacturing process. |
| `MOGP_assets.pkl` | This binary file contains the serialized model weights, likelihood parameters, feature scalers, and material metadata required to reload the pre-trained Multi-Output Gaussian Process (MOGP) surrogate for probabilistic inference. |
| `sobol_sequence_generator.py` | This script uses Scipy's quasi-Monte Carlo engine to generate a space-filling Sobol sequence Design of Experiments (DoE) , creating a 50-run table that varies Power, Speed, and Radius across five different alloy materials. |
| `MOGP_asset_creation.ipynb` | This Jupyter Notebook trains the Multi-Output Gaussian Process (MOGP) surrogate model using GPyTorch and exports the optimized state dictionary and training metadata into the reusable mogp_assets.pkl file. |

## Usage

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
