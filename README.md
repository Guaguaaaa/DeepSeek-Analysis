# Data Analysis Scripts for Behavioural Economics Research

## Overview

This repository contains the Python source code used for the data processing and statistical analysis portion of a research paper in behavioural economics. 

**Please note:** Due to the confidential nature of the underlying data, the dataset is not included. As a result, the scripts are **not runnable out-of-the-box** and are intended for review purposes only.

## Key Techniques Demonstrated

* **Data Processing:** Advanced data cleaning, preprocessing, and feature extraction using the `pandas` and `numpy` libraries.
* **Statistical Modelling:** Implementation of Partial Least Squares Structural Equation Modeling (PLS-SEM). The analysis leverages R's powerful `plspm` package within a Python environment, facilitated by the `rpy2` library bridge.
* **Reproducibility:** The code is structured in a modular way to support clarity and reproducibility in an academic research context.

## Requirements

To review the code and understand its dependencies, the following environment is recommended:

* Python 3.8+
* pandas
* numpy
* rpy2
* An underlying installation of R with the `plspm` package.

## A Note on Execution

The scripts in this repository are intended to be **read and reviewed**, not executed directly. 

Throughout the code, you will find data loading commands (e.g., `pd.read_csv(...)`) have been commented out to protect data privacy. These sections are left in place to illustrate the intended data pipeline and demonstrate the full logic of the analysis process.
