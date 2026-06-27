# EEG-Based ADHD Classification with EEGNet

This repository contains the code developed for my master's thesis:

**Deep Learning Approaches for EEG-Based ADHD Classification: The Role of Preprocessing and Frequency-Specific Features** (NTNU, 2026).

## Overview

The project investigates how preprocessing choices and frequency-specific EEG information influence deep learning models for ADHD classification. An EEGNet-based convolutional neural network is evaluated using strict Leave-One-Subject-Out (LOSO) cross-validation on a public EEG dataset of 121 children. The code is designed to be easy to understand and adapt to other EEG datasets.

The dataset used is from the public dataset:
https://ieee-dataport.org/open-access/eeg-data-adhd-control-children

## Repository Structure

```text
EEGNet.py                     # EEGNet model implementation
functions.py                  # Helper functions
Run_EEGNet.py                 # Main training and evaluation script
create_fif.ipynb              # Script to transform the dataset from csv to fif. Can be used if the dataset is downloaded from Kaggle as a csv.
```

## Method

* EEG recordings are filtered into different frequency bands.
* Signals are segmented into 6-second windows and normalized.
* EEGNet is trained using subject-level LOSO cross-validation.
* Multiple frequency-specific models are evaluated and compared.

## How To Run

* Download the csv from the 



