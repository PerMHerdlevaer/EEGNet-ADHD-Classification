# EEG-Based ADHD Classification with EEGNet

This repository contains the code developed for my master's thesis:

**Deep Learning Approaches for EEG-Based ADHD Classification: The Role of Preprocessing and Frequency-Specific Features** (NTNU, 2026).

## Overview

The project investigates how preprocessing choices and frequency-specific EEG information influence deep learning models for ADHD classification. An EEGNet-based convolutional neural network is evaluated using strict Leave-One-Subject-Out (LOSO) cross-validation on a public EEG dataset of 121 children. The code is designed to be easy to understand and adapt to other EEG datasets.

The dataset used is from the public dataset:
https://ieee-dataport.org/open-access/eeg-data-adhd-control-children

## Repository Structure


EEGNet.py                     # EEGNet model implementation
functions.py                  # Helper functions
Run_EEGNet.py                 # Main training and evaluation script


## Method

* EEG recordings are filtered into different frequency bands.
* Signals are segmented into 6-second windows and normalized.
* EEGNet is trained using subject-level LOSO cross-validation.
* Multiple frequency-specific models are evaluated and compared.

## Frequency Bands

* Full Band (0.5–40 Hz)
* Delta + Theta (0.5–8 Hz)
* Alpha (8–12 Hz)
* Alpha + Low Beta (8–20 Hz)
* Beta (12–30 Hz)
* Beta + Gamma (12–40 Hz)
* High Beta + Gamma (20–40 Hz)
* Gamma (30–40 Hz)


