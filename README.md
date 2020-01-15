# Probability_Calibration_Imbalanced
This repository implements 
[1] [Pozzolo, et al., (2015)](https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf)'s probability calibration
for imbalanced dataset.

This repository compares this probability calibration method to vanilla logistic regression, 
the common methods for imbalanced dataset, which is random undersampling method and 

## Table of contents
* [Requirements](#requirements)
* [Goal](#goal)
* [Example](#example)
* [Reference](#reference)

## Requirements
```
Python 3.6.5
imbalanced_learn == 0.5.0
matplotlib == 3.1.2
numpy == 1.15.4
pandas == 0.23.4
scikit_learn == 0.21.3
seaborn == 0.9.0
src == 0.0.7
```

## Goal
undersampling causes a bias in the posterior probabilities. This is due to the characteristic of random undersampling, 
which downsizes the majority class by removing them randomly until both classes have the same number of observations. 
This makes the class distribution of the training set different from the one in the test set. So how exactly probability 
calibration using Bayes Minimum Risk theory works on this problem? — the basic idea of this method is trying to 
reduce/remove the bias caused by random undersampling by taking into the undersampling ratio β account.

## Example


## Reference

