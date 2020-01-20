# Probability_Calibration_Imbalanced
This repository implements 
[1] [Pozzolo, et al., (2015)](https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf)'s probability calibration for imbalanced dataset.

This repository compares this probability calibration method using Bayes Minimum Risk theory to vanilla logistic regression, 
the common methods for imbalanced dataset, which is random undersampling method and bagging.

The documentation of this repository can be found at [my medium blog post](https://towardsdatascience.com/probability-calibration-for-imbalanced-dataset-64af3730eaab).

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
```

## Goal
The commonly used method for imbalanced dataset, undersampling causes a bias in the posterior probabilities. This is due to the characteristic of random undersampling, which downsizes the majority class by removing them randomly until both classes have the same number of observations. This makes the class distribution of the training set different from the one in the test set. So how exactly probability calibration using Bayes Minimum Risk theory works on this problem? — the basic idea of this method is trying to reduce/remove the bias caused by random undersampling by taking into the undersampling ratio β account.

Here is the code snippet of probability calibration with using Bayes Risk Minimum theory. beta is the undersampling ratio, tau is the threshold and calibration function does the calibration task.
```
# Probability calibration with BMR (Bayes Minimum Risk) implementation
# Pozzolo et al., 2015, Calibrating Probability with Undersampling
class BMR:
    def beta(binary_target):
        return binary_target.sum()/len(binary_target)
    def tau(binary_target, beta):
        return binary_target.sum()/len(binary_target)
    def calibration(prob, beta):
        return prob/(prob+(1-prob)/beta)
```

## Example
We will see how the probability calibration technique the model performance on a binary classification problem on the famous credit card fraud dataset on Kaggle. This dataset consists of 28 PCA features (all of them are anonymous) plus the Amount feature. The target feature is binary, either fraud or not. The positive class is 0.17% of a whole dataset, which is severely imbalanced.

Here we will compare this probability calibration method to other common methods for imbalanced dataset as mentioned above. Below is the results of it.

|           |   RUS  | RUS Bagging | Calibration with RUS | Calibration with RUS Bagging |
|:---------:|:------:|:-----------:|:--------------------:|:----------------------------:|
|  F1 Score | 0.5559 |    0.7788   |        0.7692        |            0.7914            |
| Precision | 0.4162 |    0.7364   |        0.7732        |            0.8315            |
|   Recall  | 0.8367 |    0.8265   |        0.7653        |            0.7551            |
|  Log loss | 0.0794 |    0.0279   |        0.0273        |            0.0236            |

As we can see, after calibration those scores improved, especially the difference between before and after calibration on the random undersampling model are significant. So by correcting the biased probability by using probability calibration, we could see the performance improvement.

## Reference
[1] Pozzolo, et al., [Calibrating Probability with Undersampling for Unbalanced Classification (2015)](https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf), 2015 IEEE Symposium Series on Computational Intelligence
