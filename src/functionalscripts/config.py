## config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io, os, sys, types, gc, re
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score,confusion_matrix,precision_score,recall_score,mean_squared_error, roc_auc_score, roc_curve, precision_recall_curve, cohen_kappa_score, f1_score, log_loss
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
from imblearn.ensemble import BalancedBaggingClassifier
import tqdm
from src.functionalscripts.Bayes_Mimimum_Risk import *

def make_prediction(model,X,threshold):
    y_pred = model.predict_proba(X)
    y_predicted = np.where(y_pred[:,1]>=threshold,1,0)
    return y_pred, y_predicted

def evaluation(true, pred):
    print('F1-score: ' + str(round(f1_score(true,pred),4)), '\n'
    'Precision: ' + str(round(precision_score(true,pred),4)), '\n'
    'Recall: ' + str(round(recall_score(true,pred),4)), '\n'
    'Cohen-Kappa: ' + str(round(cohen_kappa_score(true,pred),4)), '\n' 
    'Confusion matrix:' + '\n' + str(confusion_matrix(true,pred)))