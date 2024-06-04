'''
Created on Dec 28, 2020

@author: jsun
'''

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

def calc_auc(y_true, y_score):
#     print('len(y_true) = {0}'.format(len(y_true)))
    none_missing = y_true != -1 # NOTE: -1 is considered missing in the data
    y_true = y_true[none_missing]
    if (len(np.unique(y_true)) < 2):
        return np.nan
    return roc_auc_score(y_true, y_score[none_missing])
    

def auc(y_true, y_score):
    return tf.py_function(calc_auc, [y_true, y_score], tf.double)
