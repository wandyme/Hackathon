# -*- coding: utf-8 -*-
"""
This is the functions used in the Hackathon Challenge.

Function list：
   None [To be updated]  
"""
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing as prep

__author__ = "Wan Dongyang"
__copyright__ = "Copyright 2018, The Hackathon Project"
__credits__ = "Wan Dongyang"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Wan Dongyang"
__email__ = "Dongyang@u.nus.edu"
__status__ = "Production"


def computeCost(X, y, theta):
    """Cost (error) function"""
    inner = np.power(((X @ theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def normalEq(X,y):
    """Normal equation"""
    theta=np.linalg.pinv(X.T@X)@X.T@y
    return theta

def normalRegEq(X,y,beta):
    """normal equation with regulization"""
    L=np.eye(X.shape[1])
    L[0,0]=0
    theta=np.linalg.inv(X.T@X+beta*L)@X.T@y
    return theta

def formatTime(t):
    minutes, seconds_rem = divmod(t, 60)
    hours, minutes=divmod(minutes, 60)
    # use string formatting with C type % specifiers
    # %02d means integer field of 2 left padded with zero if needed
    return "%02d:%02d:%02d" % (hours, minutes, seconds_rem)


    
# def linearRegCostFunction(X, y, theta, reg):