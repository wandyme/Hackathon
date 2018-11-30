# -*- coding: utf-8 -*-
"""
This is the classes used in the Hackathon Challenge.

Class list:
    - MethodException
    - Data
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


class MethodException(Exception):
    def __init__(self, msg, value):
        self.msg=msg
        self.value=value
        
class data():
    def __init__(self, array, autoPoly = True, autoNorm= True, degree=1, method='std'):
        """The default degree of the poly array is 1."""
        if type(array) == pd.core.frame.DataFrame:
            self.array = array.values
        elif type(array) == np.ndarray:
            self.array = array
        self.autoPoly=autoPoly
        self.autoNorm=autoNorm
        if self.autoPoly==True:
            self.polyFeature(degree)
        
    def polyFeature(self, degree=1):
        polyMethod = prep.PolynomialFeatures(degree, include_bias = False)
        self.poly = polyMethod.fit_transform(self.array)
        if self.autoNorm==True:
            self.normalizeFeature('std')
        return self.poly
    
    def normalizeFeature(self, method='std'):
        X=self.poly
        try:
            if X.ndim == 1:  # Reshape n elements 1d array to [n,1] 2d array.
                X=np.reshape(X,(-1,1)) 
            self.norm=np.ones((X.shape[0],X.shape[1]+1), dtype=np.float64)
            if method == 'std':
                self.norm[:,1:]=(X-X.mean(0))/X.std(0)     
            elif method == 'range':
                self.norm[:,1:]=(X-X.min(0))/(X.max(0)-X.min(0))
            else:
                raise MethodException('method should be either \'std\' or \'range\'(case sensitive)', method)
        except MethodException as ex:
            print(f'The error is: {ex.msg}, here the input method is \'{ex.value}\'')
        else:
            return self.norm