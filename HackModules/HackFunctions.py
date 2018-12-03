# -*- coding: utf-8 -*-
"""
This is the functions used in the Hackathon Challenge.

Function list：
   None [To be updated]  
"""
import numpy as np
import HackModules.HackClasses as hc
from HackModules.progressMonitor import progressBar
from HackModules.progressMonitor import timer
from multiprocessing import Pool

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

def train_model(tcv, degree, beta, random_int=999, rep=True):
#     print(f'Loop {i}-{index}-{j} is running...\n')
    
    # Split into train set and cv set
    if rep==True:
        train=tcv.sample(frac=0.75, random_state = random_int)
        cv=tcv.drop(train.index)
    else:
        train=tcv.sample(frac=0.75) 
        cv=tcv.drop(train.index)
        
    train_data=hc.data(train.loc[:,['V','T']], degree=degree, method ='std')
    train_unc=train.uncertainty.values
    train_J=train.J.values

    cv_data=hc.data(cv.loc[:,['V','T']], degree=degree, method ='std')
    cv_unc=cv.uncertainty.values
    cv_J=cv.J.values

    theta=normalRegEq(train_data.norm, train_J, beta)
    error_train_smpl=computeCost(train_data.norm, train_J, theta)
    error_cv_smpl=computeCost(cv_data.norm, cv_J, theta)
#     print(theta)
    return theta, error_train_smpl, error_cv_smpl  
#   print(f'Loop {i}-{index}-{j} ended.\n')

def train_model_multismpl(tcv, epoch, degree, beta, rep=True, random_int=999, multiprocess='OFF', cpu_num=1):
    error_train_multismpl=np.zeros(epoch)
    error_cv_multismpl=np.zeros(epoch)
    try:
        # Calculate the mean value of error_cv_reg with differet sets of train data samples
        if multiprocess== 'OFF':
            for j in range(0,epoch,1):
                random_int=int((157*j+random_int)/3) 
                _, error_train_multismpl[j], error_cv_multismpl[j]=train_model(tcv, degree, beta, random_int, rep)
        elif multiprocess == 'ON':
            pool=Pool(processes=cpu_num)
            results=[]
            for j in range(0,epoch,1):
                random_int=int((157*j+random_int)/3)
                results.append(pool.apply_async(train_model, (tcv, degree, beta, random_int, rep)))
            pool.close()
            pool.join()
                                                
            for j, res in enumerate(results):
                error_train_multismpl[j]=res.get()[1]
                error_cv_multismpl[j]=res.get()[2]
        else:
            raise hc.ValueException('multiprocess should be either \'ON\' or \'OFF\'(case sensitive)', multiprocess)
    except hc.ValueException as ex:
        print(f'The error is: {ex.msg}, here the input multiprocess is \'{ex.value}\'')
        return  # stop the whole execuation of the this function.
    return error_train_multismpl, error_cv_multismpl
                     

# This can be used to see the variation of beta during the training
def train_loop(tcv, train_num, epoch, v_range, other_v, n_v='beta', rep=True, multiprocess='OFF', cpu_num=1, show=True):
    
    # theta_reg_multismpl=np.zeros((epoch, featureSize)a])
    error_train_multitrain=np.zeros((train_num, len(v_range)))
    error_cv_multitrain=np.zeros((train_num, len(v_range)))
    v_array=np.zeros(train_num)
    if show==True:
        # Initial call to print 0% progress
        total=train_num*len(v_range)
        pbar=progressBar(total, prefix = 'Progress:', suffix = 'Complete', decimals = 2, length = 50)
        tm=timer(total)
    
    # Calculate beta multiple times and find the mean value of them as best beta
    for i in range(0,train_num): 
        
        # Get beta_best by find the minimum error_cv_reg
        for index, v in enumerate(v_range):
            if n_v == 'beta':
                degree=other_v
                beta=v
            elif n_v == 'degree':
                degree=v
                beta=other_v
                     
            error_train_multismpl, error_cv_multismpl=train_model_multismpl(tcv, epoch, degree, beta, rep=rep,
                        random_int=train_num*epoch*len(v_range)+71*i+98*index, multiprocess=multiprocess, cpu_num=cpu_num)   
            
            # show the progess and timer
            if show==True:
                iteration=i*len(v_range)+index+1
                s1=pbar.update(iteration, ToPrint=False)
                s2=tm.update(iteration,ToPrint=False)
                print('\r'+s1+" "*5+s2,end='\r')
                if iteration==total:
                    print()
            
            error_train_multitrain[i, index] = error_train_multismpl.mean(0)
            error_cv_multitrain[i, index] = error_cv_multismpl.mean(0)

        idx=error_cv_multitrain[i,:].argmin()
        v_array[i] = v_range[idx]

    error_train=error_train_multitrain.mean(0)
    error_cv=error_cv_multitrain.mean(0)
                                  
    return v_array, error_train, error_cv


def theta_multismpl(tcv, epoch, degree, beta, featureSize, rep=True, random_int=999, multiprocess='OFF', cpu_num=1):
    error_train_multismpl=np.zeros(epoch)
    error_cv_multismpl=np.zeros(epoch)
    theta_multismpl=np.zeros((epoch, featureSize))

    try:
        # Calculate the mean value of error_cv_reg with differet sets of train data samples
        if multiprocess== 'OFF':
            for j in range(0,epoch,1):
                random_int=int((157*j+random_int)/3) 
                theta_multismpl[j,:], error_train_multismpl[j], error_cv_multismpl[j]=train_model(tcv, degree, beta,
                                                                                                  random_int, rep)
#                 print(theta_multismpl)
        elif multiprocess == 'ON':
            pool=Pool(processes=cpu_num)
            results=[]
            for j in range(0,epoch,1):
                random_int=int((157*j+random_int)/3)
                results.append(pool.apply_async(train_model, (tcv, degree, beta, random_int, rep)))
            pool.close()
            pool.join()
            
                                                          
            for j, res in enumerate(results):
                theta_multismpl[j,:]=res.get()[0]
                error_train_multismpl[j]=res.get()[1]
                error_cv_multismpl[j]=res.get()[2]
                       
        else:
            raise hc.ValueException('multiprocess should be either \'ON\' or \'OFF\'(case sensitive)', multiprocess)
    except hc.ValueException as ex:
        print(f'The error is: {ex.msg}, here the input multiprocess is \'{ex.value}\'')
        return  # stop the whole execuation of the this function.
    return theta_multismpl, error_train_multismpl, error_cv_multismpl
        
#     return error_train_poly, error_cv_poly


        
        
        
        

        
        
        
        
        
        



    
# def linearRegCostFunction(X, y, theta, reg):