# -*- coding: utf-8 -*-
"""
Print iterations progress and estimated time cost.
"""
import time, sys

__author__ = "Wan Dongyang"
__copyright__ = "Copyright 2018, The Hackathon Project"
__credits__ = "Wan Dongyang"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Wan Dongyang"
__email__ = "Dongyang@u.nus.edu"
__status__ = "Production"

class progressBar:
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """

    def __init__(self, total, prefix = '', suffix = '', decimals = 2, length = 100, fill = '█'):
        self.total=total
        self.prefix=prefix
        self.suffix=suffix
        self.decimals=decimals
        self.length=length
        self.fill=fill
        self.update(0)
        
    def update(self, iteration, ToPrint=True):
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(self.total)))
        filledLength = int(self.length * iteration // self.total)
        bar = self.fill * filledLength + '-' * (self.length - filledLength)
        self.string='%s |%s| %s%% %s' % (self.prefix, bar, percent, self.suffix)
        if ToPrint==True:
            print(self.string, end='\r')
            # Print New Line on Complete
            if iteration == self.total: 
                print()
        return self.string
            
class timer:
    def __init__(self, total):
        self.t_start=time.time()
        self.total=total
    
    def update(self, iteration, ToPrint=True):
        self.now=time.time()
        self.timerun=self.now-self.t_start
        self.eTotalTime=self.total*(self.timerun)/iteration
        self.eTimeleft=self.eTotalTime-self.timerun
        timerun_f=self._formatTime(self.timerun)
        eTotalTime_f=self._formatTime(self.eTotalTime)
        eTimeleft_f=self._formatTime(self.eTimeleft)
        self.string='Run: %s; Left: %s (Total: %s)'% (timerun_f, eTimeleft_f, eTotalTime_f)
        if ToPrint==True:
            print(self.string, end='\r')
            # Print New Line on Complete
            if iteration == self.total: 
                print()
        return self.string
        
    @staticmethod
    def _formatTime(t):
        minutes, seconds_rem = divmod(t, 60)
        hours, minutes=divmod(minutes, 60)
        # use string formatting with C type % specifiers
        # %02d means integer field of 2 left padded with zero if needed
        return "%02d:%02d:%02d" % (hours, minutes, seconds_rem)
    
            
            
            
            

    
# 
# Sample Usage
# 

# from time import sleep

# # A List of Items
# items = list(range(0, 57))
# l = len(items)

# # Initial call to print 0% progress
# printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
# for i, item in enumerate(items):
#     # Do stuff...
#     sleep(0.1)
#     # Update Progress Bar
#     ProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)