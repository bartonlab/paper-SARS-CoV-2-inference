#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
import os
import subprocess
import shutil
import data_processing as dp

REF_TAG = 'EPI_ISL_402125'

def main(args):
    """  """
    
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',             type=str,    default=None,     help='output string')
    parser.add_argument('--input_dir',    type=str,    default=None,     help='input directory')
    parser.add_argument('--days',         type=int,    default=10,       help='the number of days to cut at the beginning of the time series')

    
    arg_list = parser.parse_args(args)
    
    out_file  = arg_list.o        
    in_file   = arg_list.input_dir
    days      = arg_list.days
    
    data = np.load(in_file, allow_pickle=True)
    nVec = data['nVec']
    sVec = data['sVec']
    muts = data['mutant_sites']
    time = data['times']
    
    nVec = nVec[days-1:]
    sVec = sVec[days-1:]
    time = time[days-1:]
    
    f = open(out_file, mode='wb')
    np.savez_compressed(out_file, times=time, mutant_sites=muts, nVec=nVec, sVec=sVec)
    f.close()
                        
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
    
    