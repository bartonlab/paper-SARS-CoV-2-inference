#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
import os
import shutil
import data_processing as dp

def main(args):
    """ Trims time-series to dates that have good sampling. """
    
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',             type=str,    default=None,     help='output directory')
    parser.add_argument('--input'     ,   type=str,    default=None,     help='input directory')
    parser.add_argument('--seq_dir',      type=str,    default=None,     help='the directory with the sequences if this is different from genome-data')
    parser.add_argument('--max_dt',       type=int,    default=5,        help='the maximum number of consecutive days with zero sequences allowed')
    parser.add_argument('--window',       type=int,    default=5,        help='the number of days within which min_seqs must be found')
    parser.add_argument('--minSeqs',      type=int,    default=20,       help='the minimum number of sequences that must be found in each window days')
    parser.add_argument('--minRange',     type=int,    default=20,       help='the shortest time-series allowed')
    parser.add_argument('--trimDir',      type=str,    default=None,     help='the directory to write the trimmed files to')
    parser.add_argument('--trim_exists',  default=False, action='store_true', help='whether or not the trimmed directory has already been made')
    
    arg_list = parser.parse_args(args)
    
    out_dir   = arg_list.o        
    in_file   = arg_list.input
    max_dt    = arg_list.max_dt
    window    = arg_list.window
    min_seqs  = arg_list.minSeqs
    min_range = arg_list.minRange
    seq_dir   = arg_list.seq_dir
    
    used_files = 'used-files'
    if not os.path.exists(used_files):
        os.mkdir(used_files)
    
    if arg_list.trimDir:
        trim_dir = os.path.join(out_dir, arg_list.trimDir)
    else:
        trim_dir = os.path.join(out_dir, 'genome-trimmed')
    if not arg_list.trim_exists:
        print('restricting intervals')
        dp.trim_time_series(in_file, trim_dir, window=window, max_dt=max_dt, min_seqs=min_seqs, min_range=min_range)
        f = open(os.path.join(used_files, os.path.split(in_file)[1]), mode='w')
        f.write('used')
        f.close()
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
