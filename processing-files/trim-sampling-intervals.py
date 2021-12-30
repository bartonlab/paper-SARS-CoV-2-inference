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
from multiprocessing import Pool

REF_TAG     = 'EPI_ISL_402125'

def main(args):
    """ Trims time-series to dates that have good sampling. """
    
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',             type=str,    default=None,     help='output directory')
    parser.add_argument('--input_dir',    type=str,    default=None,     help='input directory')
    parser.add_argument('--seq_dir',      type=str,    default=None,     help='the directory with the sequences if this is different from genome-data')
    parser.add_argument('--max_dt',       type=int,    default=5,        help='the maximum number of consecutive days with zero sequences allowed')
    parser.add_argument('--window',       type=int,    default=5,        help='the number of days within which min_seqs must be found')
    parser.add_argument('--min_seqs',     type=int,    default=20,       help='the minimum number of sequences that must be found in each window days')
    parser.add_argument('--min_range',    type=int,    default=20,       help='the shortest time-series allowed')
    parser.add_argument('--trim_dir',     type=str,    default=None,     help='the directory to write the trimmed files to')
    parser.add_argument('--trim_exists',  default=False, action='store_true', help='whether or not the trimmed directory has already been made')
    parser.add_argument('--freq_cutoff',  default=False, action='store_true', help='whether or not to make files cutting off low frequency sites')
    
    arg_list = parser.parse_args(args)
    
    out_dir   = arg_list.o        
    in_file   = arg_list.input_dir
    max_dt    = arg_list.max_dt
    window    = arg_list.window
    min_seqs  = arg_list.min_seqs
    min_range = arg_list.min_range
    seq_dir   = arg_list.seq_dir
    
    #dp.restrict_sampling_intervals(in_file, out_file, max_dt=max_dt, min_seqs=min_seqs, min_range=min_range, window=window)
    #dp.eliminate_low_freq(out_file, '/'.join(out_file.split('/')[:-1])+'freq_0.05')
    #dp.eliminate_low_freq(in_file, out_file, freq_tol=0.01)
    
    ref_seq, ref_tag = dp.get_MSA(REF_TAG + '.fasta')
    ref_seq          = list(ref_seq[0])
    
    if seq_dir: genome_dir = os.path.join(out_dir, seq_dir)
    else:       genome_dir = os.path.join(out_dir, 'genome-data')
    if arg_list.trim_dir:
        trim_dir = os.path.join(out_dir, arg_list.trim_dir)
    else:
        trim_dir = os.path.join(out_dir, 'genome-trimmed')
    if not arg_list.trim_exists:
        print('restricting intervals')
        dp.restrict_sampling_intervals(genome_dir, trim_dir, window=window, max_dt=max_dt, min_seqs=min_seqs, min_range=min_range)
    
    if arg_list.freq_cutoff:
        print('eliminating sites that dont rise above 5 percent')
        freq_dir1 = os.path.join(out_dir, 'freq_0.05')
        dp.eliminate_low_freq_multisite(trim_dir, freq_dir1, ref_seq=list(ref_seq), freq_tol=0.05)
    
        print('eliminating sites that dont rise above 1 percent')
        freq_dir2 = os.path.join(out_dir, 'freq_0.01')
        dp.eliminate_low_freq_multisite(trim_dir, freq_dir2, ref_seq=list(ref_seq), freq_tol=0.01)
    
        print('eliminating sites that dont rise above 10 percent')
        freq_dir3 = os.path.join(out_dir, 'freq_0.1')
        dp.eliminate_low_freq_multisite(trim_dir, freq_dir3, ref_seq=list(ref_seq), freq_tol=0.1)
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])