# LIBRARIES

import numpy as np
import pandas as pd
import datetime as dt
import os
import sys
import argparse
from timeit import default_timer as timer   # timer for performance
import data_processing as dp
import shutil

# GLOBAL VARIABLES
REF_TAG = 'EPI_ISL_402125'
TIME_INDEX = 1


status_name = 'genome-move-files.csv'
stat_file   = open(status_name, 'w')
stat_file.close()


def main():
    """ Given the desired regions to be analyzed, collects the sequences from those regions and in the given time frames and then filters out 
    sequences and sites that have too many gaps."""
    
    
    parser = argparse.ArgumentParser(description='filtering out sequences and sites with too many gaps')
    parser.add_argument('--input_file',  type=str,    default=None,                       help='sequence alignment file')
    parser.add_argument('-o',            type=str,    default='regions',                  help='output destination')
    parser.add_argument('--regions',     type=str,    default='regional-data.csv.gz',     help='the regions and times to extract data from')
    parser.add_argument('--maxSeqs',     type=int,    default=20000,                     help='maximum number of sequences in a file')
    parser.add_argument('--find_syn_off', default=False,  action='store_true',            help='whether or not to determine which sites are synonymous and not')
    parser.add_argument('--full_index',   default=False,  action='store_true',            help='whether or not to calculate the full index for sites')
    parser.add_argument('--no_trim',      default=False,  action='store_true',            help='whether or not to trim sites based on frequency')
    
    arg_list = parser.parse_args(sys.argv[1:])
    out_dir      = arg_list.o
    input_file   = arg_list.input_file
    find_syn_off = arg_list.find_syn_off
    full_index   = arg_list.full_index
    maxSeqs      = arg_list.maxSeqs
    
    root = 'bigdata/SARS-CoV-2-Data/2022-05-10'
    dir1 = os.path.join(root, 'genome-trimmed2')
    dir_from = os.path.join(root, 'freqs')
    dir_to   = os.path.join(root, 'freqs2')
    locs = []
    for file in os.listdir(dir1):

    



if __name__ == '__main__':
    main()
