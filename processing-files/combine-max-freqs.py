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
NUC     = ['-', 'A', 'C', 'G', 'T']

def combine_max_freq(max_dir, out_dir, out_file='maximum-frequencies'):
    """ Given a directory containing the maximum frequencies in each region (max_dir), 
    combines them to find the maximum frequency of an allele mutation in any region."""
    
    mutant_sites = []
    max_freqs    = []
    region_name  = []
    mut_counts   = []
    for file in sorted(os.listdir(max_dir)):
        data_temp = np.load(os.path.join(max_dir, file), allow_pickle=True)
        mutant_sites.append(data_temp['mutant_sites'])
        max_freqs.append(data_temp['max_frequency'])
        mut_counts.append(data_temp['mut_counts'])
        #print(np.amax(data_temp['max_frequency']))
        region_name.append(file)
    allele_number  = np.sort(np.unique([mutant_sites[i][j] for i in range(len(mutant_sites)) for j in range(len(mutant_sites[i]))]))
    max_freq_tot   = np.zeros(len(allele_number))
    #mut_counts_tot = np.zeros(len(allele_number), 4)
    mut_counts_tot = np.zeros(len(allele_number), 5)
    alleles_sorted = np.argsort(allele_number)
    for i in range(len(mutant_sites)):
        positions      = np.searchsorted(allele_number[alleles_sorted], mutant_sites[i])
        for j in range(len(mutant_sites[i])):
            max_freq_tot[positions[j]]   = max(max_freq_tot[positions[j]], max_freqs[i][j])
            mut_counts_tot[positions[j]] += mut_counts[positions[j]]    # remember that these do not include the counts for the reference sequence (newer versions should)
    #print('max_freq')
    #print(list(max_freq_tot))
    f = open(os.path.join(out_dir, out_file+'.npz'), mode='wb')
    np.savez_compressed(f, allele_number=allele_number, max_freq=max_freq_tot, mutant_counts=mut_counts_tot)
    f.close()

def main(args):
    """ determine the maximum frequency that each mutant site reaches in a given region"""
    
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',             type=str,    default=None,           help='output directory')
    parser.add_argument('--input_dir',    type=str,    default=None,           help='input dir')
    
    arg_list = parser.parse_args(args)
    
    out_dir   = arg_list.o        
    data_dir  = arg_list.input_dir   
    
    combine_max_freq(data_dir, out_dir)    
    
if __name__ == '__main__': 
    main(sys.argv[1:])