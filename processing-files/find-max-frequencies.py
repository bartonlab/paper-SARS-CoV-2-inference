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

def get_MSA(ref, noArrow=True):
    """Take an input FASTA file and return the multiple sequence alignment, along with corresponding tags. """
    
    temp_msa = [i.split('\n') for i in open(ref).readlines()]
    temp_msa = [i for i in temp_msa if len(i)>0]
    
    msa = []
    tag = []
    
    for i in temp_msa:
        if i[0][0]=='>':
            msa.append('')
            if noArrow: tag.append(i[0][1:])
            else: tag.append(i[0])
        else: msa[-1]+=i[0]
    
    msa = np.array(msa)
    
    return msa, tag


def moving_average(freq, window=9):
    """ Calculates a moving average for a frequency array. """
    ret = np.cumsum(freq, axis=0)
    ret[window:] = ret[window:] - ret[:-window]
    result = ret[window - 1:] / window
    return result


def find_max_freq(file, out_dir, ref_seq=None, window=9, d=5):
    """ Finds the maximum frrequency that mutant alleles at each site reach."""
    
    data              = np.load(file, allow_pickle=True)
    nVec              = data['nVec']
    sVec              = data['sVec']
    mutant_sites_samp = data['mutant_sites']
    filename          = os.path.basename(file)
    
    if not ref_seq:
        ref_seq = get_ref_seq('./bigdata/SARS-CoV-2-Data/merged-final.csv')
    ref_poly = np.array(ref_seq)[mutant_sites_samp]
    Q = np.ones(len(nVec))
    
    
    for t in range(len(nVec)):
        if len(nVec[t]) > 0:
            Q[t] = np.sum(nVec[t])
    """
    max_freq = np.zeros((len(mutant_sites_samp), d))
    for i in range(len(mutant_sites_samp)):
        for k in range(min(window, len(nVec))):
            for j in range(len(sVec[k])):
                max_freq[i, sVec[k][j][i]] += nVec[k][j] / (window * Q[k])
    new_freq = np.zeros((len(mutant_sites_samp), d))
    for i in range(len(mutant_sites_samp)):
        for k in range(min(window, len(nVec))):
            for j in range(len(sVec[k])):
                new_freq[i, sVec[k][j][i]] += nVec[k][j] / (window * Q[k])
    for t in range(1, len(nVec) - window):
        for i in range(len(mutant_sites_samp)):
            for j in range(len(sVec[t+window])):
                new_freq[i, sVec[t+window][j][i]] += nVec[t+window][j] / (window * Q[t+window])
            for j in range(len(sVec[t])):
                new_freq[i, sVec[t][j][i]] -= nVec[t][j] / (window * Q[t])
        max_freq = np.maximum(max_freq, new_freq)
    """    
    
    # Better code for doing this
    counts = np.zeros((len(mutant_sites_samp), d))
    freq   = np.zeros((len(nVec), len(mutant_sites_samp), d))
    for t in range(len(nVec)):
        for i in range(len(mutant_sites_samp)):
            for j in range(len(sVec[t])):
                freq[t, i, sVec[t][j][i]] += nVec[t][j] / Q[t]
                counts[i, sVec[t][j][i]]  += nVec[t][j]
    freq = moving_average(freq, window=window)
    max_freq = np.amax(freq, axis=0)
    max_mut_freqs = []
    mut_counts    = []
    #lengths       = []
    for i in range(len(mutant_sites_samp)):
        mutant_freqs = [max_freq[i, j] for j in range(len(max_freq[i])) if NUC[j]!=ref_poly[i]]
        max_mut_freqs.append(np.amax(mutant_freqs))
        #mut_counts.append([counts[i][j] for j in range(len(max_freq[i])) if NUC[j]!=ref_poly[i]])
        mut_counts.append([counts[i][j] for j in range(len(max_freq[i]))])
        #lengths.append(len(mutant_freqs))
    #print(lengths)
    
    f = open(os.path.join(out_dir, filename), mode='wb')
    np.savez_compressed(f, mutant_sites=mutant_sites_samp, max_frequency=max_mut_freqs, mut_counts=mut_counts)
    f.close()

def main(args):
    """ determine the maximum frequency that each mutant site reaches in a given region"""
    
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',             type=str,    default=None,           help='output directory')
    parser.add_argument('--input_file',   type=str,    default=None,           help='input file')
    
    arg_list = parser.parse_args(args)
    
    out_dir   = arg_list.o        
    in_file   = arg_list.input_file       
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print(in_file)
                        
    ref_seq, ref_tag = get_MSA(REF_TAG + '.fasta')
    ref_seq          = ref_seq[0]
    
    find_max_freq(in_file, out_dir, ref_seq=list(ref_seq))    
    
if __name__ == '__main__': 
    main(sys.argv[1:])