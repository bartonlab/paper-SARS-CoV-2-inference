#!/usr/bin/env python
# coding: utf-8
# %%

# %%

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
import os
import subprocess
import shutil
import data_processing as dp
import pandas as pd

REF_TAG = 'EPI_ISL_402125'
NUC     = ['-', 'A', 'C', 'G', 'T']

# _ _ _ Functions for reading data _ _ _ #

def find_site_index_file(filepath):
    """ Given a sequence file find the correct corresponding file that has the site names"""
    directory, file = os.path.split(filepath)
    if file.find('---')==-1:
        return filepath[:-4] + '-sites.csv'
    else:
        return filepath[:filepath.find('---')] + '-sites.csv'   
    
    
def get_data(file, get_seqs=True):
    """ Given a sequence file, get the sequences, dates, and mutant site labels"""
    data      = pd.read_csv(file)
    if not get_seqs:
        sequences = None
    else:
        sequences = np.array([list(i) for i in list(data['sequence'])])
    dates     = list(data['date'])
    sub_dates = list(data['submission date'])
    index_file = find_site_index_file(file)
    index_data = pd.read_csv(index_file)
    mut_sites  = list(index_data['mutant_sites'])
    ref_sites  = list(index_data['ref_sites'])
    dic = {
        'ref_sites' : ref_sites,
        'mutant_sites' : mut_sites,
        'sequences' : sequences,
        'dates' : dates,
        'submission_dates' : sub_dates
    }
    return dic

# ^ ^ ^ Functions for reading data ^ ^ ^


def find_counts_window(counts, window):
    ret = np.cumsum(counts, axis=0)
    ret[window:] = ret[window:] - ret[:-window]
    result = ret[window - 1:]
    return result


def find_max_freq(file, out_dir, ref_seq=None, ref_labels=None, window=None, d=5):
    """ Finds the maximum frrequency that mutant alleles at each site reach."""
    df                = pd.read_csv(file)
    data              = get_data(file)
    sequences         = data['sequences']
    #mutant_sites_samp = data['mutant_sites']
    ref_sites         = data['ref_sites']
    mutant_sites_samp = ref_sites
    dates             = data['dates']
    filename          = os.path.basename(file)
    sequences         = np.array([list(i) for i in sequences], dtype=int)
    site_idxs = [ref_labels.index(str(i)) for i in ref_sites]
    ref_poly  = np.array(ref_seq)[np.array(site_idxs)]
    
    # Better code for doing this
    unique_dates, num_seqs = np.unique(dates, return_counts=True)
    #min_seqs    = 20
    #window_test = 1
    #while np.amin(find_counts_window(num_seqs, window_test)) < min_seqs and window_test < window:
    #    window_test += 1
    #window = window_test
    counts_tot = np.zeros((len(unique_dates), len(mutant_sites_samp), d))
    for t in range(len(unique_dates)):
        idxs   = np.where(dates==unique_dates[t])[0]
        seqs_t = sequences[idxs]
        for i in range(len(mutant_sites_samp)):
            for j in range(len(seqs_t)):
                counts_tot[t, i, seqs_t[j][i]] += 1
    counts = np.sum(counts_tot, axis=0)
    #freq = freqs_from_counts(counts_tot, num_seqs)
    freq = np.array([counts_tot[i] / num_seqs[i] for i in range(len(counts_tot))])
    mut_freqs  = np.zeros((len(unique_dates), len(mutant_sites_samp), d-1))
    mut_counts = []
    for i in range(len(mutant_sites_samp)):
        ref_idx = NUC.index(ref_poly[i])
        mut_freqs[:, i, :] = np.delete(freq[:, i, :], ref_idx, axis=1)
        #mutant_freqs = [freq[:, i, j] for j in range(len(max_freq[i])) if NUC[j]!=ref_poly[i]]
        #max_mut_freqs.append(np.amax(mutant_freqs))
        mut_counts.append([counts[i][j] for j in range(len(counts[i]))])
    
    f = open(os.path.join(out_dir, filename), mode='wb')
    np.savez_compressed(f, mutant_sites=mutant_sites_samp, frequency=mut_freqs, mut_counts=mut_counts, ref_sites=ref_sites)
    f.close()

def main(args):
    """ determine the maximum frequency that each mutant site reaches in a given region"""
    
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',             type=str,    default=None,           help='output directory')
    parser.add_argument('--input_file',   type=str,    default=None,           help='input file')
    parser.add_argument('--refFile',      type=str,    default='ref-index.csv', help='reference index file')
    
    arg_list = parser.parse_args(args)
    
    out_dir   = arg_list.o        
    in_file   = arg_list.input_file       
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print(in_file)
                        
    #ref_seq, ref_tag = get_MSA(REF_TAG + '.fasta')
    #ref_seq          = ref_seq[0]
    ref_index = pd.read_csv(arg_list.refFile)
    index     = list(ref_index['ref_index'])
    index     = [str(i) for i in index]
    ref_nucs  = list(ref_index['nucleotide'])
    
    find_max_freq(in_file, out_dir, ref_seq=ref_nucs, ref_labels=index)    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
