#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Check what the above means

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
import os
import math
import datetime as dt
import subprocess
import pandas as pd
import shutil

REF_TAG = 'EPI_ISL_402125'
NUC     = ['-', 'A', 'C', 'G', 'T']

def find_site_index_file(filepath):
    """ Given a sequence file find the correct corresponding file that has the site names"""
    directory, file = os.path.split(filepath)
    return filepath[:-4] + '-sites.csv'


def get_data(file, get_seqs=True):
    """ Given a sequence file, get the sequences, dates, and mutant site labels"""
    print('sequence file\t', file)
    data = pd.read_csv(file, engine='python', dtype={'sequence': str})
    dates     = list(data['date'])
    sub_dates = list(data['submission_date'])
    index_file = find_site_index_file(file)
    index_data = pd.read_csv(index_file)
    ref_sites  = list(index_data['ref_sites'])
    if get_seqs:
        sequences   = np.array([np.array(list(str(i)), dtype=int) for i in list(data['sequence'])])
        seq_lens, seq_counts = np.unique([len(i) for i in sequences], return_counts=True)
        print(f'sequence lengths are {seq_lens}')
        print(f'number of sequences with each length {seq_counts}')
        common_len = seq_lens[np.argmax(seq_counts)]
        mask       = [i for i in range(len(sequences)) if len(sequences[i])==common_len]
        dates      = list(np.array(dates)[mask])
        sub_dates  = list(np.array(sub_dates)[mask])
        sequences  = sequences[mask]
        assert(len(ref_sites)==common_len)
    else:
        sequences = None
    dic = {
        'ref_sites' : ref_sites,
        'sequences' : sequences,
        'dates' : dates,
        'submission_dates' : sub_dates
    }
    return dic


def find_freqs(file, window=10, d=5):
    """ Finds the maximum frrequency that mutant alleles at each site reach."""
    df                = pd.read_csv(file, dtype={'sequence' : str})
    data              = get_data(file)
    sequences         = data['sequences']
    ref_sites         = data['ref_sites']
    dates             = data['dates']
    filename          = os.path.basename(file)
    sequences         = np.array([list(i) for i in sequences], dtype=int)
    
    # Better code for doing this
    unique_dates, num_seqs = np.unique(dates, return_counts=True)
    all_dates = np.arange(np.amin(unique_dates), np.amax(unique_dates) + 1)
    n_seqs    = np.zeros(len(all_dates))
    for i in range(len(all_dates)):
        if all_dates[i] in unique_dates:
            n_seqs[i] += num_seqs[list(unique_dates).index(all_dates[i])]
    num_seqs = n_seqs
    unique_dates = all_dates
    min_seqs    = 100
    if window >= len(unique_dates) / 2:
        window = int(len(unique_dates) / 2)
    if np.amin(num_seqs) > min_seqs:
        window = 0
    #window_test = 1
    #while np.amin(find_counts_window(num_seqs, window_test)) < min_seqs and window_test < window:
    #    window_test += 1
    #window = window_test
    counts_tot = np.zeros((len(unique_dates), len(ref_sites) * d))
    for t in range(len(unique_dates)):
        idxs   = np.where(dates==unique_dates[t])[0]
        seqs_t = sequences[idxs]
        for i in range(len(ref_sites)):
            for j in range(len(seqs_t)):
                counts_tot[t, (i * d) + seqs_t[j][i]] += 1
    if window!=0:           
        ret = np.cumsum(counts_tot, axis=0)
        ret[window:] = ret[window:] - ret[:-window]
        result = ret[window - 1:]
        
        # Find total number of sequences in each window from num_seqs
        n_seqs = np.zeros(len(result))
        for i in range(len(result)):
            n_temp = 0
            for j in range(window):
                n_temp += num_seqs[i + j]
            n_seqs[i] += n_temp
            
        counts_tot   = result
        num_seqs     = n_seqs
        unique_dates = unique_dates[window - 1:]
        
    new_labels = []
    for i in ref_sites:
        for j in NUC:
            new_labels.append(str(i) + '-' + j)
            
    freq = np.array([counts_tot[i] / num_seqs[i] for i in range(len(counts_tot))])
    dic  = {'time' : unique_dates}
    for i in range(len(new_labels)):
        dic[new_labels[i]] = freq[:, i]
    df = pd.DataFrame.from_dict(dic)
    return df
    

def main(args):
    """Calculate the covariance matrix and the trajectories for a single region from SARS-CoV-2 data"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',            type=str,    default='parallel',              help='output directory')
    parser.add_argument('-q',            type=int,    default=5,                       help='number of mutant alleles per site')
    parser.add_argument('--data',        type=str,    default=None,                    help='.npz files containing the counts, sequences, times, and mutant_sites for different locations')
    parser.add_argument('--timed',       type=int,    default=0,                       help='if 0, wont print any time information, if 1 will print some information')
    parser.add_argument('--window',      type=int,    default=10,                      help='the number of days over which to take the moving average')

    arg_list = parser.parse_args(args)
    
    out_str       = arg_list.o
    q             = arg_list.q
    window        = arg_list.window
    timed         = arg_list.timed
    input_str     = arg_list.data
    
    if not os.path.exists(out_str):
        os.mkdir(out_str)
        
    working_dir = os.getcwd()
    identifier  = out_str.split('/')
    if len(identifier) > 1:
        identifier = identifier[-1]
    else:
        identifier = identifier[0]
            
    if timed > 0:
        t_start = timer()
        print("starting")
        
    # Load data
    filepath = input_str
    location = os.path.split(filepath)[-1][:-4]
    
    # Make status file that records the progress of the script
    status_name = f'traj-{location}.csv'
    status_file = open(status_name, 'w')
    status_file.close()
    
    def print2(*args):
        """ Print the status of the processing and save it to a file."""
        stat_file = open(status_name, 'a+')
        line      = [str(i) for i in args]
        string    = '\t'.join(line)
        stat_file.write(string+'\n')
        stat_file.close()
        print(string)
    
    # get data for the specific location
    location_data  = location.split('-')
    timestamps     = location_data[-6:]
    for i in range(len(timestamps)):
        if len(timestamps[i]) == 1:
            timestamps[i] = '0' + timestamps[i]
        
    region = location[:-22]
    if   region[-2:] == '--': region = region[:-2]
    elif region[-1]  =='-':   region = region[:-1]
        
    # make trajectories
    df = find_freqs(arg_list.data, window=window)
    filename = os.path.split(arg_list.data)[-1]
    df.to_csv(os.path.join(out_str, filename), index=False)

        
    

    
    
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    
    
    

