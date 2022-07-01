#!/usr/bin/env python
# coding: utf-8
# %%

# %%

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
import data_processing as dp

REF_TAG = 'EPI_ISL_402125'
NUC     = ['-', 'A', 'C', 'G', 'T']

def find_site_index_file(filepath):
    """ Given a sequence file find the correct corresponding file that has the site names"""
    directory, file = os.path.split(filepath)
    return filepath[:-4] + '-sites.csv'


def find_counts_window(counts, window):
    ret = np.cumsum(counts, axis=0)
    ret[window:] = ret[window:] - ret[:-window]
    result = ret[window - 1:]
    return result


def moving_average(freq, window=9):
    """ Calculates a moving average for a frequency array. """
    ret = np.cumsum(freq, axis=0)
    ret[window:] = ret[window:] - ret[:-window]
    result = ret[window - 1:] / window
    return result


def get_data(file, get_seqs=True):
    """ Given a sequence file, get the sequences, dates, and mutant site labels"""
    print('sequence file\t', file)
    #data = pd.read_csv(file, engine='python', converters={'sequence': lambda x: str(x)})
    data = pd.read_csv(file, engine='python', dtype={'sequence': str})
    #if not get_seqs:
    #    sequences = None
    #else:
    #    sequences   = np.array([np.array(list(str(i)), dtype=int) for i in list(data['sequence'])])
    #    seq_lens, seq_counts = np.unique([len(i) for i in sequences], return_counts=True)
    #    print(f'sequence lengths are {seq_lens}')
    #    print(f'number of sequences with each length {seq_counts}')
    #    common_len = seq_lens[np.argmax(seq_counts)]
    #    mask  = [i for i in range(len(sequences)) if len(sequences[i])==common_len]
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


def find_freqs(file, var_muts, window=10, d=5):
    """ Finds the frequency trajectory of a variant in the region."""
    var_sites = [i[:-2]           for i in var_muts]
    var_nucs  = [NUC.index(i[-1]) for i in var_muts]
    
    df                = pd.read_csv(file, dtype={'sequence' : str})
    data              = get_data(file)
    sequences         = data['sequences']
    ref_sites         = data['ref_sites']
    dates             = data['dates']
    filename          = os.path.basename(file)
    sequences         = np.array([list(i) for i in sequences], dtype=int)
    #site_idxs = [ref_labels.index(str(i)) for i in ref_sites]
    #ref_poly  = np.array(ref_seq)[np.array(site_idxs)]
    
    # Better code for doing this
    unique_dates, num_seqs = np.unique(dates, return_counts=True)
    all_dates = np.arange(np.amin(unique_dates), np.amax(unique_dates) + 1)
    n_seqs    = np.zeros(len(all_dates))
    for i in range(len(all_dates)):
        if all_dates[i] in unique_dates:
            n_seqs[i] += num_seqs[list(unique_dates).index(all_dates[i])]
    num_seqs = n_seqs
    unique_dates = all_dates
    
    ref_muts = [dp.get_label_new(str(i) + '-A')[:-2] for i in ref_sites]
    if not all([i in ref_muts for i in var_sites]):
        return None, None
    var_idxs = [ref_muts.index(i) for i in var_sites ]
    #sorter   = 
    #if len(var_idxs)==0:
    #    return None, None
    if 'england' in file:
        window=0
    if window > len(unique_dates) / 2:
        window = int(len(unique_dates) / 2)
    
    counts = np.zeros(len(all_dates))
    for t in range(len(unique_dates)):
        idxs   = np.where(dates==unique_dates[t])[0]
        seqs_t = sequences[idxs]
        for j in range(len(seqs_t)):
            seq_var = np.array(seqs_t[j])[np.array(var_idxs)]
            if all(seq_var==np.array(var_nucs)):
                counts[t] += 1
    if window!=0:           
        ret = np.cumsum(counts, axis=0)
        ret[window:] = ret[window:] - ret[:-window]
        result = ret[window - 1:]
        
        # Find total number of sequences in each window from num_seqs
        n_seqs = np.zeros(len(result))
        for i in range(len(result)):
            n_temp = 0
            for j in range(window):
                n_temp += num_seqs[i + j]
            n_seqs[i] += n_temp
            
        counts       = result
        num_seqs     = n_seqs
        unique_dates = unique_dates[window - 1:]
        
    #counts = np.sum(counts_tot, axis=0)
    #freq = freqs_from_counts(counts_tot, num_seqs)
    new_labels = []
    for i in ref_sites:
        for j in NUC:
            new_labels.append(str(i) + '-' + j)
            
    freq = np.array([counts[i] / num_seqs[i] for i in range(len(counts))])
    return unique_dates, freq
    
    
delta = ['NSP12-671-0-A', 'NC-209-T', 'NSP13-77-1-T', 'S-19-1-G', 'S-478-1-A', 'S-681-1-G', 'S-950-0-A', 
         'ORF3a-26-1-T', 'M-82-1-C', 'ORF7a-82-1-C', 'ORF7a-120-1-T', 'N-63-1-G', 'N-203-1-T', 
         'N-377-0-T', 'S-452-1-G']  

ba2   = ['S-405-0-A', 'S-452-1-T']

lambda_ = ['NSP5-15-0-A', 'S-76-1-T', 'S-452-1-A', 'S-490-1-C', 'S-859-1-A']


def main(args):
    """Calculate the covariance matrix and the trajectories for a single region from SARS-CoV-2 data"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',            type=str,    default='parallel',              help='output directory')
    parser.add_argument('-q',            type=int,    default=5,                       help='number of mutant alleles per site')
    parser.add_argument('--data',        type=str,    default=None,                    help='.npz files containing the counts, sequences, times, and mutant_sites for different locations')
    parser.add_argument('--timed',       type=int,    default=0,                       help='if 0, wont print any time information, if 1 will print some information')
    parser.add_argument('--outflow',     type=str,    default=None,                    help='.npz file containing the counts and sequences that outflow')
    parser.add_argument('--inflow',      type=str,    default=None,                    help='.npz file containing the counts and sequences that inflow, and the locations')
    parser.add_argument('--end_cutoff',  type=int,    default=0,                       help='the number of days at the end of the simulations to ignore due to bad sampling')
    parser.add_argument('--start_cutoff',type=int,    default=0,                       help='the number of days at the end of the simulations to ignore due to bad sampling')
    parser.add_argument('--window',      type=int,    default=10,                      help='the number of days over which to take the moving average')
    parser.add_argument('--clip_end',    type=int,    default=29700,                   help='the last site to clip the genomes to')
    parser.add_argument('--clip_start',  type=int,    default=150,                     help='the first site to clip the genomes to')
    parser.add_argument('--c_directory', type=str,    default='Archive',               help='directory containing the c++ scripts')
    parser.add_argument('--mask_site',   type=int,    default=None,                    help='the site to mask (change to WT) when inferring the other coefficients, in order to check effect of linkage (write)')
    parser.add_argument('--mask_group',  type=str,    default=None,                    help='the .npy file containing a list of sites (as nucleotide numbers) to mask for the inference')
    parser.add_argument('--final_t',     type=int,    default=None,                    help='the last time point to be considered, cutoff all time series after this time (in days after 01-01-2020)')
    parser.add_argument('--decay_rate',  type=float,  default=0,                       help='the exponential decay rate used to correct for infection lasting multiple generations')
    parser.add_argument('--nm_popsize',  type=str,    default=None,                    help='.csv file containing the population sizes used to correct for infection lasting multiple generations')
    parser.add_argument('--delay',       type=int,    default=None,                    help='the delay between the newly infected individuals and the individuals that ')
    parser.add_argument('--variantMuts', type=str,    default=None,                    help='A file containing the mutations that are in the variant whose trajectory is to be calculated')
    parser.add_argument('--variant',     type=str,    default=None,                    help='the name of the variant')
    parser.add_argument('--mutation_on',    action='store_true', default=False,  help='whether or not to include mutational term in inference')
    parser.add_argument('--cutoff_on',      action='store_true', default=False,  help='whether or not to cutoff frequencies below freq_cutoff')
    parser.add_argument('--find_linked',    action='store_true', default=False,  help='whether or not to find the sites that are (almost) fully linked')
    parser.add_argument('--find_counts',    action='store_true', default=False,  help='whether to find the single and double site counts instead of the frequencies, mostly used for finding linked sites')
    
    arg_list = parser.parse_args(args)
    
    out_str       = arg_list.o
    q             = arg_list.q
    end_cutoff    = arg_list.end_cutoff
    start_cutoff  = arg_list.start_cutoff
    window        = arg_list.window
    timed         = arg_list.timed
    c_directory   = arg_list.c_directory
    clip_end      = arg_list.clip_end
    clip_start    = arg_list.clip_start
    mask_site     = arg_list.mask_site
    final_time    = arg_list.final_t
    decay_rate    = arg_list.decay_rate
    delay         = arg_list.delay
    #data          = np.load(arg_list.data, allow_pickle=True)
    if arg_list.variant is None:
        variant  = 'delta'
        var_muts = delta
    elif arg_list.variant.lower() == 'ba.2':
        variant  = arg_list.variant
        var_muts = ba2
    elif arg_list.variant.lower() == 'lambda':
        variant  = arg_list.variant
        var_muts = lambda_
    else:
        variant_file = arg_list.variantMuts
        variant      = os.path.split(variant_file)[-1][:-4]
        var_muts     = np.load(variant_file, allow_pickle=True)    
            
    if timed > 0:
        t_start = timer()
        print("starting")
    
    # Make status file that records the progress of the script
    status_name = f'traj-{variant}.csv'
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
    
    data_dir  = arg_list.data
    dates_all = []
    freqs_all = []
    locs_all  = []
    for file in os.listdir(data_dir):
        if 'sites' in file:
            continue
        location = file[:file.find('---')]
        #location_data  = location.split('-')
        #timestamps     = location_data[-6:]
        #for i in range(len(timestamps)):
        #    if len(timestamps[i]) == 1:
        #        timestamps[i] = '0' + timestamps[i]
    
        #region = location[:-22]
        #if   region[-2:] == '--': region = region[:-2]
        #elif region[-1]  =='-':   region = region[:-1]

        dates, freq = find_freqs(os.path.join(data_dir, file), var_muts, window=window)
        if dates is None:
            continue
        dates_all.append(' '.join([str(i) for i in dates]))
        freqs_all.append(' '.join([str(i) for i in freq]))
        locs_all.append(location)

    
    dic = {'times' : dates_all, 
           'frequencies' : freqs_all,
           'location' : locs_all}
    df = pd.DataFrame.from_dict(dic)
    df.to_csv(out_str + '.csv', index=False)

        
    

    
    
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    
    
    

