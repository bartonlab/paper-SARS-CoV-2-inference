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


def find_freqs(file, window=10, d=5):
    """ Finds the maximum frrequency that mutant alleles at each site reach."""
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
        
    #counts = np.sum(counts_tot, axis=0)
    #freq = freqs_from_counts(counts_tot, num_seqs)
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


def get_label(i, d=5):
    """ For a SARS-CoV-2 reference sequence index i, return the label in the form 'coding region - protein number'. 
    For example, 'ORF1b-204'."""
    i_residue = str(i % d)
    i = int(i) / d
    frame_shift = str(i - get_codon_start_index(i))
    if   (25392<=i<26220):
        return "ORF3a-" + str(int((i - 25392) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
    elif (26244<=i<26472):
        return "E-"     + str(int((i - 26244) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
    elif (27201<=i<27387):
        return "ORF6-"  + str(int((i - 27201) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
    elif (27393<=i<27759):
        return "ORF7a-" + str(int((i - 27393) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
    elif (27755<=i<27887):
        return "ORF7b-" + str(int((i - 27755) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
    elif (  265<=i<805):
        return "NSP1-"  + str(int((i - 265  ) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
    elif (  805<=i<2719):
        return "NSP2-"  + str(int((i - 805  ) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
    elif ( 2719<=i<8554):
        return "NSP3-"  + str(int((i - 2719 ) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
            # Compound protein containing a proteinase, a phosphoesterase transmembrane domain 1, etc.
    elif ( 8554<=i<10054):
        return "NSP4-"  + str(int((i - 8554 ) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
            # Transmembrane domain 2
    elif (10054<=i<10972):
        return "NSP5-"  + str(int((i - 10054) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
            # Main proteinase
    elif (10972<=i<11842):
        return "NSP6-"  + str(int((i - 10972) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
            # Putative transmembrane domain
    elif (11842<=i<12091):
        return "NSP7-"  + str(int((i - 11842) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
    elif (12091<=i<12685):
        return "NSP8-"  + str(int((i - 12091) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
    elif (12685<=i<13024):
        return "NSP9-"  + str(int((i - 12685) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
            # ssRNA-binding protein
    elif (13024<=i<13441):
        return "NSP10-" + str(int((i - 13024) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
            # CysHis, formerly growth-factor-like protein
    # Check that aa indexing is correct for NSP12, becuase there is a -1 ribosomal frameshift at 13468. It is not, because the amino acids in the first frame
    # need to be added to the counter in the second frame.
    elif (13441<=i<13467):
        return "NSP12-" + str(int((i - 13441) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
    elif (13467<=i<16236):
        return "NSP12-" + str(int((i - 13467) / 3) + 10) + '-' + frame_shift + '-' + i_residue
            # RNA-dependent RNA polymerase
    elif (16236<=i<18039):
        return "NSP13-" + str(int((i - 16236) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
            # Helicase
    elif (18039<=i<19620):
        return "NSP14-" + str(int((i - 18039) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
            # 3' - 5' exonuclease
    elif (19620<=i<20658):
        return "NSP15-" + str(int((i - 19620) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
            # endoRNAse
    elif (20658<=i<21552):
        return "NSP16-" + str(int((i - 20658) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
            # 2'-O-ribose methyltransferase
    elif (21562<=i<25384):
        return "S-"     + str(int((i - 21562) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
    elif (28273<=i<29533):
        return "N-"     + str(int((i - 28273) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
    elif (29557<=i<29674):
        return "ORF10-" + str(int((i - 29557) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
    elif (26522<=i<27191):
        return "M-"     + str(int((i - 26522) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
    elif (27893<=i<28259):
        return "ORF8-"  + str(int((i - 27893) / 3) + 1)  + '-' + frame_shift + '-' + i_residue
    else:
        return "NC-"    + str(int(i))


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
    parser.add_argument('--mutation_on',    action='store_true', default=False,  help='whether or not to include mutational term in inference')
    parser.add_argument('--cutoff_on',      action='store_true', default=False,  help='whether or not to cutoff frequencies below freq_cutoff')
    
    arg_list = parser.parse_args(args)
    
    out_str       = arg_list.o
    q             = arg_list.q
    end_cutoff    = arg_list.end_cutoff
    start_cutoff  = arg_list.start_cutoff
    window        = arg_list.window
    timed         = arg_list.timed
    input_str     = arg_list.data
    c_directory   = arg_list.c_directory
    clip_end      = arg_list.clip_end
    clip_start    = arg_list.clip_start
    mask_site     = arg_list.mask_site
    final_time    = arg_list.final_t
    decay_rate    = arg_list.decay_rate
    delay         = arg_list.delay
    #data          = np.load(arg_list.data, allow_pickle=True)
    
    if not os.path.exists(out_str):
        os.mkdir(out_str)
        
    # creating directories for c++ files
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
    
    #if os.path.exists(os.path.join(out_str, location + '.npz')):
    #    sys.exit()
    
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
    

    
    
    
    
    
    
    

