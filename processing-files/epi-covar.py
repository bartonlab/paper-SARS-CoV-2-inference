# coding: utf-8

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
import gzip

REF_TAG = 'EPI_ISL_402125'
NUC     = ['-', 'A', 'C', 'G', 'T']

def find_site_index_file(filepath):
    """ Given a sequence file find the correct corresponding file that has the site names"""
    directory, file = os.path.split(filepath)
    return filepath[:-4] + '-sites.csv'


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


def allele_counter(df, d=5):
    """ Calculates the counts for each allele at each time. """
    #df = df.astype({'sequences' : str})
    sequences    = np.array(list(df['sequences']), dtype=int)
    dates        = np.array(list(df['times']))
    unique_dates = np.unique(dates)
    full_dates   = np.arange(np.amin(unique_dates), np.amax(unique_dates) + 1)
    single = np.zeros((len(full_dates), len(sequences[0]) * d))
    for t in range(len(full_dates)):
        if full_dates[t] not in unique_dates:
            continue
        seqs_t = sequences[dates==full_dates[t]]
        for i in range(len(sequences[0])):
            for j in range(len(seqs_t)):
                single[t, (i * d) + seqs_t[j][i]] += 1
    return single

"""
def allele_counter(df, d=5):
    # DECONVOLUTION CODE
    L = len(df['sequence'].iloc[0])
    single = np.zeros((len(np.unique(df['times'])), L * d))
    for df_iter, df_entry in df.iterrows():
        seq = np.array(list(df_entry['sequence']))
        for i in range(len(seq)):
            single[df_iter, (i * d) + seq[j]] += df_entry['counts']
    return single
"""


def trajectory_calc(nVec, sVec, mutant_sites_samp, d=5):
    """ Calculates the frequency trajectories"""
    Q = np.ones(len(nVec))
    for t in range(len(nVec)):
        if len(nVec[t]) > 0:
            Q[t] = np.sum(nVec[t])
    single_freq_s = allele_counter(nVec, sVec, mutant_sites_samp, d=q)
    single_freq_s = np.swapaxes(np.swapaxes(single_freq_s, 0, 1) / Q, 0, 1)                
    return single_freq_s

"""
def write_seq_file(seq_file, df):
    # DECONVOLUTION CODE
    f = open(seq_file, 'w')
    for df_iter, df_row in df.iterrows():
        seq    = [str(i) for i in df_row['sequence']]
        time   = df_row['times']
        counts = df_row['counts']
        f.write('%d\t%d\t%s\n' % (time, counts, ' '.join(seq)))
    f.close()
"""

def write_seq_file(seq_file, df):
    """ Write the sequences into a file that can be read by the C++ code to calculate the covariance."""
    f = open(seq_file, 'w')
    seqs         = np.array(list(df['sequences']))
    dates        = np.array(list(df['times']))
    unique_dates = np.unique(dates)
    L = len(seqs[0])
    for i in range(len(unique_dates)):
        seqs_t = seqs[dates==unique_dates[i]]
        for j in range(len(seqs_t)):
            temp_seq = [str(k) for k in seqs_t[j]]
            if len(temp_seq) != L:
                print(f'sequence length is {len(temp_seq)} when it should be {L}')
                continue
            f.write('%d\t%d\t%s\n' % (unique_dates[i], 1, ' '.join(temp_seq)))
    f.close()


def run_mpl(N, q, seq_file, covar_dir, location, ID=None, counts=False):
    """ Run C++ code that calculates the covariance"""
    
    covar_file  = f'covar-{location}.dat'
    num_file    = f'num-{location}.dat'
    out_file    = f'out-{location}.dat'
    double_file = f'double-{location}.dat'
    if os.path.exists(os.path.join(covar_dir, covar_file)):
        os.remove(os.path.join(covar_dir, covar_file))
    
    cpp_script = './bin/mpl'
    if ID:
        cpp_script = f'./bin/mpl-{ID}'
        print(ID)
    if counts:
        proc = subprocess.Popen([cpp_script, '-d', covar_dir, '-i', seq_file, '-o', out_file, '-g', str(0), '-N', f'{N}', '-sc', covar_file, '-sn', num_file, '-q', str(q), '-dc', double_file], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    else: 
        proc = subprocess.Popen([cpp_script, '-d', covar_dir, '-i', seq_file, '-o', out_file, '-g', str(0), '-N', f'{N}', '-sc', covar_file, '-sn', num_file, '-q', str(q)], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = proc.communicate()
    return_code    = proc.returncode
    proc.wait()
    if return_code != 0:
        print('directory\t', covar_dir)
        print('file\t', seq_file)
        print(os.listdir(covar_dir))
    return stdout, stderr, return_code


def read_covariance(covar_path, time_varying=False, counts=False, tv_dir=None, location=None, dates_covar=None, coefficient=None):
    """ Read in the covariance file. 
    tv_dir, location, and dates_covar are needed to save the covariance at each time"""
    covar_int     = []
    counter       = 0
    if time_varying:
        tv_covar_dir = tv_dir
        if not os.path.exists(tv_covar_dir):
            os.mkdir(tv_covar_dir)
        #for line in gzip.open(covar_path + '.gz', mode='rt'):
        #for line in open(covar_path, mode='r'):
        for line in gzip.open(covar_path, mode='rt'):
            if line!='\n':
                new_line = line.replace('\n', '')
                covar_int.append(np.array(new_line.split(' '), dtype=float))
            else:
                if counter >= len(dates_covar):
                    continue     
                f = open(os.path.join(tv_covar_dir, location + f'___{dates_covar[counter]}.npz'), mode='wb')
                covar = np.array(covar_int, dtype=float) * coefficient / 5
                np.savez_compressed(f, covar=covar)
                f.close()
                counter   += 1
                covar_int  = []
    else: 
        for line in open(os.path.join(covar_path)):
            if counts:
                covar_int.append(np.array(np.array(line.split(' ')[:-1])[::2], dtype=float))
            else:
                new_line     = line.split(' ')
                new_line[-1] = new_line[-1][:-1]
                #try:
                #    new2 = np.array(new_line, dtype=float)
                #except:
                #    print(new_line)
                covar_int.append(np.array(new_line, dtype=float))
                counter += 1
    return np.array(covar_int, dtype=float)


def allele_counter_in(sVec_in, nVec_in, mutant_sites, traj, pop_in, N, k, R, T, d=5):
    """ Counts the single-site frequencies of the inflowing sequences"""
    if isinstance(N, int) or isinstance(N, float):
        popsize = N * np.ones(T)
    else:
        popsize = N
    single_freq = np.zeros((T, d * len(sVec_in[0][0])))
    for t in range(T):
        for i in range(len(sVec_in[0][0])):
            for j in range(len(sVec_in[t])):
                single_freq[t, i * d + sVec_in[t][j][i]] = nVec_in[t][j] / popsize[t]
    coefficient = (N * k * R) * (1 / R) / (k + R)
    if not isinstance(coefficient, float):
        coefficient = coefficient[:-1]
    term_2            = np.swapaxes(traj, 0, 1) * np.array(pop_in[:len(traj)]) / popsize
    integrated_inflow = np.sum((np.swapaxes(single_freq, 0, 1) - term_2) * coefficient,  axis=1)
    return integrated_inflow


def add_previously_infected(nVec, sVec, popsize, decay_rate):
    """ Adds previously infected individuals to the population at later dates."""
    if type(popsize)==int or type(popsize)==float:
        popsize = np.ones(len(nVec)) * popsize
    new_nVec = [[i for i in nVec[0]]]
    new_sVec = [[i for i in sVec[0]]]
    for t in range(1, len(nVec)):
        nVec_t = list(np.array([i for i in nVec[t]]) * popsize[t] / np.sum(nVec[t]))
        sVec_t = [i for i in sVec[t]]
        for t_old in range(t):
            probability = np.exp(- decay_rate * (t - t_old)) * (1 - np.exp(- decay_rate))
            old_nVec    = list(probability * np.array(nVec[t_old]) * popsize[t_old] / np.sum(nVec[t_old]))
            old_sVec    = sVec[t_old]
            nVec_temp   = []
            sVec_temp   = []
            for j in range(len(old_nVec)):
                if int(round(old_nVec[j]))>0:
                    nVec_temp.append(old_nVec[j])
                    sVec_temp.append(old_sVec[j])
                    nVec_t, sVec_t = combine(nVec_t, sVec_t, nVec_temp, sVec_temp)
        new_nVec.append(nVec_t)
        new_sVec.append(sVec_t)
    return new_nVec, new_sVec


def freqs_from_counts(freq, num_seqs, window=5, min_seqs=200, hard_window=False):
    """ Smooths the counts and then finds the total frequencies"""
    # Determine window size, requiring min_seqs sequences in the beginning and end or a maximum of window days
    max_window  = window
    start_days  = 1
    end_days    = 1
    start_count = num_seqs[0]
    end_count   = num_seqs[-1]
    while start_count < min_seqs and start_days < window and start_days < len(freq) / 2:
        start_count += num_seqs[start_days]
        start_days  += 1
    while end_count < min_seqs and end_days < window and end_days < len(freq) / 2:
        end_count   += num_seqs[-1-end_days]
        end_days    += 1
    if start_days < max_window and end_days < max_window:
        window = max([start_days, end_days])    
    if window > len(freq / 2):
        window = int(len(freq / 2))
    if window > max_window:
        window = max_window
        
    # if hard_window is True, don't adjust the window size based on the number of sequences in a region
    if hard_window:
        window=max_window
    
    #print(np.shape(freq))
    ret = np.cumsum(freq, axis=0)
    #print(np.shape(ret))
    print('window', window)
    ret[window:] = ret[window:] - ret[:-window]
    result = ret[window - 1:]
        
    # Find total number of sequences in each window from num_seqs
    n_seqs = np.zeros(len(result))
    for i in range(len(result)):
        n_temp = 0
        for j in range(window):
            n_temp += num_seqs[i + j]
        n_seqs[i] += n_temp
            
    #print(len(n_seqs))
    #print(np.shape(result))
    final  = []
    for i in range(len(result)):
        if n_seqs[i]==0:
            final.append(np.zeros(len(result[i])))
        else:
            final.append(result[i] / np.array(n_seqs[i]))
    final  = np.array(final)
    return final


def main(args):
    """Calculate the covariance matrix and the trajectories for a single region from SARS-CoV-2 data"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',            type=str,    default='parallel',              help='output directory')
    parser.add_argument('-R',            type=float,  default=2,                       help='the basic reproduction number')
    parser.add_argument('-k',            type=float,  default=0.1,                     help='parameter determining shape of distribution of new infected')
    parser.add_argument('-q',            type=int,    default=5,                       help='number of mutant alleles per site')
    parser.add_argument('--data',        type=str,    default=None,                    help='.npz files containing the counts, sequences, times, and mutant_sites for different locations')
    parser.add_argument('--scratch',     type=str,    default='scratch',               help='scratch directory to write temporary files to')
    parser.add_argument('--record',      type=int,    default=1,                       help='number of generations between samples')
    parser.add_argument('--timed',       type=int,    default=0,                       help='if 0, wont print any time information, if 1 will print some information')
    parser.add_argument('--pop_size',    type=int,    default=10000,                   help='the population size')
    parser.add_argument('--ss_pop_size', type=str,    default=None,                    help='.npy file containing the simulation specific population size')
    parser.add_argument('--ss_k',        type=str,    default=None,                    help='.npy file containing the simulation specific k')
    parser.add_argument('--ss_R',        type=str,    default=None,                    help='.npy file containing the simulation specific R')
    parser.add_argument('--ss_record',   type=str,    default=None,                    help='.npy file containing the simulation specific record')
    parser.add_argument('--outflow',     type=str,    default=None,                    help='.npz file containing the counts and sequences that outflow')
    parser.add_argument('--inflow',      type=str,    default=None,                    help='.npz file containing the counts and sequences that inflow, and the locations')
    parser.add_argument('--window',      type=int,    default=15,                      help='the number of days over which to take the moving average')
    parser.add_argument('--delta_t',     type=int,    default=0,                       help='the amount of dates at the beginning and the end of the time series to use to calculate delta_x')
    parser.add_argument('--mask_site',   type=int,    default=None,                    help='the site to mask (change to WT) when inferring the other coefficients, in order to check effect of linkage (write)')
    parser.add_argument('--minSeqs',     type=int,    default=50,                      help='the minimum number of sequences to use at the end at the beginning to calculate delta_x.')
    parser.add_argument('--mask_group',  type=str,    default=None,                    help='the .npy file containing a list of sites (as nucleotide numbers) to mask for the inference')
    parser.add_argument('--remove_sites',type=str,    default=None,                    help='the sites to eliminate when inferring coefficients')
    parser.add_argument('--decay_rate',  type=float,  default=0,                       help='the exponential decay rate used to correct for infection lasting multiple generations')
    parser.add_argument('--nm_popsize',  type=str,    default=None,                    help='.csv file containing the population sizes used to correct for infection lasting multiple generations')
    parser.add_argument('--delay',       type=int,    default=None,                    help='the delay between the newly infected individuals and the individuals that')
    parser.add_argument('--ID',          type=str,    default=None,                    help='the slurm array task ID in case there are multiple C++ files, one for each parallel job')
    parser.add_argument('--tv_inference',   action='store_true', default=False,  help='whether or not inference is done at every time')
    parser.add_argument('--find_counts',    action='store_true', default=False,  help='whether or not to calculate the single and double site frequencies')
    
    arg_list = parser.parse_args(args)
    
    out_str       = arg_list.o
    q             = arg_list.q
    record        = arg_list.record
    window        = arg_list.window
    timed         = arg_list.timed
    input_str     = arg_list.data
    mask_site     = arg_list.mask_site
    decay_rate    = arg_list.decay_rate
    delay         = arg_list.delay
    min_seqs      = arg_list.minSeqs
    scratch_dir   = arg_list.scratch
    
    if arg_list.ss_record:
        record = np.load(arg_list.ss_record)     # The time differences between recorded sequences
    else:
        record = arg_list.record * np.ones(1)
    if arg_list.ss_pop_size:
        pop_size = np.load(arg_list.ss_pop_size) 
    else:
        pop_size = arg_list.pop_size 
    if arg_list.ss_R:
        R = np.load(arg_list.ss_R) # The basic reproductive number                   
    else:
        R = arg_list.R
    if arg_list.ss_k:
        k_ss = np.load(arg_list.ss_k) # The dispersion parameter
    else:
        k_ss = arg_list.k
    if arg_list.delta_t == 0 :
        delta_t = window
    else:
        delta_t = arg_list.delta_t
    if arg_list.remove_sites:
        remove_sites = np.load(arg_list.remove_sites)
    if arg_list.ID:
        task_ID = arg_list.ID
    else:
        task_ID = None
    
    # creating directories for c++ files
    if not os.path.exists(out_str):
        os.mkdir(out_str)
    if not os.path.exists(scratch_dir):
        os.mkdir(scratch_dir)
    identifier = os.path.split(out_str)[-1]
    covar_dir  = os.path.join(scratch_dir, f'{identifier}-sd-covar-dir')  # the directory that the covariance matrices, will be written to
    if not os.path.exists(covar_dir):
        os.mkdir(covar_dir)     
    print('scratch directory:   ', scratch_dir)
    print('covariance directory:', covar_dir)
    print(f'outfile {out_str}')
    
    # Load the inflowing sequence data if it is known
    if arg_list.inflow:
        inflow_data  = np.load(arg_list.inflow, allow_pickle=True) # sequences that flow migrate into the populations
        in_counts    = inflow_data['counts']        # the number of each sequence at each time
        in_sequences = inflow_data['sequences']     # the sequences at each time
        in_locs      = inflow_data['locations']     # the locations that the sequences flow into
        ### The below only works if a constant population size is used and a single region has inflowing population###
        pop_in       = [np.sum(in_counts[0][i] / np.mean(pop_size)) for i in range(len(in_counts[0]))]  # the total number of inflowing sequences at each time
        #print(pop_in)
            
    if timed > 0:
        t_start = timer()
        print("starting")
    
    # Make status file that records the progress of the script
    filepath    = input_str
    location    = os.path.split(filepath)[-1][:-4]
    status_name = f'covar-test-{location}.csv'
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
    
    # Load data
    data = get_data(filepath)
    times_temp   = data['dates']
    ref_sites    = np.array(data['ref_sites'])
    sub_dates    = data['submission_dates']
    sequences    = [list(i) for i in data['sequences']]
    unique_dates = np.unique(sub_dates)
    new_data = {
        'times' : times_temp,
        'sub_dates' : sub_dates,
        'sequences' : sequences
    }
    df = pd.DataFrame(data=new_data)
    print2(ref_sites)
    
    location_data = location.split('-')
    timestamps    = location_data[-6:]
    for i in range(len(timestamps)):
        if len(timestamps[i]) == 1:
            timestamps[i] = '0' + timestamps[i]
                
    #dates       = times_temp[(window - 1):]
    dates_full  = np.arange(np.amin(times_temp), np.amax(times_temp) + 1)
    nm_popsize  = pop_size
    L           = len(ref_sites)
    ss_record   = record
    coefficient = pop_size * k_ss * R / (R + k_ss)
    print2('number of sites:', L)
        
    # get data for the specific location
    region = location[:-22]
    if   region[-2:] == '--': region = region[:-2]
    elif region[-1]  ==  '-': region = region[:-1]
        
    # Load the sequences due to travel
    if arg_list.inflow:
        if region in list(in_locs):
            ss_incounts = in_counts[list(in_locs).index(region)]     # Contains numbers of each sequence that migrates to the location for each time
            ss_inseqs   = in_sequences[list(in_locs).index(region)]  # Contains containing the sequences migrating to this location for each time
                
    # Mask all sites in the list of sites given
    if arg_list.mask_group:
        ### NEED TO UPDATE TO DEAL WITH MULTIPLE ALLELES AT EACH SITE
        ref_file   = pd.read_csv('ref-index-short.csv')
        index      = list(ref_file['ref_index'])
        ref_seq    = list(ref_file['nucleotide'])
        ref_poly   = []
        for i in range(len(index)):
            if index[i] in ref_sites:
                ref_poly.append(ref_seq[i])
                
        mutants    = [get_label_new(i) for i in ref_sites]
        mask_group = np.load(arg_list.mask_group, allow_pickle=True)
        mask_sites = [i[:-2] for i in mask_group]
        if np.any([i in mutants for i in mask_sites]):
            mask_nucs  = [NUC.index(i[-1]) for i in mask_group]
            mask_nums  = [ref_sites[mutants.index(i)] for i in mask_sites]
            ref_mask   = [NUC.index(i) for i in ref_poly[mask_nums]]
            mask_idxs  = [mutants.index(i) for i in mask_sites]
            for i in range(len(mask_group)):
                if mask_nums[i] in list(ref_sites):
                    for j in range(len(sVec)):
                        for k in range(len(sVec[i])):
                            if sVec[j][k][mask_idxs[i]] == mask_nucs[i]:
                                sVec[j][k][mask_idxs[i]] = ref_mask[i]
    # mask single site if given                            
    if arg_list.mask_site:
        # mask out 'mask_site' if there is one or multiple
        if mask_site in list(ref_sites):
            site_loc  = list(ref_sites).index(mask_site)
            for i in range(len(sVec)):
                for j in range(len(sVec[i])):
                    sVec[i][j][site_loc] = 0
                        
    # process the sequences that flow to this location from elsewhere, if they are known.
    ### DOES THIS NEED TO BE FIXED FOR MULTIPLE STATES AT EACH SITE?
    if arg_list.inflow and not mask_site:
        if region in list(in_locs):
            traj_temp = trajectory_calc(nVec, sVec, ref_sites, d=q)
            print2('number of time points in the region  \t', len(nVec))
            print2('number of time points inflowing      \t', len(ss_incounts))
            print2('number of time points in trajectories\t', len(traj_temp))
            inflow_term = allele_counter_in(ss_inseqs, ss_incounts, ref_sites, traj_temp, pop_in, pop_size, k_ss, R, len(nVec)) # Calculates the correction due to migration
            print2(f'region {region} present')
        else:
            inflow_term = np.zeros(len(ref_sites) * q)
    else:
        inflow_term = np.zeros(len(ref_sites) * q)
    
    ### Run inference trimming the data by submission date
    if timed > 0:
        bootstrap_start_time = timer()
            
    df = df.sort_values(by='times')
        
    # Write nVec and sVec to file
    seq_file = f'seqs-{location}.dat'
    seq_path = os.path.join(covar_dir, seq_file)
    write_seq_file(seq_path, df)
    
    if arg_list.find_counts:
        stdout, stderr, exit_code = run_mpl(pop_size, q, seq_file, covar_dir, location, counts=True)
    else:
        stdout, stderr, exit_code = run_mpl(pop_size, q, seq_file, covar_dir, location, counts=False)
    if exit_code != 0:
        print2('exit code:', exit_code)
        print2('stdout', stdout)
        print2('stderr', stderr)
    
    covar_file = f'covar-{location}.dat'
    if arg_list.tv_inference:
        cov_path_old = os.path.join(covar_dir, covar_file)
        proc         = subprocess.Popen(['gzip', cov_path_old])
        proc.wait()
        if os.path.exists(cov_path_old):
            os.remove(cov_path_old)
        
    # read in covariance file  
    if arg_list.tv_inference:
        tv_covar_dir     = os.path.join(out_str, 'tv_covar')
        t_unique, counts = np.unique(times_temp, return_counts=True)
        total_seqs    = []
        #dates_full    = np.arange(np.amin(t_unique), np.amax(t_unique) + 1)
        for i in range(len(dates_full)):
            if dates_full[i] in t_unique:
                total_seqs.append(counts[list(t_unique).index(dates_full[i])])
            else: 
                total_seqs.append(0)
        covar_path = os.path.join(covar_dir, covar_file + '.gz')
        dates_covar   = [dates_full[i] for i in range(1, len(dates_full)) if total_seqs[i-1]!=0]
        dates_nocovar = [dates_full[i] for i in range(1, len(dates_full)) if (dates_full[i] not in dates_covar and dates_full[i]>dates_covar[0])]
        covar_int     = read_covariance(covar_path, time_varying=True, tv_dir=tv_covar_dir, 
                                        location=location, dates_covar=dates_covar, coefficient=coefficient)
        print2(f'dates without a covariance matrix: {dates_nocovar}')
        for t in dates_nocovar:
            old_file = os.path.join(tv_covar_dir, location + f'___{t-1}.npz')
            new_file = os.path.join(tv_covar_dir, location + f'___{t}.npz')
            shutil.copy(old_file, new_file)
        covar_int = []
    elif arg_list.find_counts:
        double_file = f'double-{location}.dat'
        covar_path  = os.path.join(covar_dir, double_file)
        covar_int   = read_covariance(covar_path, counts=True)
    else:
        covar_path = os.path.join(covar_dir, covar_file)
        covar_int  = read_covariance(covar_path) * coefficient / 5
    #covar_file = f'covar-{location}.dat'
    #covar_path = os.path.join(covar_dir, covar_file)
    #covar_int  = read_covariance(covar_path) * coefficient / 5  
    
    # delete files
    for file in [seq_file, f'covar-{location}.dat', f'num-{location}.dat', f'out-{location}.dat', f'covar-{location}.dat.gz']:
        if os.path.exists(os.path.join(covar_dir, file)):
            os.remove(os.path.join(covar_dir, file))
    print2(ref_sites)
    
    # calculate the frequencies
    counts = allele_counter(df, d=q)
    u_dates, num_seqs = np.unique(list(df['times']), return_counts=True)
    total_seqs    = []
    for i in range(len(dates_full)):
        if dates_full[i] in u_dates:
            total_seqs.append(num_seqs[list(u_dates).index(dates_full[i])])
        else: 
            total_seqs.append(0)
    if arg_list.tv_inference and 'united kingdom' not in location:
        counts = freqs_from_counts(counts, total_seqs, window=delta_t, min_seqs=min_seqs, hard_window=True)
    elif not arg_list.find_counts:
        counts = freqs_from_counts(counts, total_seqs, window=delta_t, min_seqs=min_seqs)
    else:
        counts = np.sum(counts, axis=0)
        
    # save data
    file = os.path.join(out_str, location + '.npz')
    g = open(file, mode='wb')
    if not arg_list.find_counts:
        np.savez_compressed(
            g, 
            location=location, 
            times=dates_full[-len(counts):],
            times_full=dates_full,
            ref_sites=ref_sites, 
            allele_number=ref_sites,  
            k=k_ss, N=pop_size, R=R,
            covar=covar_int, 
            counts=counts, 
            inflow=inflow_term
        )
    else:
        np.savez_compressed(
            g, 
            location=location, 
            ref_sites=ref_sites, 
            allele_number=ref_sites,  
            double_counts=covar_int, 
            single_counts=counts, 
        )
    g.close()                    
        
    if timed > 0:
        new_run_time = timer()
        print2(f'{new_run_time - bootstrap_start_time} seconds to run the covariance')
        bootstrap_start_time = new_run_time

        


if __name__ == '__main__': 
    main(sys.argv[1:])

