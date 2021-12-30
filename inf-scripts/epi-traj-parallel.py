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
    parser.add_argument('--find_linked',    action='store_true', default=False,  help='whether or not to find the sites that are (almost) fully linked')
    parser.add_argument('--find_counts',    action='store_true', default=False,  help='whether to find the single and double site counts instead of the frequencies, mostly used for finding linked sites')
    
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
    data          = np.load(arg_list.data, allow_pickle=True)
    
    if not os.path.exists(out_str):
        os.mkdir(out_str)
        
    # creating directories for c++ files
    working_dir = os.getcwd()
    identifier  = out_str.split('/')
    if len(identifier) > 1:
        identifier = identifier[-1]
    else:
        identifier = identifier[0]
    
    # Load the inflowing sequence data if it is known
    if arg_list.inflow:
        inflow_data  = np.load(arg_list.inflow, allow_pickle=True) # sequences that flow migrate into the populations
        in_counts    = inflow_data['counts']        # the number of each sequence at each time
        in_sequences = inflow_data['sequences']     # the sequences at each time
        in_locs      = inflow_data['locations']     # the locations that the sequences flow into
        ### The below only works if a constant population size is used and a single region has inflowing population###
        pop_in       = [np.sum(in_counts[0][i] / np.mean(pop_size)) for i in range(len(in_counts[0]))]  # the total number of inflowing sequences at each time
        #print(pop_in)
        
    
    def trajectory_calc(nVec, sVec, mutant_sites_samp, d=q):
        """ Calculates the frequency trajectories"""
        Q = np.ones(len(nVec))
        for t in range(len(nVec)):
            if len(nVec[t]) > 0:
                Q[t] = np.sum(nVec[t])
        single_freq_s = np.zeros((len(nVec), len(mutant_sites_samp) * d))
        print(np.shape(single_freq_s))
        for t in range(len(nVec)):
            for i in range(len(mutant_sites_samp)):
                for j in range(len(sVec[t])):
                    single_freq_s[t, i * d + sVec[t][j][i]] += nVec[t][j] / Q[t]
        return single_freq_s
    
    
    def allele_counter(nVec, sVec, mutant_sites_samp, d=q):
        """ Calculates the counts for each allele at each time. """
    
        single = np.zeros((len(nVec), len(mutant_sites_samp) * d))
        for t in range(len(nVec)):
            for i in range(len(mutant_sites_samp)):
                for j in range(len(sVec[t])):
                    single[t, i * d + sVec[t][j][i]] += nVec[t][j]
        return single
                
    
    def allele_counter_in(sVec_in, nVec_in, mutant_sites, traj, pop_in, N, k, R, T, d=q):
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
        coefficient = (1 / ((1 / (N * k)) + ((k / R) / (N * k - 1)))) * (1 / R)
        if not isinstance(coefficient, float):
            coefficient = coefficient[:-1]
        term_2            = np.swapaxes(traj, 0, 1) * np.array(pop_in[:len(traj)]) / popsize
        integrated_inflow = np.sum((np.swapaxes(single_freq, 0, 1) - term_2) * coefficient,  axis=1)
        return integrated_inflow
    
    
    def transfer_in(seqs1, counts1, seqs2, counts2):
        """ Combine sequences transfered from another population with the current population"""
        
        for i in range(len(counts2)):
            unique = True
            for j in range(len(counts1)):
                if seqs1[j] == seqs2[i]:
                    unique = False
                    counts1[j] += counts2[i]
                    break
            if unique:
                seqs1.append(seqs2[i])
                counts1.append(counts2[i])
                
    def transfer_in_alt(seqs1, counts1, seqs2, counts2):
        """ Combine sequences transfered from another population with the current population"""
        
        for i in range(len(counts2)):
            unique = True
            for j in range(len(counts1)):
                if (seqs1[j] == seqs2[i]).all():
                    unique = False
                    counts1[j] += counts2[i]
                    break
            if unique:
                seqs1.append(seqs2[i])
                counts1.append(counts2[i])
                
                
    def add_previously_infected(nVec, sVec, popsize, decay_rate):
        """ Adds previously infected individuals to the population at later dates."""
        
        def combine(nVec1, sVec1, nVec2, sVec2):
            """ Combines sequence and count arrays at a specific time."""
            sVec_new = [list(i) for i in sVec1]
            nVec_new = [i for i in nVec1]
            for i in range(len(sVec2)):
                if list(sVec2[i]) in sVec_new:
                    nVec_new[sVec_new.index(list(sVec2[i]))] += nVec2[i]
                else:
                    nVec_new.append(nVec2[i])
                    sVec_new.append(list(sVec2[i]))
            return nVec_new, sVec_new
        
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
    
    
    def combine(nVec1, sVec1, nVec2, sVec2):
        """ Combines sequence and count arrays at a specific time."""
        sVec_new = [list(i) for i in sVec1]
        nVec_new = [i for i in nVec1]
        for i in range(len(sVec2)):
            if list(sVec2[i]) in sVec_new:
                nVec_new[sVec_new.index(list(sVec2[i]))] += nVec2[i]
            else:
                nVec_new.append(nVec2[i])
                sVec_new.append(list(sVec2[i]))
        return nVec_new, sVec_new

    
    def moving_average(freq, window=9):
        """ Calculates a moving average for a frequency array. """
        ret = np.cumsum(freq, axis=0)
        ret[window:] = ret[window:] - ret[:-window]
        result = ret[window - 1:] / window
        return result
    
    
    def freqs_from_counts(freq, window=9):
        """ Smooths the counts and then finds the total frequencies"""
        ret = np.cumsum(freq, axis=0)
        ret[window:] = ret[window:] - ret[:-window]
        result = ret[window - 1:]
        final  = []
        for i in result:
            final.append(i / np.sum(i))
        final  = np.array(final)
        return final
    
    
    def gaussian_window(nVec, sVec, window, t):
        """ Calculates the single and double site frequencies at a single time
        by using a Gaussian window to smooth the actual frequencies at nearby times. 
        NOT UPDATED TO WORK FOR MULTIPLE STATES AT EACH SITE"""
        single = np.zeros((len(sVec[0][0])))
        double = np.zeros((len(sVec[0][0], len(sVec[0][0]))))
        gaussian = np.exp((- 1 / 2) * (((np.arange(window) - (window / 2))/ (0.5 * window / 2) ) ** 2))
        norm = 0
        for j in range(window):
            norm += np.sum(nVec[t+j]) * gaussian[j]
        for i in range(len(sVec[0][0])):
            for j in range(window):
                single[i] += gaussian[j] * np.sum([nVec[t+j][k] * sVec[t+j][k][i] for k in range(len(sVec[t+j]))]) / norm
            for k in range(i+1, len(sVec[0][0])):
                for j in range(window):
                    double[i, k] += gaussian[j] * np.sum([nVec[t+j][l] * sVec[t+j][l][i] * sVec[t+j][l][k] for l in range(len(sVec[t+j]))]) / norm
                    double[k, i] += gaussian[j] * np.sum([nVec[t+j][l] * sVec[t+j][l][i] * sVec[t+j][l][k] for l in range(len(sVec[t+j]))]) / norm
        return single, double

    
    def get_codon_start_index(i, d=q):
        """ Given a sequence index i, determine the codon corresponding to it. """
        i = int(i/d)
        if   (13467<=i<=21554):
            return i - (i - 13467)%3
        elif (25392<=i<=26219):
            return i - (i - 25392)%3
        elif (26244<=i<=26471):
            return i - (i - 26244)%3
        elif (27201<=i<=27386):
            return i - (i - 27201)%3
        elif (27393<=i<=27886):
            return i - (i - 27393)%3
        elif (  265<=i<=13482):
            return i - (i - 265  )%3
        elif (21562<=i<=25383):
            return i - (i - 21562)%3
        elif (28273<=i<=29532):
            return i - (i - 28273)%3
        elif (29557<=i<=29673):
            return i - (i - 29557)%3
        elif (26522<=i<=27190):
            return i - (i - 26522)%3
        elif (27893<=i<=28258):
            return i - (i - 27893)%3
        else:
            return 0
        
        
    def smooth_nvec(nVec, sVec, window):
        """ Takes a moving average of nVec across the time series"""
        nVec_new = []
        sVec_new = []
        for t in range(len(nVec) - window):
            nVec_temp = [i for i in nVec[t]]
            sVec_temp = [i for i in sVec[t]]
            for i in range(1, window + 1):
                nVec_temp, sVec_temp = combine(nVec_temp, sVec_temp, nVec[t+i], sVec[t+i])
            nVec_new.append(nVec_temp)
            sVec_new.append(sVec_temp)
        return nVec_new, sVec_new
              
            
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
    
    # Load data
    print2(f'\tloading location {location}')
    data       = np.load(filepath, allow_pickle=True)  # Genome sequence data
    nVec       = data['nVec']                        # The number of each genome for each time
    sVec       = data['sVec']                        # The genomes for each time
    times_temp = data['times']                       # The times 
    labels     = data['mutant_sites']                # The genome locations of mutations
        
    # Cutoff some time points in the end and the beginning if told to
    if end_cutoff > 0:
        nVec       = nVec[:-end_cutoff]
        sVec       = sVec[:-end_cutoff]
        times_temp = times_temp[:-end_cutoff]
    if start_cutoff > 0:
        nVec       = nVec[start_cutoff:]
        sVec       = sVec[start_cutoff:]
        times_temp = times_temp[start_cutoff:]
    location_data  = location.split('-')
    timestamps     = location_data[-6:]
    for i in range(len(timestamps)):
        if len(timestamps[i]) == 1:
            timestamps[i] = '0' + timestamps[i]
    #print(timestamps)
        
    clip_length = len(labels) - len(labels[labels<clip_end])
    if clip_length > 0:
        labels      = labels[labels<clip_end]
        sVec_temp   = []
        for i in range(len(sVec)):
            sVec_t = []
            for j in range(len(sVec[i])):
                sVec_t.append(sVec[i][j][:-clip_length])
            sVec_temp.append(sVec_t)
        labels = labels[labels<=clip_end]
        sVec   = sVec_temp
        
    clip_length2 = len(labels) - len(labels[labels>clip_start])
    if clip_length2 > 0:
        labels      = labels[labels>clip_start]
        sVec_temp   = []
        for i in range(len(sVec)):
            sVec_t = []
            for j in range(len(sVec[i])):
                sVec_t.append(sVec[i][j][clip_length2:])
            sVec_temp.append(sVec_t)
        labels = labels[labels>clip_start]
        sVec   = sVec_temp
        
    if final_time:
        days_end   = len(times_temp[times_temp>final_time])
        if days_end > 0:
            times_temp = times_temp[:-days_end]
            nVec       = nVec[:-days_end]
            sVec       = sVec[:-days_end]
        elif days_end == len(times_temp):
            times_temp, nVec, sVec = [], [[]], [[[]]]
                
    dates      = times_temp[(window - 1):]
    dates_full = times_temp
    lengths    = len(sVec[0][0])
        
    # Preprocessing of the time-varying population size to be used to correct for the non-markovian property of the disease spread
    
    # Finds all sites at which there are mutations
    mutant_sites_full = [labels]
    allele_number     = labels
    
    L = len(allele_number)
    print2('number of sites:', L)
        
    # get data for the specific location
        
    region = location[:-22]
    if   region[-2:] == '--': region = region[:-2]
    elif region[-1]  =='-':   region = region[:-1]
    if arg_list.inflow:
        if region in list(in_locs):
            ss_incounts = in_counts[list(in_locs).index(region)]     # Contains numbers of each sequence that migrates to the location for each time
            ss_inseqs   = in_sequences[list(in_locs).index(region)]  # Contains containing the sequences migrating to this location for each time
            #print(ss_incounts, ss_inseqs)
    mutant_sites = labels    # The simulation specific site labels
                
    # Mask all sites in the list of sites given
    if arg_list.mask_group:
        ref_seq, ref_tag  = get_MSA(REF_TAG +'.fasta')
        ref_seq           = list(ref_seq[0])
        ref_poly          = np.array(ref_seq)[allele_number]
        mutants    = [get_label(i) for i in allele_number]
        mask_group = np.load(arg_list.mask_group, allow_pickle=True)
        mask_sites = [i[:-2] for i in mask_group]
        if np.any([i in mutants for i in mask_sites]):
            mask_nucs  = [NUC.index(i[-1]) for i in mask_group]
            mask_nums  = [allele_number[mutants.index(i)] for i in mask_sites]
            ref_mask   = [NUC.index(i) for i in ref_poly[mask_nums]]
            mask_idxs  = [mutants.index(i) for i in mask_sites]
            for i in range(len(mask_group)):
                if mask_nums[i] in list(mutant_sites):
                    for j in range(len(sVec)):
                        for k in range(len(sVec[i])):
                            if sVec[j][k][mask_idxs[i]] == mask_nucs[i]:
                                sVec[j][k][mask_idxs[i]] = ref_mask[i]
        
    if arg_list.find_counts:
        single_counts = allele_counter(nVec, sVec, labels, d=q)
    else:
        single_temp = trajectory_calc(nVec, sVec, mutant_sites, d=q)
        single_site = moving_average(single_temp, window)
    
    allele_new = []
    for i in range(len(allele_number)):
        for j in range(q):
            allele_new.append(str(allele_number[i]) + '-' + NUC[j])
    allele_number = allele_new
                        
    g = open(os.path.join(out_str, location + '.npz'), mode='wb')
    if arg_list.find_counts:
        np.savez_compressed(g, counts=single_counts, times=dates, allele_number=allele_number)
    else:
        np.savez_compressed(g, traj=single_site, times=dates, times_full=dates_full, traj_nosmooth=single_temp,
                            allele_number=allele_number)
    g.close()
        
    

    
    
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    
    
    

