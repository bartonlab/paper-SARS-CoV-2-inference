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
import gzip

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
    
    
def max_freq_counter(nVec, sVec, mutant_sites_samp, q=5):
        """ Calculates the counts for each allele. """
        norm = np.array([np.sum(nVec[t]) for t in range(len(nVec))])
        norm[norm==0]==1
        if q == 1:
            single = np.zeros((len(mutant_sites_samp)))
            for i in range(len(mutant_sites_samp)):
                single[i] += np.amax([np.sum([nVec[t][j] * sVec[t][j][i] for j in range(len(sVec[t]))]) / norm[t] for t in range(len(nVec))]) 
        else:
            single = np.zeros(len(mutant_sites_samp) * q)
            for t in range(len(nVec)):
                single_temp = np.zeros(len(mutant_sites_samp) * q)
                for j in range(len(nVec[t])):
                    for i in range(len(mutant_sites_samp)):
                        single_temp[i * q + sVec[t][j][i]] += nVec[t][j] / norm[t]
                single = np.maximum(single, single_temp)
        return single


def main(args):
    """Calculate the covariance matrix and the trajectories for a single region from SARS-CoV-2 data"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',            type=str,    default='parallel',              help='output directory')
    parser.add_argument('-R',            type=float,  default=2,                       help='the basic reproduction number')
    parser.add_argument('-k',            type=float,  default=0.1,                     help='parameter determining shape of distribution of new infected')
    parser.add_argument('-q',            type=int,    default=5,                       help='number of mutant alleles per site')
    parser.add_argument('--mu',          type=float,  default=0.,                      help='the mutation rate')
    parser.add_argument('--data',        type=str,    default=None,                    help='.npz files containing the counts, sequences, times, and mutant_sites for different locations')
    parser.add_argument('--g1',          type=float,  default=1,                       help='regularization restricting the magnitude of the selection coefficients')
    parser.add_argument('--scratch',     type=str,    default='scratch',               help='scratch directory to write temporary files to')
    parser.add_argument('--freq_cutoff', type=float,  default=0,                       help='if a mutant frequency never rises above this number, it will not be used for inference')
    parser.add_argument('--record',      type=int,    default=1,                       help='number of generations between samples')
    parser.add_argument('--timed',       type=int,    default=0,                       help='if 0, wont print any time information, if 1 will print some information')
    parser.add_argument('--pop_size',    type=int,    default=10000,                   help='the population size')
    parser.add_argument('--ss_pop_size', type=str,    default=None,                    help='.npy file containing the simulation specific population size')
    parser.add_argument('--ss_k',        type=str,    default=None,                    help='.npy file containing the simulation specific k')
    parser.add_argument('--ss_R',        type=str,    default=None,                    help='.npy file containing the simulation specific R')
    parser.add_argument('--ss_record',   type=str,    default=None,                    help='.npy file containing the simulation specific record')
    parser.add_argument('--outflow',     type=str,    default=None,                    help='.npz file containing the counts and sequences that outflow')
    parser.add_argument('--inflow',      type=str,    default=None,                    help='.npz file containing the counts and sequences that inflow, and the locations')
    parser.add_argument('--end_cutoff',  type=int,    default=0,                       help='the number of days at the end of the simulations to ignore due to bad sampling')
    parser.add_argument('--start_cutoff',type=int,    default=0,                       help='the number of days at the end of the simulations to ignore due to bad sampling')
    parser.add_argument('--window',      type=int,    default=15,                      help='the number of days over which to take the moving average')
    parser.add_argument('--clip_end',    type=int,    default=29700,                   help='the last site to clip the genomes to')
    parser.add_argument('--clip_start',  type=int,    default=150,                     help='the first site to clip the genomes to')
    parser.add_argument('--delta_t',     type=int,    default=0,                       help='the amount of dates at the beginning and the end of the time series to use to calculate delta_x')
    parser.add_argument('--mut_type',    type=str,    default=None,                    help='.npz file containing the location along the genome of a mutation under kw locations, and the type of mutation under kw types')
    parser.add_argument('--c_directory', type=str,    default='Archive',               help='directory containing the c++ scripts')
    parser.add_argument('--mask_site',   type=int,    default=None,                    help='the site to mask (change to WT) when inferring the other coefficients, in order to check effect of linkage (write)')
    parser.add_argument('--mask_group',  type=str,    default=None,                    help='the .npy file containing a list of sites (as nucleotide numbers) to mask for the inference')
    parser.add_argument('--remove_sites',type=str,    default=None,                    help='the sites to eliminate when inferring coefficients')
    parser.add_argument('--final_t',     type=int,    default=None,                    help='the last time point to be considered, cutoff all time series after this time (in days after 01-01-2020)')
    parser.add_argument('--decay_rate',  type=float,  default=0,                       help='the exponential decay rate used to correct for infection lasting multiple generations')
    parser.add_argument('--nm_popsize',  type=str,    default=None,                    help='.csv file containing the population sizes used to correct for infection lasting multiple generations')
    parser.add_argument('--delay',       type=int,    default=None,                    help='the delay between the newly infected individuals and the individuals that ')
    parser.add_argument('--mutation_on',    action='store_true', default=False,  help='whether or not to include mutational term in inference')
    parser.add_argument('--cutoff_on',      action='store_true', default=False,  help='whether or not to cutoff frequencies below freq_cutoff')
    parser.add_argument('--find_linked',    action='store_true', default=False,  help='whether or not to find the sites that are (almost) fully linked')
    parser.add_argument('--find_counts',    action='store_true', default=False,  help='whether to find the single and double site counts instead of the frequencies, mostly used for finding linked sites')
    parser.add_argument('--tv_inference',   action='store_true', default=False,  help='if true, then covariance matrix at every time instead of the integrated covariance matrix is returned')
    parser.add_argument('--trajectories',   action='store_true', default=False,  help='whether or not to save the trajectories')
    parser.add_argument('--numerator_only', action='store_true', default=False,  help='if true, dont calculate the covariances')
    parser.add_argument('--covar_only',     action='store_true', default=False,  help='if true, dont calculate the numerator')
    parser.add_argument('--readCovar',      action='store_true', default=False,  help='use this option if the covariance at every time has already been constructed')
    
    arg_list = parser.parse_args(args)
    
    out_str       = arg_list.o
    g1            = arg_list.g1
    mu            = arg_list.mu
    q             = arg_list.q
    freq_cutoff   = arg_list.freq_cutoff
    end_cutoff    = arg_list.end_cutoff
    start_cutoff  = arg_list.start_cutoff
    record        = arg_list.record
    mut_on        = arg_list.mutation_on
    window        = arg_list.window
    timed         = arg_list.timed
    find_linked   = arg_list.find_linked
    input_str     = arg_list.data
    c_directory   = arg_list.c_directory
    clip_end      = arg_list.clip_end
    clip_start    = arg_list.clip_start
    mask_site     = arg_list.mask_site
    final_time    = arg_list.final_t
    decay_rate    = arg_list.decay_rate
    delay         = arg_list.delay
    data          = np.load(arg_list.data, allow_pickle=True)
    scratch_dir   = arg_list.scratch
    
    working_dir   = os.getcwd()
    if arg_list.find_counts:
        c_directory = 'Archive-alt2'  
    elif arg_list.tv_inference:
        c_directory = 'Archive-tv-int'
    else:
        c_directory = arg_list.c_directory
    if not os.path.exists(c_directory):
        c_directory = os.path.join(working_dir, c_directory)
    current_dir = os.path.split(working_dir)[1]
    if current_dir[:current_dir.find('-')]!='Archive':
        print('current directory is:', current_dir)
    if not os.path.exists(out_str):
        os.mkdir(out_str)
        
    # creating directories for c++ files
    working_dir = os.getcwd()
    identifier  = out_str.split('/')
    if len(identifier) > 1:
        identifier = identifier[-1]
    else:
        identifier = identifier[0]
        
    if not os.path.exists(scratch_dir):
        os.mkdir(scratch_dir)
    covar_dir = os.path.join(scratch_dir, f'{identifier}-covar-dir')  # the directory that the covariance matrices, will be written to
    #if not os.path.exists(c_directory):
    #    print(f'directory {c_directory} doesnt exist')
    #    print(f'current working directory is {os.getcwd()}')
    #else:
    #    os.chdir(c_directory)
    if not os.path.exists(covar_dir):
        os.mkdir(covar_dir)  
    print('scratch directory:   ', scratch_dir)
    print('covariance directory:', covar_dir)
    
    # Create a dictionary telling you which sites are synonymous and which are not. CAN ELIMINATE THIS
    if arg_list.mut_type:
        mutation_type_data = np.load(arg_list.mut_type, allow_pickle=True)
        locs = mutation_type_data['locations']       # The indicies for the polymorphic sites
        types = mutation_type_data['types']          # S or NS depending on if the mutation is synonymous or not.
        mutation_type = {}                           # Map from indices to S or NS
        for i in range(len(locs)):
            mutation_type[locs[i]] = types[i]
    
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
        k = np.load(arg_list.ss_k) # The dispersion parameter
    else:
        k = arg_list.k
    if arg_list.delta_t == 0 :
        delta_t = window
    else:
        delta_t = arg_list.delta_t
    
    # Load the inflowing sequence data if it is known
    if arg_list.inflow:
        inflow_data  = np.load(arg_list.inflow, allow_pickle=True) # sequences that flow migrate into the populations
        in_counts    = inflow_data['counts']        # the number of each sequence at each time
        in_sequences = inflow_data['sequences']     # the sequences at each time
        in_locs      = inflow_data['locations']     # the locations that the sequences flow into
        ### The below only works if a constant population size is used and a single region has inflowing population###
        pop_in       = [np.sum(in_counts[0][i] / np.mean(pop_size)) for i in range(len(in_counts[0]))]  # the total number of inflowing sequences at each time
        #print(pop_in)
        
    if arg_list.remove_sites:
        remove_sites = np.load(arg_list.remove_sites)
    
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
    
    
    def freqs_from_counts(freq, num_seqs, window=9):
        """ Smooths the counts and then finds the total frequencies"""
        ret = np.cumsum(freq, axis=0)
        ret[window:] = ret[window:] - ret[:-window]
        result = ret[window - 1:]
        
        # Find total number of sequences in each window from num_seqs
        n_seqs = np.zeros(len(result))
        for i in range(len(result)):
            n_temp = 0
            for j in range(window):
                n_temp += num_seqs[i + j]
            n_seqs[i] += n_temp
            
        print(len(n_seqs))
        print(np.shape(result))
        final  = []
        for i in range(len(result)):
            final.append(result[i] / np.array(n_seqs[i]))
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
    
    
    def calculate_deltax(nVec, sVec, k_ss, R_ss, N_ss, window, d=q):
        """ Calculates the change in frequency by finding the average frequency of each allele in the first and last time-points.
        A number of time-points equal to the value of the window variable is used. """
        beginning = np.zeros(len(sVec[0][0]) * d)
        end       = np.zeros(len(sVec[0][0]) * d)
        nVec_start, sVec_start, nVec_end, sVec_end = [], [], [], []
        for t in range(window):
            transfer_in_alt(sVec_start, nVec_start, list(sVec[t]), list(nVec[t]))
            transfer_in_alt(sVec_end, nVec_end, list(sVec[-1-t]), list(nVec[-1-t]))
        coeff1 = (1 / ((1 / (N_ss * k_ss)) + ((k_ss / R_ss) / (N_ss * k_ss - 1))))
        for i in range(len(sVec[0][0])):
            for j in range(len(sVec_start)):
                beginning[i * d + sVec_start[j][i]] += coeff1 * nVec_start[j] / np.sum(nVec_start)
            for j in range(len(sVec_end)):
                end[i * d + sVec_end[j][i]]         += coeff1 * nVec_end[j] / np.sum(nVec_end)
        return (end - beginning)
    
    
    def calculate_deltax_tv(nVec, sVec, k_ss, R_ss, N_ss, mut_sites, window):
        """ Integrates the change in frequency over the time series allowing parameters to be time-varying. """
        coeff1 = (1 / ((1 / (N_ss * k_ss)) + ((k_ss / R_ss) / (N_ss * k_ss - 1))))
        single_site = trajectory_calc(nVec, sVec, mut_sites)
        #delta_x     = np.zeros(single_site[0])
        #for i in range(1, len(nVec)):
            #delta_x += (single_site[i] - single_site[i - 1]) * coeff1[i - 1]
        delta_x = np.sum(np.diff(single_site, axis=0) * coeff1[:len(single_site)], axis=0)
        return delta_x
    
    
    def integrate_deltax(nVec, sVec, k_ss, R_ss, N_ss, nm_pop, mut_sites):
        """ Integrates the delta x term if the process is considered to be nonmarkovian."""
        single_new  = trajectory_calc(nVec, sVec, mut_sites)   #Calculates the frequency of new infections.
        single_freq = np.zeros(np.shape(single_new))    # the frequency of still infectious.
        for t in range(len(single_new)):
            if isinstance(nm_pop, list): single_freq[t] += single_new[t] * nm_pop[t] 
            else:                        single_freq[t] += single_new[t] * nm_pop 
            for t_old in range(t):
                probability     = np.exp(- decay_rate * (t - t_old))
                if isinstance(nm_pop, list): single_freq[t] += probability * np.array(single_new[t_old]) * nm_pop[t_old]
                else:                        single_freq[t] += probability * np.array(single_new[t_old]) * nm_pop 
            single_freq[t] = single_freq[t] / np.sum(single_freq[t])
        delta_x = np.zeros(np.shape(single_new))
        coeff1  = (1 / ((1 / (N_ss * k_ss)) + ((k_ss / R_ss) / (N_ss * k_ss - 1))))
        for i in range(1, len(nVec)):
            delta_x[:, i] = (single_new[:, i] - single_freq[:, i-1])
        delta_int = np.sum(np.swapaxes(delta_x, 0, 1) * coeff1, axis=1)
        return delta_int
    
    
    def integrate_deltax_old(nVec, sVec, k_ss, R_ss, N_ss, nm_pop, mut_sites):
        """ Integrates the delta x term if the process is considered to be nonmarkovian."""
        single_new         = np.swapaxes(trajectory_calc(nVec, sVec, mut_sites), 0, 1)   #Calculates the frequency of new infections.
        nVec_new, sVec_new = add_previously_infected(nVec, sVec, nm_pop, decay_rate)
        single_freq        = np.swapaxes(trajectory_calc(nVec_new, sVec_new, mut_sites), 0, 1)
        delta_x            = np.zeros(np.shape(single_new))
        coeff1             = (1 / ((1 / (N_ss * k_ss)) + ((k_ss / R_ss) / (N_ss * k_ss - 1))))
        for i in range(1, len(nVec)):
            delta_x[i] = single_new[i] - single_freq[i-1]
        delta_int = np.sum(delta_x * coeff1, axis=1)
        return delta_int

    
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
    
    # Loading and organizing the data from the different locations. Combining it into new arrays.
    #allele_number, mutant_sites_full, dates, locations, lengths, dates_full = [], [], [], [], [], []
    #nVec_full, sVec_full = [], []
    #filepaths = []
        
    # Load data
    filepath = input_str
    location = os.path.split(filepath)[-1][:-4]
    
    # Make status file that records the progress of the script
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
    print2(f'loading location {location}')
    data       = np.load(filepath, allow_pickle=True)  # Genome sequence data
    nVec       = data['nVec']                        # The number of each genome for each time
    sVec       = data['sVec']                        # The genomes for each time
    times_temp = data['times']                       # The times 
    labels     = data['mutant_sites']                # The genome locations of mutations
    if 'ref_sites' in data:
        ref_sites = data['ref_sites']
    #print(labels)
    print(times_temp)
    print('lengths of times and nVec', len(times_temp), len(nVec))
        
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
        sVec_temp   = []
        for i in range(len(sVec)):
            sVec_t = []
            for j in range(len(sVec[i])):
                sVec_t.append(sVec[i][j][labels<clip_end])
            sVec_temp.append(sVec_t)
        ref_sites = ref_sites[labels<clip_end]
        labels    = labels[labels<clip_end]
        sVec      = sVec_temp
        
    clip_length2 = len(labels) - len(labels[labels>clip_start])
    if clip_length2 > 0:
        sVec_temp   = []
        for i in range(len(sVec)):
            sVec_t = []
            for j in range(len(sVec[i])):
                sVec_t.append(sVec[i][j][labels>clip_start])
            sVec_temp.append(sVec_t)
        ref_sites = ref_sites[labels>clip_start]
        labels    = labels[labels>clip_start]
        sVec      = sVec_temp
        
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
    
    if arg_list.nm_popsize:
        ### FUTURE: Deal with subregional information, or when the entries of regions are a list of location ###
        # finding the different regions in the sequence data
        regional_data = [i.split('-') for i in location]
        countries     = [i[1] for i in regional_data]
        regions       = [i[2] for i in regional_data]
        main_locs     = []
        for i in range(len(countries)):
            if countries[i]=='usa': main_locs.append(regions[i])
            else: main_locs.append(countries[i])
        
        # getting the estimated population sizes for the regions in the sequence data
        nm_popdata    = pd.read_csv(arg_list.nm_popsize)
        dates_file    = list(nm_popdata.date)
        locs_file     = [i.lower() for i in list(nm_popdata.location_name)]
        popsize_file  = list(nm_popdata.est_infections_mean)
        for l in locs_file:
            if l=='czechia':
                locs_file[locs_file.index(l)] = 'czech republic'
        nm_popsize    = []
        nm_times      = []
        for i in range(len(regional_data)):
            if main_locs[i] in locs_file:
                dates_temp = np.array(dates_file)[np.array(locs_file)==main_locs[i]]   
                pop_temp   = np.array(popsize_file)[np.array(locs_file)==main_locs[i]]
                dates_temp = dates_temp[~np.isnan(pop_temp)]
                times_temp = [(dt.date.fromisoformat(dates_temp[i]) - dt.date(2020,1,1)).days for i in range(len(dates_temp))]
                pop_temp   = pop_temp[~np.isnan(pop_temp)]
                
                # making sure the time series in the sequence data and the estimated number of infected are the same length
                if times_temp[0] > dates_full[i][0]:
                    pop_temp   = [pop_temp[0] for i in range(dates_full[i][0] - times_temp[0])] + list(pop_temp)
                    times_temp = list(np.arange(dates_full[i][0], times_temp[0])) + list(times_temp)
                    print2(f'location {main_locs[i]} has earlier information for sequencing than for estimated infected population size')
                    print2(f'earliest time in sequence data:   \t {dates_full[i][0]}')
                    print2(f'earliest time in population data: \t {times_temp[0]}')
                if times_temp[0] < dates_full[i][0]:
                    pop_temp   = pop_temp[list(times_temp).index(dates_full[i][0]):]
                    times_temp = times_temp[list(times_temp).index(dates_full[i][0]):]
                if times_temp[-1] < dates_full[i][-1]:
                    pop_temp   = list(pop_temp) + [pop_temp[-1] for i in range(dates_full[i][-1] - times_temp[-1])]
                    times_temp = list(times_temp) + list(np.arange(times_temp[-1]+1, dates_full[i][-1]+1))
                    print2(f'location {main_locs[i]} has later information for sequencing than for estimated infected population size')
                    print2(f'latest time in sequence data:   \t {dates_full[i][-1]}')
                    print2(f'latest time in population data: \t {times_temp[-1]}')
                if times_temp[-1] > dates_full[i][-1]:
                    pop_temp   = pop_temp[:list(times_temp).index(dates_full[i][-1])+1]
                    times_temp = times_temp[:list(times_temp).index(dates_full[i][-1])+1]
                nm_popsize.append(pop_temp)
                nm_times.append(times_temp)
            else:
                print2(f'{main_locs[i]} not in the estimated population size file')
    else:
        if len(np.shape(pop_size))==0:
            nm_popsize = pop_size * np.ones(1)
        elif len(np.shape(pop_size))==1:
            nm_popsize = pop_size * np.ones((1, len(pop_size)))
        else:
            nm_popsize = pop_size
    
    # Finds all sites at which there are mutations
    mutant_sites_full = [labels]
    allele_number     = labels
    print([len(i) for i in nVec])
    
    # eliminate sites in remove_sites
    #if arg_list.remove_sites:
    #    allele_number = [i for i in allele_number if i not in list(remove_sites)]
    #    mutant_sites_tot, sVec_full = filter_sites_alternative(sVec_full, nVec_full, mutant_sites_tot, allele_number)
    
    L = len(allele_number)
    print2('number of sites:', L)
        
    # get data for the specific location
    k_ss        = k
    R_ss        = R
    ss_record   = record
    ss_pop_size = pop_size
        
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
        
    if arg_list.trajectories:
        single_temp = trajectory_calc(nVec, sVec, mutant_sites, d=q)
        single_site = moving_average(single_temp, window)
        T = int(ss_record * len(nVec) - 1) 
    if arg_list.mask_site:
        # mask out 'mask_site' if there is one or multiple
        if mask_site in list(mutant_sites):
            site_loc  = list(mutant_sites).index(mask_site)
            for i in range(len(sVec)):
                for j in range(len(sVec[i])):
                    sVec[i][j][site_loc] = 0
                        
    # process the sequences that flow to this location from elsewhere, if they are known.
    ### DOES THIS NEED TO BE FIXED FOR MULTIPLE STATES AT EACH SITE?
    if arg_list.inflow and not mask_site:
        if region in list(in_locs):
            traj_temp = trajectory_calc(nVec, sVec, mutant_sites, d=q)
            print2('number of time points in the region  \t', len(nVec))
            print2('number of time points inflowing      \t', len(ss_incounts))
            print2('number of time points in trajectories\t', len(traj_temp))
            inflow_term = allele_counter_in(ss_inseqs, ss_incounts, mutant_sites, traj_temp, pop_in, ss_pop_size, k_ss, R_ss, len(nVec)) # Calculates the correction due to migration
            print2(f'region {region} present')
            #print(inflow_term)
        else:
            inflow_term = np.zeros(len(mutant_sites) * q)
    else:
        inflow_term = np.zeros(len(mutant_sites) * q)
            
    ### Find the covariance 
    # Write nVec and sVec to file
    seq_file =  f'seqs-{location}.dat'
    infectious_freq_temp = []
    newly_infected_temp  = []
    print(covar_dir)
    f = open(os.path.join(covar_dir, seq_file), mode='w')
    if decay_rate==0:
        single_infectious = []
        single_newly_inf  = []
        deltax_int        = []
        for i in range(len(nVec)):
            for j in range(len(nVec[i])):
                if not isinstance(nVec[0][0], (float, np.floating)):
                    f.write('%d\t%d\t%s\n' % (dates_full[i], nVec[i][j], ' '.join([str(k) for k in sVec[i][j]])))
                    #print('nVec data type is integer')
                else:
                    f.write('%d\t%.8f\t%s\n' % (dates_full[i], nVec[i][j], ' '.join([str(k) for k in sVec[i][j]])))
                    #print('nVec data type is float')
    else:
        #nVec_new, sVec_new = add_previously_infected(nVec, sVec, nm_popsize[sim], decay_rate)
            
        # smooth nVec over the window
        nVec, sVec = smooth_nvec(nVec, sVec, window)
            
        single_infectious = np.zeros((len(nVec), len(mutant_sites)))
        single_newly_inf  = np.zeros((len(nVec), len(mutant_sites)))
        for t in range(len(nVec)):
            """
            if t < len(nVec)-1:
                for k in range(len(mutant_sites)):
                    if np.sum(nVec[t+1])==0: Q = 1
                    else:                    Q = np.sum(nVec[t+1])
                    single_new_temp[k] += np.sum([nVec[t+1][j] * sVec[t+1][j][k] for j in range(len(sVec[t+1]))]) / Q
            """
            for k in range(len(mutant_sites)):
                if np.sum(nVec[t])==0: Q = 1
                else:                  Q = np.sum(nVec[t])
                single_newly_inf[t, k]  += np.sum([nVec[t][j] * sVec[t][j][k] for j in range(len(sVec[t]))]) / Q
            if type(ss_pop_size)==type(np.zeros(5)): temp_pop = ss_pop_size[t]
            else:                                    temp_pop = ss_pop_size
            nVec_t = list(np.array([i for i in nVec[t]]) * temp_pop / np.sum(nVec[t]))
            sVec_t = [i for i in sVec[t]]
            for t_old in range(t):
                if type(ss_pop_size)==type(np.zeros(5)): temp_pop2 = ss_pop_size[t_old]
                else:                                    temp_pop2 = ss_pop_size
                probability = np.exp(- decay_rate * (t - t_old)) 
                old_nVec    = list(probability * np.array(nVec[t_old]) * temp_pop2 / np.sum(nVec[t_old]))
                old_sVec    = sVec[t_old]
                nVec_temp   = []
                sVec_temp   = []
                for j in range(len(old_nVec)):
                    if int(round(old_nVec[j]))>0:
                        nVec_temp.append(old_nVec[j])
                        sVec_temp.append(old_sVec[j])
                nVec_t, sVec_t = combine(nVec_t, sVec_t, nVec_temp, sVec_temp)
            for k in range(len(mutant_sites)):
                if np.sum(nVec_t)==0: Q  = 1
                else:                 Q  = np.sum(nVec_t)
                single_infectious[t, k] += np.sum([nVec_t[j] * sVec_t[j][k] for j in range(len(sVec_t))]) / Q
            for i in range(len(nVec_t)):
                f.write('%d\t%d\t%s\n' % (dates_full[t], nVec_t[i], ' '.join([str(k) for k in sVec_t[i]])))
        coefficient       = (1 / ((1 / (ss_pop_size * k_ss)) + ((k_ss / R_ss) / (ss_pop_size * k_ss - 1))))
        deltax_temp = np.zeros(np.shape(single_infectious))
        for t in range(len(single_infectious)):
            if not delay and t < len(single_infectious) - 1:
                deltax_temp[t] = single_newly_inf[t+1] - single_infectious[t]
            elif delay:
                if t < len(single_infectious) - 1 - delay:
                    deltax_temp[t] = single_newly_inf[t+delay] - single_infectious[t]
        deltax_int  = np.sum(coefficient * np.swapaxes(deltax_temp, 0, 1), axis=1)
    f.close()
    
    if timed > 0:
        covar_time = timer()
    
    covar_file = f'covar-{location}.dat'
    num_file   = f'num-{location}.dat'
    out_file   = f'out-{location}.dat'
    if os.path.exists(os.path.join(covar_dir, covar_file)):
        os.remove(os.path.join(covar_dir, covar_file))
        
    if not arg_list.numerator_only:
        if arg_list.find_counts:
            double_file = f'double-{location}.dat'
            proc = subprocess.Popen(['./bin/mpl', '-d', covar_dir, '-i', seq_file, '-o', out_file, '-g', str(0), '-N', f'{ss_pop_size}', '-sc', covar_file, '-sn', num_file, '-q', str(q), '-dc', double_file], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            covar_file = double_file
        else: 
            proc = subprocess.Popen(['./bin/mpl', '-d', covar_dir, '-i', seq_file, '-o', out_file, '-g', str(0), '-N', f'{ss_pop_size}', '-sc', covar_file, '-sn', num_file, '-q', str(q)], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            #proc = subprocess.Popen(['./bin/mpl', '-d', covar_dir, '-i', seq_file, '-o', out_file, '-g', str(0), '-N', f'{ss_pop_size}', '-sc', covar_file, '-sn', num_file, '-q', str(q)], 
            #                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
            #for line in proc.stdout:
                #print(line, end='')
        
        #proc = subprocess.run(['./bin/mpl', '-d', covar_dir, '-i', seq_file, '-o', out_file, '-g', str(0), '-N', f'{ss_pop_size}', '-sc', covar_file, '-sn', num_file, '-q', str(q)], 
        #                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        #print('stdout:', proc.stdout)
        #print('stderr:', proc.stderr)
        #if proc.returncode!=0:
            #print('return code:', proc.returncode)
        
        #proc.wait()
        
        stdout, stderr = proc.communicate()
        print2(proc.returncode)
        print2('stdout', stdout)
        print2('stderr', stderr)
        proc.wait()
        
        if arg_list.tv_inference:
            cov_path = os.path.join(covar_dir, covar_file)
            proc     = subprocess.Popen(['gzip', cov_path])
            proc.wait()
        
        if timed > 0:
            covar_finish_time = timer()
            print2(f'calculating the covariance or counts in C++ took {covar_finish_time - covar_time} seconds')
        
        # Load the integrated covariance matrix
        ### WHY IS THIS INDEXED TO START AFTER THE FIRST TIME POINT ###
        total_seqs    = [np.sum(i) for i in nVec][1:]
        dates_covar   = [dates_full[i] for i in range(1, len(dates_full)) if total_seqs[i-1]!=0]
        dates_nocovar = [dates_full[i] for i in range(1, len(dates_full)) if (dates_full[i] not in dates_covar and dates_full[i]>dates_covar[0])]
        covar_int     = []
        counter       = 0
        if arg_list.tv_inference:
            #covar_int = [[]]
            counter = 0
            tv_covar_dir = os.path.join(out_str, 'tv_covar')
            if not os.path.exists(tv_covar_dir):
                os.mkdir(tv_covar_dir)
            print2('made time-varying covariance directory')
        
        if timed > 0:
            save_covar_old = timer()
        
        if arg_list.tv_inference:
            for line in gzip.open(cov_path + '.gz', mode='rt'):
                if line!='\n':
                    #covar_int[-1].append(np.array(line.split(' '), dtype=float))
                    covar_int.append(np.array(line.split(' '), dtype=float))
                else:
                    #old_file = os.path.join(tv_covar_dir, location + f'___{dates_full[counter - 1]}.npz')
                    #new_file = os.path.join(tv_covar_dir, location + f'___{dates_full[counter]}.npz')
                    #shutil.copy(old_file, new_file)
                    print2(f'saving covariance at time {counter} out of {len(nVec)}')
                    if timed > 0:
                        save_covar_new = timer()
                        print(f'\ttook {save_covar_new - save_covar_old} seconds to save covariance')
                        save_covar_old = timer()
                        
                    #temp_out = os.path.join(tv_covar_dir, location + f'___{dates_covar[counter]}.npz')
                    if counter >= len(dates_covar):
                        continue                      
                    f = open(os.path.join(tv_covar_dir, location + f'___{dates_covar[counter]}.npz'), mode='wb')
                    np.savez_compressed(f, covar=np.array(covar_int, dtype=float))
                    f.close()
                    counter   += 1
                    covar_int  = []
        else: 
            for line in open(os.path.join(covar_dir, covar_file)):
                if arg_list.find_counts:
                    #print(line.split(' '))
                    covar_int.append(np.array(np.array(line.split(' ')[:-1])[::2], dtype=float))
                else:
                    new_line     = line.split(' ')
                    new_line[-1] = new_line[-1][:-1]
                    covar_int.append(np.array(new_line, dtype=float))
                    counter += 1
                
        """
        for line in open(os.path.join(covar_dir, covar_file)):
            if arg_list.tv_inference:
                if line!='\n':
                    #covar_int[-1].append(np.array(line.split(' '), dtype=float))
                    covar_int.append(np.array(line.split(' '), dtype=float))
                else:
                    #old_file = os.path.join(tv_covar_dir, location + f'___{dates_full[counter - 1]}.npz')
                    #new_file = os.path.join(tv_covar_dir, location + f'___{dates_full[counter]}.npz')
                    #shutil.copy(old_file, new_file)
                    print2(f'saving covariance at time {counter} out of {len(nVec)}')
                    #temp_out = os.path.join(tv_covar_dir, location + f'___{dates_covar[counter]}.npz')
                    if counter >= len(dates_covar):
                        continue                      
                    f = open(os.path.join(tv_covar_dir, location + f'___{dates_covar[counter]}.npz'), mode='wb')
                    np.savez_compressed(f, covar=np.array(covar_int, dtype=float))
                    f.close()
                    counter   += 1
                    covar_int  = []
            elif arg_list.find_counts:
                #print(line.split(' '))
                covar_int.append(np.array(np.array(line.split(' ')[:-1])[::2], dtype=float))
            else:
                new_line     = line.split(' ')
                #if counter==0:
                #    print(new_line)
                new_line[-1] = new_line[-1][:-1]
                #if counter==0:
                #    print(new_line)
                covar_int.append(np.array(new_line, dtype=float))
                counter += 1
        """
        #if arg_list.tv_inference:
        #    covar_int = np.array([np.array(covar_int[i], dtype=float) for i in range(len(covar_int)-1)]) 
        covar_int = np.array(covar_int, dtype=float)
        # print(covar_int[:10])
        if timed > 0:
            covar_load_time = timer()
            print2(f'loading the covariance and counts took {covar_load_time - covar_finish_time} seconds')
    else:
        covar_int = [0]
        if timed > 0:
            covar_load_time = timer()
            print2(f'no covariance calculated')
            
    if arg_list.tv_inference and not arg_list.numerator_only:
        for t in dates_nocovar:
            old_file = os.path.join(tv_covar_dir, location + f'___{t-1}.npz')
            new_file = os.path.join(tv_covar_dir, location + f'___{t}.npz')
            shutil.copy(old_file, new_file)
            
    if os.path.exists(os.path.join(covar_dir, seq_file)):
        os.remove(os.path.join(covar_dir, seq_file))
    if os.path.exists(os.path.join(covar_dir, covar_file)):
        os.remove(os.path.join(covar_dir, covar_file))
    elif os.path.exists(os.path.join(covar_dir, covar_file + '.gz')):
        os.remove(os.path.join(covar_dir, covar_file + '.gz'))
    if os.path.exists(os.path.join(covar_dir, num_file)):
        os.remove(os.path.join(covar_dir, num_file))
    if os.path.exists(os.path.join(covar_dir, out_file)):
        os.remove(os.path.join(covar_dir, out_file))
    os.chdir(working_dir)
    
    if arg_list.find_counts:
        single_temp = trajectory_calc(nVec, sVec, mutant_sites, d=q)
        single_site = moving_average(single_temp, window)
        #max_f         = max_freq_counter(nVec, sVec, mutant_sites, q)
        max_f         = np.amax(single_site, axis=0)
        single_counts = np.sum(allele_counter(nVec, sVec, mutant_sites, d=q), axis=0)
        single_counts = np.array(single_counts).flatten()
        g = open(os.path.join(out_str, location + '.npz'), mode='wb')
        np.savez_compressed(g, double_counts=covar_int, single_counts=single_counts, allele_number=allele_number, max_freq=max_f, ref_sites=ref_sites)
        g.close()
        sys.exit()
    coefficient = (1 / ((1 / (ss_pop_size * k_ss)) + ((k_ss / R_ss) / (ss_pop_size * k_ss - 1)))) * (ss_pop_size * k_ss) / (ss_pop_size * k_ss + 1)
    covar_int = np.array(covar_int, dtype=float) * coefficient / 5
        
    # Find the numerator
    if timed > 0:
        last_RHS_time = timer()
    if not arg_list.tv_inference:
        dates_RHS = dates_full
        if decay_rate==0:
            if not isinstance(ss_pop_size, int):
                RHS_temp = calculate_deltax_tv(nVec, sVec, k_ss, R_ss, ss_pop_size, mutant_sites, delta_t) - inflow_term
            else:
                RHS_temp = calculate_deltax(nVec, sVec, k_ss, R_ss, ss_pop_size, delta_t, d=q) - inflow_term
        else:
            RHS_temp = (deltax_int - inflow_term) / 5
    else:
        # The arrays for the RHS will have length of the time series minus delta_t entries do to the smoothing. So really the first time point is dates_full[0] + delta_t
        RHS_temp  = []
        dates_RHS = []
        #dates_RHS = dates_full[(delta_t - 1):]
        #covar_int = np.cumsum(covar_int, axis=0)
        
        #if timed > 0:
        #    t_cumsum = timer()
        #    print2(f'took {t_cumsum - covar_load_time} seconds to calculate the cumulative sum of the covariance matrix')
        
        """
        for t_max in dates_full:
            nVec_temp = nVec[:list(dates_full).index(t_max)]
            sVec_temp = sVec[:list(dates_full).index(t_max)]
            if len(nVec_temp) < delta_t:
                continue
            if isinstance(k_ss, list): k_temp = k_ss[:list(dates_full).index(t_max)]
            else:                      k_temp = k_ss
            if isinstance(R_ss, list): R_temp = R_ss[:list(dates_full).index(t_max)]
            else:                      R_temp = R_ss
            if isinstance(ss_pop_size, list): pop_temp = ss_pop_size[:list(dates_full).index(t_max)]
            else:                             pop_temp = ss_pop_size
            RHS_temp.append(calculate_deltax(nVec_temp, sVec_temp, k_temp, R_temp, pop_temp, delta_t))
            dates_RHS.append(t_max)
            if timed > 0:
                new_RHS_time  = timer()
                print2(f'calculating delta x for time {list(dates_full).index(t_max)} out of {len(dates_full)} took {new_RHS_time - last_RHS_time} seconds')
                last_RHS_time = new_RHS_time
        """
            
    print2('shape of RHS:       ', np.shape(RHS_temp))
    print2('shape of covariance:', np.shape(covar_int))
    
    if timed > 0:
        numerator_time = timer()
        print2(f'calculating the numerator took {numerator_time - covar_load_time} seconds')
        print2(f'total time = {numerator_time - t_start}')    
    
    if not arg_list.tv_inference and not arg_list.find_counts:
        g = open(os.path.join(out_str, location + '.npz'), mode='wb')
        if not arg_list.trajectories:
            np.savez_compressed(g, covar=covar_int, RHS=RHS_temp, location=location, times=dates, times_full=dates_full, 
                                allele_number=allele_number, k=k, N=pop_size, R=R, inflow=inflow_term, dates_RHS=dates_RHS,
                                ref_sites=ref_sites)
        else:
            np.savez_compressed(g, covar=covar_int, RHS=RHS_temp, location=location, traj=single_site, times=dates,
                                times_full=dates_full, allele_number=allele_number, k=k, N=pop_size, R=R, inflow=inflow_term, 
                                traj_nosmooth=single_temp, dates_RHS=dates_RHS, ref_sites=ref_sites)
        g.close()
    elif not arg_list.find_counts:
        counts   = allele_counter(nVec, sVec, mutant_sites, d=q)
        num_seqs = [np.sum(nVec[i]) for i in range(len(nVec))]
        counts   = freqs_from_counts(counts, num_seqs, window=delta_t)
        g = open(os.path.join(out_str, location + '.npz'), mode='wb')
        if not arg_list.trajectories:
            np.savez_compressed(g, RHS=RHS_temp, location=location, times=dates, times_full=dates_full, 
                                allele_number=allele_number, k=k, N=pop_size, R=R, inflow=inflow_term, 
                                dates_RHS=dates_RHS, counts=counts, ref_sites=ref_sites)
        else:
            np.savez_compressed(g, RHS=RHS_temp, location=location, traj=single_site, times=dates,
                                times_full=dates_full, allele_number=allele_number, k=k, N=pop_size, R=R, 
                                inflow=inflow_term, traj_nosmooth=single_temp, dates_RHS=dates_RHS, counts=counts,
                                ref_sites=ref_sites)
        g.close()
        
    

    
    
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    
    
    

