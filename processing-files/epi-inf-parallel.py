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
from scipy import linalg

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


def get_codon_start_index(i):
    """ Given a sequence index i, determine the index of the first nucleotide in the codon. """
    if   (13467<=i<=21554):
        return i - (i - 13467)%3
    elif (25392<=i<=26219):
        return i - (i - 25392)%3
    elif (26244<=i<=26471):
        return i - (i - 26244)%3
    elif (27201<=i<=27386):
        return i - (i - 27201)%3
    # new to account for overlap of orf7a and orf7b
    #elif (27393<=i<=27754):
    #    return i - (i - 27393)%3
    #elif (27755<=i<=27886):
    #    return i - (i - 27755)%3
    ### considered orf7a and orf7b as one reading frame (INCORRECT)
    elif (27393<=i<=27886):
        return i - (i - 27393)%3
    ### REMOVE ABOVE
    elif (  265<=i<=13467):
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
    
def get_label2(i):
    return get_label(i[:-2]) + '-' + i[-1]


def main(args):
    """Infer selection coefficients from individual files containing the covariance, trajectory, and single site frequencies for different regions"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',            type=str,    default='inferred-coefficients', help='output file')
    parser.add_argument('-R',            type=float,  default=2,                       help='the basic reproduction number')
    parser.add_argument('-k',            type=float,  default=0.1,                     help='parameter determining shape of distribution of new infected')
    parser.add_argument('-q',            type=int,    default=5,                       help='number of mutant alleles per site')
    parser.add_argument('--mu',          type=float,  default=1e-4,                    help='the mutation rate')
    parser.add_argument('--data',        type=str,    default=None,                    help='directory to .npz files containing the counts, sequences, times, and mutant_sites for different locations')
    parser.add_argument('--g1',          type=float,  default=1,                       help='regularization restricting the magnitude of the selection coefficients')
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
    #parser.add_argument('--link_tol',    type=float,  default=1,                       help='the tolerance for correlation check used for determining linked sites')
    parser.add_argument('--end_cutoff',  type=int,    default=0,                       help='the number of days at the end of the simulations to ignore due to bad sampling')
    parser.add_argument('--start_cutoff',type=int,    default=0,                       help='the number of days at the end of the simulations to ignore due to bad sampling')
    parser.add_argument('--window',      type=int,    default=10,                      help='the number of days over which to take the moving average')
    parser.add_argument('--clip_end',    type=int,    default=29700,                   help='the last site to clip the genomes to')
    parser.add_argument('--delta_t',     type=int,    default=10,                      help='the amount of dates at the beginning and the end of the time series to use to calculate delta_x')
    parser.add_argument('--mut_type',    type=str,    default=None,                    help='.npz file containing the location along the genome of a mutation under kw locations, and the type of mutation under kw types')
    parser.add_argument('--c_directory', type=str,    default='Archive',               help='directory containing the c++ scripts')
    parser.add_argument('--mask_site',   type=int,    default=None,                    help='the site to mask (change to WT) when inferring the other coefficients, in order to check effect of linkage (write)')
    parser.add_argument('--mask_group',  type=str,    default=None,                    help='the .npy file containing a list of sites (as nucleotide numbers) to mask for the inference')
    parser.add_argument('--remove_sites',type=str,    default=None,                    help='the sites to eliminate when inferring coefficients')
    parser.add_argument('--final_t',     type=int,    default=None,                    help='the last time point to be considered, cutoff all time series after this time (in days after 01-01-2020)')
    parser.add_argument('--decay_rate',  type=float,  default=0,                       help='the exponential decay rate used to correct for infection lasting multiple generations')
    parser.add_argument('--nm_popsize',  type=str,    default=None,                    help='.csv file containing the population sizes used to correct for infection lasting multiple generations')
    parser.add_argument('--delay',       type=int,    default=None,                    help='the delay between the newly infected individuals and the individuals that ')
    parser.add_argument('--mutation_on',    action='store_true',  default=False,  help='whether or not to include mutational term in inference')
    parser.add_argument('--cutoff_on',      action='store_true',  default=False,  help='whether or not to cutoff frequencies below freq_cutoff')
    parser.add_argument('--find_linked',    action='store_true',  default=False,  help='whether or not to find the sites that are (almost) fully linked')
    parser.add_argument('--tv_inference',   action='store_true',  default=False,  help='if true then the coefficients are inferred at every time point')
    parser.add_argument('--trajectory',     action='store_true',  default=False,  help='whether or not to save the trajectories in the different regions')
    parser.add_argument('--eliminateNC',    action='store_true',  default=False,  help='whether or not to eliminate non-coding sites')
    
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
    #link_tol      = arg_list.link_tol
    window        = arg_list.window
    timed         = arg_list.timed
    find_linked   = arg_list.find_linked
    directory_str = arg_list.data
    c_directory   = arg_list.c_directory
    clip_end      = arg_list.clip_end
    mask_site     = arg_list.mask_site
    final_time    = arg_list.final_t
    decay_rate    = arg_list.decay_rate
    delay         = arg_list.delay
    
    status_name = f'inference-status.csv'
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
    
    simulations = len([name for name in os.listdir(directory_str)]) # number of locations to be combined
    
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
        record = arg_list.record * np.ones(simulations)
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
        single_freq = np.zeros((T, len(mutant_sites)))
        for t in range(T):
            for i in range(len(mutant_sites)):
                for j in range(len(sVec_in[t])):
                    single_freq[t, i * d + sVec_in[t][j][i]] = nVec_in[t][j] / popsize[t]
        coefficient = (N * k * R / (R + k)) * (1 / R)
        if not isinstance(coefficient, float):
            coefficient = coefficient[:-1]
        term_2            = np.swapaxes(traj, 0, 1) * np.array(pop_in[:len(traj)]) / popsize[t]
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
        coeff1 = N_ss * k_ss * R_ss / (R_ss + k_ss)
        for i in range(len(sVec[0][0])):
            for j in range(len(sVec_start)):
                beginning[i * d + sVec_start[j][i]] += coeff1 * nVec_start[j] / np.sum(nVec_start)
            for j in range(len(sVec_end)):
                end[i * d + sVec_end[j][i]]         += coeff1 * nVec_end[j] / np.sum(nVec_end)
        return (end - beginning)
    
    
    def calculate_deltax_tv(nVec, sVec, k_ss, R_ss, N_ss, mut_sites, window):
        """ Integrates the change in frequency over the time series allowing parameters to be time-varying. """
        coeff1 = N_ss * k_ss * R_ss / (R_ss + k_ss)
        single_site = trajectory_calc(nVec, sVec, mut_sites)
        delta_x     = np.zeros(single_site[0])
        for i in range(1, len(nVec)):
            delta_x += (single_site[i] - single_site[i - 1]) * coeff1[i - 1]
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
        coeff1  = N_ss * k_ss * R_ss / (R_ss + k_ss)
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
        coeff1             = N_ss * k_ss * R_ss / (R_ss + k_ss)
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
        t_elimination = timer()
        print2("starting")
    
    # Loading and organizing the data from the different locations. Combining it into new arrays.
    scratch_dir = os.path.join(os.getcwd(), 'scratch')
    if not os.path.exists(scratch_dir):
        os.mkdir(scratch_dir)
    allele_number     = []
    mutant_sites_full = []
    dates             = []
    dates_full        = []
    filepaths         = []
    k_full            = []
    R_full            = []
    N_full            = []
    locations         = []
    directory         = os.fsencode(directory_str)
    for file in sorted(os.listdir(directory)):
        
        # Load data
        filename = os.fsdecode(file)
        filepath = os.path.join(directory_str, filename)
        location = filename[:-4]
        print2(f'\tloading location {location}')
        data     = np.load(filepath, allow_pickle=True)  # Genome sequence data
        
        alleles = np.array(data['allele_number'])
        #nuc_num = np.array([i[:-2] for i in alleles], dtype=int)
        #alleles = alleles[200<=nuc_num]
        #nuc_num = nuc_num[200<=nuc_num]
        #alleles = alleles[nuc_num>=19500]
        #alleles = alleles[alleles>=200]
        #alleles = alleles[alleles<=29500]
        mutant_sites_full.append(alleles)
        dates.append(data['times'])
        dates_full.append(data['times_full'])
        locations.append(location)
        filepaths.append(filepath)
        k_full.append(data['k'])
        R_full.append(data['R'])
        N_full.append(data['N'])
    
    # Finds all sites at which there are mutations
    mutant_sites_all = np.sort(np.unique(np.array([mutant_sites_full[i][j] for i in range(len(mutant_sites_full)) for j in range(len(mutant_sites_full[i]))])))
    
    allele_number    = mutant_sites_all
    mutant_sites_tot = mutant_sites_full
    print2("number of inferred coefficients is {mutants} out of {mutants_all} total".format(mutants=len(allele_number), mutants_all=len(mutant_sites_all)))

    #new_mut_types = np.zeros(1)
    
    L = len(allele_number)
    A = np.zeros((L * q, L * q))     # The matrix on the left hand side of the SOE used to infer selection
    b = np.zeros(L * q)         # The vector on the right hand side of the SOE used to infer selection
    g1 *= np.mean(N_full) * np.mean(k_full) * np.mean(R_full) / (k_full + R_full)  # The regularization
    print2('regularization: \t', g1)
    print2('number of sites:\t', L)
   
    if timed > 0:
        tracker   = 0
        t_combine = 0
        t_add_old = timer()
    inflow_full            = []    # The correction to the first moment due to migration into the population
    traj_nosmooth          = []    # The unsmoothed frequency trajectories
    single_freq_infectious = []
    single_newly_infected  = []
    delta_tot              = []
    covar_full             = []
    RHS_full               = []
    traj_full              = []
    times_full             = []
    for sim in range(len(filepaths)):
        # get data for the specific location
        data         = np.load(filepaths[sim], allow_pickle=True)  # Genome sequence data
        covar        = data['covar']
        RHS          = data['RHS']
        inflow       = data['inflow']
        times        = data['times']
        mutant_sites = mutant_sites_tot[sim]
        if 'traj' in data and arg_list.trajectory:
            traj = data['traj']
        else:
            traj = []
        if arg_list.tv_inference:
            covar = covar[:-len(RHS)]
            times = data['dates_RHS']
            
        times_full.append(times)
        traj_full.append(traj)
        RHS_full.append(RHS)
        #covar_full.append(covar)
        inflow_full.append(inflow)
        
        #print(sim, tracker)
        region = locations[sim][:-22]
        if   region[-2:] == '--': region = region[:-2]
        elif region[-1]  =='-':   region = region[:-1]
        
        if timed > 0:
            t_traj = timer()
            if tracker == 0:
                print2("eliminating sites whose frequency is too low", t_traj - t_elimination)
            else:
                print2(f"calculating the integrated covariance and the delta x term for location {tracker-1}, {locations[tracker-1]}", t_traj - t_combine)
        
        if timed > 0:
            t_combine = timer()
            print2(f"calculating the trajectories for location {tracker}, {locations[tracker]}", t_combine - t_traj)
            tracker += 1
        
        # Combine the covariance and the change in frequency for this location with the big covariance and frequency change arrays
        alleles_sorted = np.argsort(allele_number)
        positions      = np.searchsorted(allele_number[alleles_sorted], mutant_sites)
        positions      = alleles_sorted[positions]
        
        for i in range(len(mutant_sites)):
            b[positions[i] * q : (positions[i] + 1) * q] += RHS[i * q : (i + 1) * q]
            A[positions[i] * q : (positions[i] + 1) * q, positions[i] * q : (positions[i] + 1) * q] += covar[i * q : (i + 1) * q, i * q : (i + 1) * q]
            for k in range(i + 1, len(mutant_sites)):
                A[positions[i] * q : (positions[i] + 1) * q, positions[k] * q : (positions[k] + 1) * q] += covar[i * q : (i + 1) * q, k * q : (k + 1) * q]
                A[positions[k] * q : (positions[k] + 1) * q, positions[i] * q : (positions[i] + 1) * q] += covar[k * q : (k + 1) * q, i * q : (i + 1) * q]
        
        if timed > 0:
            t_add_new = timer()
            print(f'Took {t_add_new - t_add_old} seconds to add the covariance and frequency changes from region {locations[sim]}')
            t_add_old = t_add_new
        
    if timed > 0:
        t_solve_system = timer()
        print2(f"loading the integrated covariance and the delta x term for location {tracker-1}, {locations[tracker-1]}", t_solve_system - t_combine)
                
    print2('means')
    print2(np.mean(b))
    print2(np.mean(A))
    print2('max')
    print2(np.amax(b))
    print2(np.amax(A))
    
    print('number of nonzero entries in A:', np.nonzero(A))
    print('total number of entries in A:  ', L * L * q * q)
    
    if arg_list.eliminateNC:
        prots = [get_label(i).split('-')[0] for i in allele_number]
        mask  = []
        for i in range(len(prots)):
            if prots[i]!='NC':
                for j in range(q):
                    mask.append(q * i + j)
        mask = np.array(mask)
        A    = A[:, mask][mask]
        b    = b[mask]
        L    = int(len(b) / q)
        allele_number = allele_number[np.array(prots)!='NC']
    
    print2('g1', g1)
    print2('average absolute covariance', np.mean(np.absolute(A)))
    print2('L', L)
    print2('length of allele_number', len(allele_number))
    print2('length of b', len(b))
    print2('shape of A', np.shape(A))
    
    # Apply the regularization
    for i in range(L * q):
        A[i,i] += g1
    print2('regularization applied')
    #error_bars        = np.sqrt(np.absolute(np.diag(linalg.inv(A))))
    error_bars      = 1 / np.sqrt(np.absolute(np.diag(A)))
    print2('error bars found')
    if timed > 0:
        t_presolve = timer()
    selection      = linalg.solve(A, b, assume_a='sym')
    print2('selection coefficients found')
    if timed > 0:
        t_postsolve   = timer()
        print2(f'Took {t_postsolve - t_presolve} seconds to solve the system of equations')
    selection_nocovar = b / np.diag(A)
    
    allele_new        = []
    for i in range(len(allele_number)):
        for j in range(q):
            allele_new.append(str(allele_number[i]) + '-' + NUC[j])
    
    np.savez_compressed(out_str + '-unnormalized.npz', selection=selection, allele_number=allele_new)
    
    # Normalize selection coefficients so reference allele has selection coefficient of zero
    print2('max selection coefficient before Gauge transformation:', np.amax(selection))
    selection  = np.reshape(selection, (L, q))
    error_bars = np.reshape(error_bars, (L, q))
    selection_nocovar = np.reshape(selection_nocovar, (L, q))
    ref_seq, ref_tag  = get_MSA(REF_TAG +'.fasta')
    ref_seq  = list(ref_seq[0])
    ref_poly = np.array(ref_seq)[allele_number]
    s_new = []
    s_SL  = []
    for i in range(L):
        if q==4 or q==5: idx = NUC.index(ref_poly[i])
        elif q==2:       idx = 0
        else: print2(f'number of states at each site is {q}, which cannot be handeled by the current version of the code')
        temp_s    = selection[i]
        temp_s    = temp_s - temp_s[idx]
        temp_s_SL = selection_nocovar[i]
        temp_s_SL = temp_s_SL - temp_s_SL[idx]
        s_new.append(temp_s)
        s_SL.append(temp_s_SL)
    selection         = s_new
    selection_nocovar = s_SL
    
    ### Maybe flatten the arrays here so downstream functions dont have to be changed. Would need to relabel sites though....
    selection         = np.array(selection).flatten()
    selection_nocovar = np.array(selection_nocovar).flatten()
    error_bars        = np.array(error_bars).flatten()
    mutant_sites_new  = []
    for i in range(len(mutant_sites_tot)):
        mut_temp = []
        for j in range(len(mutant_sites_tot[i])):
            for k in range(q):
                mut_temp.append(str(mutant_sites_tot[i][j]) + '-' + NUC[k])
        mutant_sites_new.append(mut_temp)
    allele_number    = allele_new
    mutant_sites_tot = mutant_sites_new
          
    if timed > 0:
        t_linked = timer()
        print2("calculating the inferred coefficients", t_linked - t_solve_system)
    
    if timed > 0:
        t_end = timer()
        print2("total time", t_end - t_elimination)
        
    # save the solution  
    for i in range(L * q):
        A[i,i] -= g1
    g = open(out_str+'.npz', mode='w+b') 
    if not mask_site:
        #np.savez_compressed(g, error_bars=error_bars, selection=selection, traj=traj_full, 
        #                    mutant_sites=mutant_sites_tot, allele_number=allele_number, locations=locations, 
        #                    times=dates, covar_tot=covar_full, selection_independent=selection_nocovar, 
        #                    covar_int=A, inflow_tot=inflow_full, times_full=dates_full, delta_tot=RHS_full)
        np.savez_compressed(g, error_bars=error_bars, selection=selection, traj=traj_full, 
                            mutant_sites=mutant_sites_tot, allele_number=allele_number, locations=locations, 
                            times=dates, selection_independent=selection_nocovar, 
                            covar_int=A, inflow_tot=inflow_full, times_full=dates_full, delta_tot=RHS_full,
                            numerator=b)
    else:
        np.savez_compressed(g, selection=selection, allele_number=allele_number)
    g.close()
    
    

    
    
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    
    
    

