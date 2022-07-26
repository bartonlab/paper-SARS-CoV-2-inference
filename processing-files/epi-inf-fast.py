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
ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+++++++++++++++++++++++++++'
NUMBERS  = list('0123456789')


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


def get_label(i):
    """ For a SARS-CoV-2 reference sequence index i, return the label in the form 
    'coding region - protein number-nucleotide in codon number'. 
    For example, 'ORF1b-204-1'. 
    Should check to make sure NSP12 labels are correct due to the frame shift."""
    i = int(i)
    frame_shift = str(i - get_codon_start_index(i))
    if   (25392<=i<26220):
        return "ORF3a-" + str(int((i - 25392) / 3) + 1)  + '-' + frame_shift
    elif (26244<=i<26472):
        return "E-"     + str(int((i - 26244) / 3) + 1)  + '-' + frame_shift
    elif (27201<=i<27387):
        return "ORF6-"  + str(int((i - 27201) / 3) + 1)  + '-' + frame_shift
    # ORF7a and ORF7b overlap by 4 nucleotides
    elif (27393<=i<27759):
        return "ORF7a-" + str(int((i - 27393) / 3) + 1)  + '-' + frame_shift
    elif (27755<=i<27887):
        return "ORF7b-" + str(int((i - 27755) / 3) + 1)  + '-' + frame_shift
    elif (  265<=i<805):
        return "NSP1-"  + str(int((i - 265  ) / 3) + 1)  + '-' + frame_shift
    elif (  805<=i<2719):
        return "NSP2-"  + str(int((i - 805  ) / 3) + 1)  + '-' + frame_shift
    elif ( 2719<=i<8554):
        return "NSP3-"  + str(int((i - 2719 ) / 3) + 1)  + '-' + frame_shift
            # Compound protein containing a proteinase, a phosphoesterase transmembrane domain 1, etc.
    elif ( 8554<=i<10054):
        return "NSP4-"  + str(int((i - 8554 ) / 3) + 1)  + '-' + frame_shift
            # Transmembrane domain 2
    elif (10054<=i<10972):
        return "NSP5-"  + str(int((i - 10054) / 3) + 1)  + '-' + frame_shift
            # Main proteinase
    elif (10972<=i<11842):
        return "NSP6-"  + str(int((i - 10972) / 3) + 1)  + '-' + frame_shift
            # Putative transmembrane domain
    elif (11842<=i<12091):
        return "NSP7-"  + str(int((i - 11842) / 3) + 1)  + '-' + frame_shift
    elif (12091<=i<12685):
        return "NSP8-"  + str(int((i - 12091) / 3) + 1)  + '-' + frame_shift
    elif (12685<=i<13024):
        return "NSP9-"  + str(int((i - 12685) / 3) + 1)  + '-' + frame_shift
            # ssRNA-binding protein
    elif (13024<=i<13441):
        return "NSP10-" + str(int((i - 13024) / 3) + 1)  + '-' + frame_shift
            # CysHis, formerly growth-factor-like protein
    # Check that aa indexing is correct for NSP12, becuase there is a -1 ribosomal frameshift at 13468. It is not, because the amino acids in the first frame
    # need to be added to the counter in the second frame.
    elif (13441<=i<13467):
        return "NSP12-" + str(int((i - 13441) / 3) + 1)  + '-' + frame_shift
    elif (13467<=i<16236):
        return "NSP12-" + str(int((i - 13467) / 3) + 10) + '-' + frame_shift
            # RNA-dependent RNA polymerase
    elif (16236<=i<18039):
        return "NSP13-" + str(int((i - 16236) / 3) + 1)  + '-' + frame_shift
            # Helicase
    elif (18039<=i<19620):
        return "NSP14-" + str(int((i - 18039) / 3) + 1)  + '-' + frame_shift
            # 3' - 5' exonuclease
    elif (19620<=i<20658):
        return "NSP15-" + str(int((i - 19620) / 3) + 1)  + '-' + frame_shift
            # endoRNAse
    elif (20658<=i<21552):
        return "NSP16-" + str(int((i - 20658) / 3) + 1)  + '-' + frame_shift
            # 2'-O-ribose methyltransferase
    elif (21562<=i<25384):
        return "S-"     + str(int((i - 21562) / 3) + 1)  + '-' + frame_shift
    elif (28273<=i<29533):
        return "N-"     + str(int((i - 28273) / 3) + 1)  + '-' + frame_shift
    elif (29557<=i<29674):
        return "ORF10-" + str(int((i - 29557) / 3) + 1)  + '-' + frame_shift
    elif (26522<=i<27191):
        return "M-"     + str(int((i - 26522) / 3) + 1)  + '-' + frame_shift
    elif (27893<=i<28259):
        return "ORF8-"  + str(int((i - 27893) / 3) + 1)  + '-' + frame_shift
    else:
        return "NC-"    + str(int(i))
    
    
def get_label2(i):
    return get_label(i[:-2]) + '-' + i[-1]


def get_label_new(i):
    nuc   = i[-1]
    index = i.split('-')[0]
    if index[-1] in NUMBERS:
        return get_label2(i)
    else:
        if index[-1] in list(ALPHABET) and index[-2] in list(ALPHABET):
            temp = get_label(index[:-2])
            gap  = index[-2:]
        elif index[-1] in list(ALPHABET):
            temp = get_label(index[:-1])
            gap  = index[-1]
        else:
            temp = get_label(index)
            gap  = None
        temp = temp.split('-')
        if gap is not None:
            temp[1] += gap
            #print(temp, gap)
        temp.append(nuc)
        label = '-'.join(temp)
        return label
    
def get_label_short(i):
    index = i[:-2]
    if index[-1] in NUMBERS:
        return get_label(index)
    else:
        if index[-1] in list(ALPHABET) and index[-2] in list(ALPHABET):
            temp = get_label(index[:-2])
            gap  = index[-2:]
        elif index[-1] in list(ALPHABET):
            temp = get_label(index[:-1])
            gap  = index[-1]
        else:
            temp = get_label(index)
            gap  = None
        temp = temp.split('-')
        if gap is not None:
            temp[1] += gap
            #print(temp, gap)
        label = '-'.join(temp)
        return label


def main(args):
    """Infer selection coefficients from individual files containing the covariance, trajectory, and single site frequencies for different regions"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',            type=str,    default='inferred-coefficients', help='output file')
    parser.add_argument('-R',            type=float,  default=2,                       help='the basic reproduction number')
    parser.add_argument('-k',            type=float,  default=0.1,                     help='parameter determining shape of distribution of new infected')
    parser.add_argument('-q',            type=int,    default=5,                       help='number of mutant alleles per site')
    parser.add_argument('--data',        type=str,    default=None,                    help='directory to .npz files containing the RHS and the covariance matrix, and mutant_sites for different locations')
    parser.add_argument('--traj_data',   type=str,    default=None,                    help='the trajectory data if it exists')
    parser.add_argument('--g1',          type=float,  default=40,                      help='regularization restricting the magnitude of the selection coefficients')
    parser.add_argument('--timed',       type=int,    default=0,                       help='if 0, wont print any time information, if 1 will print some information')
    parser.add_argument('--pop_size',    type=int,    default=10000,                   help='the population size')
    parser.add_argument('--alleles',     type=str,    default=None,                    help='the mutations')
    parser.add_argument('--mask_site',   type=int,    default=None,                    help='the site to mask (change to WT) when inferring the other coefficients, in order to check effect of linkage (write)')
    parser.add_argument('--mask_group',  type=str,    default=None,                    help='the .npy file containing a list of sites (as nucleotide numbers) to mask for the inference')
    parser.add_argument('--numerator',   type=str,    default=None,                    help='the numerator for doing the inference')
    parser.add_argument('--covariance',  type=str,    default=None,                    help='the covariance matrix for doing the inference')
    parser.add_argument('--decay_rate',  type=float,  default=0,                       help='the exponential decay rate used to correct for infection lasting multiple generations')
    parser.add_argument('--nm_popsize',  type=str,    default=None,                    help='.csv file containing the population sizes used to correct for infection lasting multiple generations')
    parser.add_argument('--delay',       type=int,    default=None,                    help='the delay between the newly infected individuals and the individuals that ')
    parser.add_argument('--infFile',     type=str,    default=None,                    help='a file containing the results of an inference, including the site names, covariance, and numerator')
    parser.add_argument('--count_data',  type=str,    default=None,                    help='the file containing the total counts for the alleles at different sites')
    parser.add_argument('--syn_data',    type=str,    default=None,                    help='the file containing information about which mutations are synonymous and which are nonsynonymous')
    parser.add_argument('--refFile',     type=str,    default='ref-index.csv',         help='the file containing the reference sequence indices and nucleotides')
    parser.add_argument('--eliminateNC',    action='store_true',  default=False,  help='whether or not to eliminate non-coding sites')
    parser.add_argument('--eliminateNS',    action='store_true',  default=False,  help='whether or not to eliminate non-synonymous mutations')
    
    arg_list = parser.parse_args(args)
    
    out_str       = arg_list.o
    g1            = arg_list.g1
    q             = arg_list.q
    timed         = arg_list.timed
    directory_str = arg_list.data
    mask_site     = arg_list.mask_site
    decay_rate    = arg_list.decay_rate
    delay         = arg_list.delay
    g1            = arg_list.g1
    k             = arg_list.k
    R             = arg_list.R
    N             = arg_list.pop_size
    old_data      = np.load(arg_list.infFile, allow_pickle=True)
    numerator     = old_data['numerator']
    covar         = old_data['covar_int']
    allele_number = old_data['allele_number']
    locations     = old_data['locations']
    #numerator     = np.load(arg_list.numerator,  allow_pickle=True)
    #covar         = np.load(arg_list.covariance, allow_pickle=True)
    #allele_number = np.load(arg_list.alleles,    allow_pickle=True)
    A = covar
    b = numerator
    L = int(len(b) / 5) 
    print(L)
    print(len(b))
    print(np.shape(A))
    print(len(allele_number))
    
    status_name = f'inference-fast-status.csv'
    status_file = open(status_name, 'w')
    status_file.close()
    
    print(allele_number)
    if arg_list.eliminateNC and not arg_list.eliminateNS:
        prots = [get_label_short(i).split('-')[0] for i in allele_number]
        mask  = []
        for i in range(len(prots)):
            if prots[i]!='NC':
                mask.append(i)
        mask = np.array(mask)
        A    = A[:, mask][mask]
        b    = b[mask]
        L    = int(len(b) / q)
        allele_number = allele_number[np.array(prots)!='NC']
        
    elif arg_list.eliminateNS:
        # Eliminating synonymous mutations
        cutoff     = 20
        count_data = np.load(arg_list.count_data, allow_pickle=True)
        counts     = count_data['mutant_counts']
        alleles2   = count_data['allele_number']
        mask       = np.isin(alleles2, allele_number)
        alleles2   = alleles2[mask]
        counts     = counts[mask]

        syn_data  = np.load(arg_list.syn_data, allow_pickle=True)
        syn_muts  = syn_data['nuc_index']
        types     = syn_data['types']
        mask      = np.isin(syn_muts, allele_number)
        types     = types[mask]
        syn_muts  = syn_muts[mask]
        new_types = []
        new_muts  = []
        for i in range(len(allele_number)):
            if allele_number[i] not in syn_muts:
                if allele_number[i][-1]=='-':
                    new_types.append('S')
                else:
                    new_types.append('NS')
            else:
                new_types.append(types[list(syn_muts).index(allele_number[i])])
            new_muts.append(allele_number[i])
        types = np.array(new_types)
        assert all(np.array(new_muts)==allele_number)
        assert all(alleles2==allele_number)

        idxs_keep = []
        for i in range(int(len(allele_number) / 5)):
            types_temp  = types[5 * i : 5 * (i + 1)]
            counts_temp = counts[5 * i : 5 * (i + 1)]
            types_temp  = [types_temp[j] for j in range(len(types_temp)) if counts_temp[j] > cutoff]
            if 'NS' in types_temp:
                for j in range(5):
                    idxs_keep.append((5 * i) + j)
        idxs_keep = np.array(idxs_keep)

        numerator = numerator[idxs_keep]
        covar     = covar[idxs_keep][:, idxs_keep]
        counts    = counts[idxs_keep]
        types     = types[idxs_keep]
        allele_number = allele_number[idxs_keep]
        
        A = covar
        b = numerator
        L = int(len(b) / 5) 
        print(L)
        print(len(b))
        print(np.shape(A))
        print(len(allele_number))
    
    def print2(*args):
        """ Print the status of the processing and save it to a file."""
        stat_file = open(status_name, 'a+')
        line      = [str(i) for i in args]
        string    = '\t'.join(line)
        stat_file.write(string+'\n')
        stat_file.close()
        print(string)
    
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
        coefficient = (1 / ((1 / (N * k)) + ((k / R) / (N * k - 1)))) * (1 / R)
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
        t_elimination = timer()
        print2("starting")
    
    #new_mut_types = np.zeros(1)
    
    g1 *= np.mean(N) * np.mean(k) / (1 + (np.mean(k) / np.mean(R)))  # The regularization
    print2('regularization: \t', g1)
    print2('number of sites:\t', L)
   
    if timed > 0:
        tracker = 0
        t_combine = 0
    inflow_full            = []    # The correction to the first moment due to migration into the population
    traj_nosmooth          = []    # The unsmoothed frequency trajectories
    single_freq_infectious = []
    single_newly_infected  = []
    delta_tot              = []
    covar_full             = []
    RHS_full               = []
    traj_full              = []
    times_full             = []
    """
    for sim in range(len(filepaths)):
        # get data for the specific location
        data         = np.load(filepaths[sim], allow_pickle=True)  # Genome sequence data
        covar        = data['covar']
        RHS          = data['RHS']
        traj         = data['traj']
        inflow       = data['inflow']
        times        = data['times']
        mutant_sites = mutant_sites_tot[sim]
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
        
        for i in range(len(mutant_sites)):
            for j in range(q):
                b[positions[i] * q + j] += RHS[i * q + j]
                for k in range(i + 1, len(mutant_sites)):
                    for l in range(q):
                        A[positions[i] * q + j, positions[k] * q + l] += covar[i * q + j, k * q + l]
                        A[positions[k] * q + l, positions[i] * q + j] += covar[i * q + j, k * q + l]

        for i in range(len(mutant_sites)):
            loc1 = np.where(allele_number == mutant_sites[i])[0][0]
            for k in range(q):
                b[loc1 * q + k] += RHS[i * q + k]
                for j in range(i+1, len(mutant_sites)):
                    loc2 = np.where(allele_number == mutant_sites[j])[0][0]
                    for l in range(q):
                        A[loc1 * q + k, loc2 * q + l] += covar[i * q + k, j * q + l]
                        A[loc2 * q + l, loc1 * q + k] += covar[i * q + k, j * q + l]
    """
                
    print2('means')
    print2(np.mean(b))
    print2(np.mean(A))
    print2('max')
    print2(np.amax(b))
    print2(np.amax(A))
    
    print('number of nonzero entries in A:', np.nonzero(A))
    #print('total number of entries in A:  ', len(A) * len(A[0]))
    
    print2('g1', g1)
    print2('average absolute covariance', np.mean(np.absolute(A)))
    
    if timed > 0:
        t_solve_system = timer()
    # Apply the regularization
    for i in range(L * q):
        A[i,i] += g1
    print2('regularization applied')
    #error_bars        = np.sqrt(np.absolute(np.diag(linalg.inv(A))))
    error_bars        = 1 / np.sqrt(np.absolute(np.diag(A)))
    print2('error bars found')
    selection         = linalg.solve(A, b, assume_a='sym')
    print2('selection coefficients found')
    selection_nocovar = b / np.diag(A)
    
    # loading reference sequence index
    alleles_temp = [i[:-2] for i in allele_number]
    ref_index = pd.read_csv(arg_list.refFile)
    index     = list(ref_index['ref_index'])
    index     = [str(i) for i in index]
    ref_full  = list(ref_index['nucleotide'])
    ref_poly  = []
    for i in range(len(index)):
        if index[i] in alleles_temp:
            ref_poly.append(ref_full[i])
    
    # Normalize selection coefficients so reference allele has selection coefficient of zero
    print2('max selection coefficient before Gauge transformation:', np.amax(selection))
    selection  = np.reshape(selection, (L, q))
    error_bars = np.reshape(error_bars, (L, q))
    selection_nocovar = np.reshape(selection_nocovar, (L, q))
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
    
    f = open(out_str+'-unmasked.npz', mode='w+b')
    np.savez_compressed(f, error_bars=error_bars, selection=selection, allele_number=allele_number,
                            selection_independent=selection_nocovar, inflow_tot=inflow_full)
    f.close()
    
    if arg_list.eliminateNS:
        print(len(counts))
        print(len(types))
        print(len(allele_number))
    # Eliminating the remaining synonymous mutations and mutations that don't appear in the data
    
    ### THERE APPEARS TO BE AN ISSUE WITH THIS PART ###
    mask      = np.nonzero(np.logical_and(types=='NS', counts>cutoff))[0]
    selection = selection[mask]
    s_ind     = selection_nocovar[mask]
    errors    = error_bars[mask]
    allele_number = allele_number[mask]
    
          
    if timed > 0:
        t_linked = timer()
        print2("calculating the inferred coefficients", t_linked - t_solve_system)
        
    # save the solution  
    for i in range(L * q):
        A[i,i] -= g1
    g = open(out_str+'.npz', mode='w+b') 
    if not mask_site:
        #np.savez_compressed(g, error_bars=error_bars, selection=selection, traj=traj_full, 
        #                    mutant_sites=mutant_sites_tot, allele_number=allele_number, locations=locations, 
        #                    times=dates, covar_tot=covar_full, selection_independent=selection_nocovar, 
        #                    covar_int=A, inflow_tot=inflow_full, times_full=dates_full, delta_tot=RHS_full)
        np.savez_compressed(g, error_bars=error_bars, selection=selection, allele_number=allele_number,
                            selection_independent=selection_nocovar, inflow_tot=inflow_full, locations=locations)
    else:
        np.savez_compressed(g, selection=selection, allele_number=allele_number)
    g.close()
    
    

    
    
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    
    
    

