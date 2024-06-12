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
import data_processing as dp

REF_TAG = 'EPI_ISL_402125'
NUC     = ['-', 'A', 'C', 'G', 'T']
ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+++++++++++++++++++++++++++'
NUMBERS  = list('0123456789')

    
def get_label_short(i):
    index = i[:-2]
    if index[-1] in NUMBERS:
        return dp.get_label(index)
    else:
        if index[-1] in list(ALPHABET) and index[-2] in list(ALPHABET):
            temp = dp.get_label(index[:-2])
            gap  = index[-2:]
        elif index[-1] in list(ALPHABET):
            temp = dp.get_label(index[:-1])
            gap  = index[-1]
        else:
            temp = dp.get_label(index)
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
        np.savez_compressed(
            g, 
            error_bars=error_bars, 
            selection=selection, 
            allele_number=allele_number,
            selection_independent=selection_nocovar, 
            inflow_tot=inflow_full, 
            locations=locations)
    else:
        np.savez_compressed(g, selection=selection, allele_number=allele_number)
    g.close()
    
    

    
    
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    
    
    

