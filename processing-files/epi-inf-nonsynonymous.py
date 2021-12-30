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
    """Infer selection coefficients from individual files containing the covariance, trajectory, and single site frequencies for different regions"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',            type=str,    default='inferred-coefficients', help='output file')
    parser.add_argument('-R',            type=float,  default=2,                       help='the basic reproduction number')
    parser.add_argument('-k',            type=float,  default=0.1,                     help='parameter determining shape of distribution of new infected')
    parser.add_argument('-q',            type=int,    default=5,                       help='number of mutant alleles per site')
    parser.add_argument('--data',        type=str,    default=None,                    help='directory to .npz files containing the RHS and the covariance matrix, and mutant_sites for different locations')
    parser.add_argument('--traj_data',   type=str,    default=None,                    help='the trajectory data if it exists')
    parser.add_argument('--count_data',  type=str,    default=None,                    help='the file containing the total counts for the alleles at different sites')
    parser.add_argument('--syn_data',    type=str,    default=None,                    help='the file containing information about which mutations are synonymous and which are nonsynonymous')
    parser.add_argument('--g1',          type=float,  default=40,                      help='regularization restricting the magnitude of the selection coefficients')
    parser.add_argument('--timed',       type=int,    default=0,                       help='if 0, wont print any time information, if 1 will print some information')
    parser.add_argument('--pop_size',    type=int,    default=10000,                   help='the population size')
    #parser.add_argument('--alleles',     type=str,    default=None,                    help='the mutations')
    parser.add_argument('--mask_site',   type=int,    default=None,                    help='the site to mask (change to WT) when inferring the other coefficients, in order to check effect of linkage (write)')
    parser.add_argument('--mask_group',  type=str,    default=None,                    help='the .npy file containing a list of sites (as nucleotide numbers) to mask for the inference')
    #parser.add_argument('--numerator',   type=str,    default=None,                    help='the numerator for doing the inference')
    #parser.add_argument('--covariance',  type=str,    default=None,                    help='the covariance matrix for doing the inference')
    parser.add_argument('--decay_rate',  type=float,  default=0,                       help='the exponential decay rate used to correct for infection lasting multiple generations')
    parser.add_argument('--nm_popsize',  type=str,    default=None,                    help='.csv file containing the population sizes used to correct for infection lasting multiple generations')
    parser.add_argument('--delay',       type=int,    default=None,                    help='the delay between the newly infected individuals and the individuals that ')
    
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
    #numerator     = np.load(arg_list.numerator,  allow_pickle=True)
    #covar         = np.load(arg_list.covariance, allow_pickle=True)
    #allele_number = np.load(arg_list.alleles,    allow_pickle=True)
    data          = np.load(arg_list.data, allow_pickle=True)
    alleles       = data['allele_number']
    covar_int     = data['covar_int']
    numerator     = data['numerator']
    
    status_name = f'inference-fast-status.csv'
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
    
    # Eliminating synonymous mutations

    cutoff     = 0

    count_data = np.load(arg_list.count_data, allow_pickle=True)
    counts     = count_data['counts']
    alleles2   = count_data['allele_number']

    ### DELETE THE BELOW ###
    mask2     = np.isin(alleles, alleles2)
    alleles   = alleles[mask2]
    numerator = numerator[mask2]
    covar_int = covar_int[mask2][:, mask2]
    ### DELETE THE ABOVE

    syn_data  = np.load(arg_list.syn_data, allow_pickle=True)
    syn_muts  = syn_data['nuc_index']
    types     = syn_data['types']
    types     = types[np.isin(syn_muts, alleles)]

    idxs_keep = []
    for i in range(int(len(alleles) / 5)):
        types_temp  = types[5 * i : 5 * (i + 1)]
        counts_temp = counts[5 * i : 5 * (i + 1)]
        types_temp  = [types_temp[j] for j in range(len(types_temp)) if counts_temp[j] > cutoff]
        if 'NS' in types_temp:
            for j in range(5):
                idxs_keep.append((5 * i) + j)
    idxs_keep = np.array(idxs_keep)

    numerator = numerator[idxs_keep]
    covar_int = covar_int[idxs_keep][:, idxs_keep]
    alleles   = alleles[idxs_keep]
    counts    = counts[idxs_keep]
    types     = types[idxs_keep]

    #np.savez_compressed(os.path.join(INF_DIR, f'inf-data-nonsynonymous-{DATA_DATE}-1pct.npz'), allele_number=alleles, 
    #                    traj=traj, mutant_sites=mutants, numerator=numerator, covar_int=covar_int, counts=counts)
    
    g1 *= np.mean(N) * np.mean(k) np.mean(R) / (np.mean(R) + np.mean(k))  # The regularization
    print2('regularization: \t', g1)
   
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
        
    A = covar_int
    b = numerator
    L = int(len(b) / 5) 
                
    print2('means')
    print2(np.mean(b))
    print2(np.mean(A))
    print2('max')
    print2(np.amax(b))
    print2(np.amax(A))
    
    #print('number of nonzero entries in A:', np.nonzero(A))
    #print('total number of entries in A:  ', len(A) * len(A[0]))
    
    print2('g1', g1)
    print2('average absolute covariance', np.mean(np.absolute(A)))
    
    # Apply the regularization
    for i in range(L * q):
        A[i,i] += g1
    print2('regularization applied')
    #error_bars        = np.sqrt(np.absolute(np.diag(linalg.inv(A))))
    error_bars        = 1 / np.sqrt(np.absolute(np.diag(A)))
    print2('error bars found')
    if timed > 0:
        t_presolve  = timer()
    selection         = linalg.solve(A, b, assume_a='sym')
    if timed > 0: 
        t_postsolve = timer()
        print(f'took {t_postsolve - t_presolve} seconds to solve the system of equations')
    print2('selection coefficients found')
    selection_nocovar = b / np.diag(A)
    
    h = open(out_str + '-unnormalized.npz', mode='wb')
    np.savez_compressed(h, selection=selection, allele_number=alleles)
    h.close()
    
    # Normalize selection coefficients so reference allele has selection coefficient of zero
    print2('max selection coefficient before Gauge transformation:', np.amax(selection))
    selection  = np.reshape(selection, (L, q))
    error_bars = np.reshape(error_bars, (L, q))
    selection_nocovar = np.reshape(selection_nocovar, (L, q))
    allele_nums = np.array([int(i[:-2]) for i in alleles])
    allele_nums = allele_nums[::5]
    ref_seq, ref_tag  = get_MSA(REF_TAG +'.fasta')
    ref_seq  = list(ref_seq[0])
    ref_poly = np.array(ref_seq)[allele_nums]
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
    
    # Flatten the arrays 
    selection         = np.array(selection).flatten()
    selection_nocovar = np.array(selection_nocovar).flatten()
    error_bars        = np.array(error_bars).flatten()
    
    f = open(out_str+'-unmasked.npz', mode='w+b')
    np.savez_compressed(f, error_bars=error_bars, selection=selection, allele_number=alleles,
                            selection_independent=selection_nocovar, inflow_tot=inflow_full)
    f.close()
    
    print(len(counts))
    print(len(types))
    print(len(alleles))
    # Eliminating the remaining synonymous mutations and mutations that don't appear in the data
    ### THERE APPEARS TO BE AN ISSUE WITH THIS PART ###
    mask      = np.nonzero(np.logical_and(types=='NS', counts>cutoff))[0]
    alleles   = alleles[mask]
    selection = selection[mask]
    s_ind     = selection_nocovar[mask]
    errors    = error_bars[mask]
   
        
    # save the solution  
    for i in range(L * q):
        A[i,i] -= g1
    g = open(out_str+'.npz', mode='w+b') 
    if not mask_site:
        #np.savez_compressed(g, error_bars=error_bars, selection=selection, traj=traj_full, 
        #                    mutant_sites=mutant_sites_tot, allele_number=allele_number, locations=locations, 
        #                    times=dates, covar_tot=covar_full, selection_independent=selection_nocovar, 
        #                    covar_int=A, inflow_tot=inflow_full, times_full=dates_full, delta_tot=RHS_full)
        np.savez_compressed(g, error_bars=errors, selection=selection, allele_number=alleles,
                            selection_independent=s_ind, inflow_tot=inflow_full)
    else:
        np.savez_compressed(g, selection=selection, allele_number=allele_number)
    g.close()
    
    

    
    
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    
    
    

