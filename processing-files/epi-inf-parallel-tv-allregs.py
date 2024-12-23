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
from scipy import linalg

REF_TAG = 'EPI_ISL_402125'
NUC     = ['-', 'A', 'C', 'G', 'T']


def main(args):
    """Infer selection coefficients from individual files containing the covariance, trajectory, and single site frequencies for different regions"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',            type=str,    default='inferred-coefficients', help='output file')
    parser.add_argument('-R',            type=float,  default=2,                       help='the basic reproduction number')
    parser.add_argument('-k',            type=float,  default=0.1,                     help='parameter determining shape of distribution of new infected')
    parser.add_argument('-q',            type=int,    default=5,                       help='number of mutant alleles per site')
    parser.add_argument('--data',        type=str,    default=None,                    help='directory to .npz files containing the counts, sequences, times, and mutant_sites for different locations')
    parser.add_argument('--g1',          type=float,  default=1,                       help='regularization restricting the magnitude of the selection coefficients')
    parser.add_argument('--record',      type=int,    default=1,                       help='number of generations between samples')
    parser.add_argument('--timed',       type=int,    default=0,                       help='if 0, wont print any time information, if 1 will print some information')
    parser.add_argument('--pop_size',    type=int,    default=10000,                   help='the population size')
    parser.add_argument('--ss_pop_size', type=str,    default=None,                    help='.npy file containing the simulation specific population size')
    parser.add_argument('--ss_k',        type=str,    default=None,                    help='.npy file containing the simulation specific k')
    parser.add_argument('--ss_R',        type=str,    default=None,                    help='.npy file containing the simulation specific R')
    parser.add_argument('--ss_record',   type=str,    default=None,                    help='.npy file containing the simulation specific record')
    parser.add_argument('--mask_site',   type=int,    default=None,                    help='the site to mask (change to WT) when inferring the other coefficients, in order to check effect of linkage (write)')
    parser.add_argument('--mask_group',  type=str,    default=None,                    help='the .npy file containing a list of sites (as nucleotide numbers) to mask for the inference')
    parser.add_argument('--decay_rate',  type=float,  default=0,                       help='the exponential decay rate used to correct for infection lasting multiple generations')
    parser.add_argument('--nm_popsize',  type=str,    default=None,                    help='.csv file containing the population sizes used to correct for infection lasting multiple generations')
    parser.add_argument('--delay',       type=int,    default=None,                    help='the delay between the newly infected individuals and the individuals that ')
    parser.add_argument('--infDate',     type=int,    default=None,                    help='if specified, inference is performed only on the specific date given (in days after 1/1/2020)')
    parser.add_argument('--refFile',     type=str,    default=None,                    help='file containing the reference index')
    
    arg_list = parser.parse_args(args)
    
    out_str       = arg_list.o
    g1            = arg_list.g1
    q             = arg_list.q
    record        = arg_list.record
    window        = arg_list.window
    timed         = arg_list.timed
    directory_str = arg_list.data
    mask_site     = arg_list.mask_site
    decay_rate    = arg_list.decay_rate
    delay         = arg_list.delay
    infDate       = arg_list.infDate
    
    covar_dir   = os.path.join(directory_str, 'tv_covar')
    simulations = len([name for name in os.listdir(directory_str)]) # number of locations to be combined
    
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
    
    status_name = f'tv-inference-status.csv'
        
    print(os.getcwd())
    print(status_name)
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
    
    print2(f'outfile is {out_str}')
            
    if timed > 0:
        t_elimination = timer()
        print2("starting")
        
    if arg_list.infDate:
        filepaths   = []
        covar_paths = []
        for folder in sorted(os.listdir(directory_str)):
            temp_dir   = os.path.join(directory_str, folder)
            covar_temp = os.path.join(temp_dir, 'tv_covar')
            for file in sorted(os.listdir(temp_dir)):
                temp_file = os.path.join(temp_dir, file)
                if os.path.isfile(temp_file):
                    times_temp = np.load(temp_file, allow_pickle=True)['times_full']
                    if infDate in list(times_temp) or infDate > times_temp[-1]:
                        filepaths.append(temp_file)
                    if infDate in list(times_temp):
                        covar_paths.append(os.path.join(covar_temp, file[:-4] + f'___{infDate}.npz'))
                    elif infDate > times_temp[-1]:
                        covar_paths.append(os.path.join(covar_temp, file[:-4] + f'___{times_temp[-1]}.npz'))
            """
            for file in covar_temp:
                temp_file = os.path.join(covar_temp, file)
                if int(file[:-4].split('___')[-1])==infDate:
                    covar_paths.append(temp_file)
            """
    print(f'number of data files = {len(filepaths)}')
    print(f'number of covariance files = {len(covar_paths)}')
    
    # Loading and organizing the data from the different locations. Combining it into new arrays.
    scratch_dir = os.path.join(os.getcwd(), 'scratch')
    if not os.path.exists(scratch_dir):
        os.mkdir(scratch_dir)
    allele_number     = []
    ref_sites         = []
    mutant_sites_full = []
    dates             = []
    dates_full        = []
    k_full            = []
    R_full            = []
    N_full            = []
    locations         = []
    RHS_full          = []
    traj_full         = []
    counts_full       = []
    regions_full      = []
    inflow_full       = []    # The correction to the first moment due to migration into the population
    for file in filepaths:
        location  = os.path.split(file)[-1][:-4]
        print(f'\tloading location {location}')
        data = np.load(file, allow_pickle=True)  # Genome sequence data
            
        ref_sites.append(data['ref_sites'])    
        mutant_sites_full.append(np.array(data['allele_number']))
        full_dates_temp = data['times_full']
        dates_full.append(data['times_full'])
        locations.append(location)
        k_full.append(data['k'])
        R_full.append(data['R'])
        N_full.append(data['N'])
        counts_temp = data['counts']
        counts_full.append(counts_temp)
        dates_temp = data['times_full']
        dates_temp = np.arange(dates_temp[-1] - len(counts_temp) + 1, dates_temp[-1] + 1)
        dates.append(dates_temp)
            
        region_temp = location.split('-')
        if region_temp[3].find('2')==-1:
            region = '-'.join(region_temp[:4])
        else:
            region = '-'.join(region_temp[:3])
        if region[-1]=='-':
            region = region[:-1]
        regions_full.append(region)       
    
    # Finds all sites at which there are mutations
    mutant_sites_all = np.sort(np.unique(np.array([mutant_sites_full[i][j] for i in range(len(mutant_sites_full)) for j in range(len(mutant_sites_full[i]))])))
    all_times        = np.unique([dates[i][j] for i in range(len(dates)) for j in range(len(dates[i]))])
    max_times        = [np.amax(i) for i in dates]
    min_times        = [np.amin(i) for i in dates]
    
    allele_number    = np.array(mutant_sites_all)
    mutant_sites_tot = ref_sites
    print2("number of inferred coefficients is {mutants} out of {mutants_all} total".format(mutants=len(allele_number), mutants_all=len(mutant_sites_all)))
    
    allele_new = []
    for i in range(len(allele_number)):
        for j in range(q):
            allele_new.append(str(allele_number[i]) + '-' + NUC[j])

    #new_mut_types = np.zeros(1)
    
    L   = len(allele_number)
    g1 *= np.mean(pop_size) * np.mean(k) * np.mean(R) / (np.mean(R) + np.mean(k))  # The regularization
    
    coefficient = np.mean(pop_size) * np.mean(k) * np.mean(R) / (np.mean(R) + np.mean(k))
    
    print2('number of sites:', L)
   
    if timed > 0:
        t_solve_old = timer()
        
    mutant_sites_tot = mutant_sites_full
    alleles_temp  = list(np.unique([ref_sites[i][j] for i in range(len(ref_sites)) for j in range(len(ref_sites[i]))]))
    
    mutant_sites_tot = mutant_sites_full
    alleles_temp  = list(np.unique([str(ref_sites[i][j]) for i in range(len(ref_sites)) for j in range(len(ref_sites[i]))]))
    
    # loading reference sequence index
    ref_index = pd.read_csv(arg_list.refFile)
    index     = list(ref_index['ref_index'])
    index     = [str(i) for i in index]
    ref_full  = list(ref_index['nucleotide'])
    allele_number = []
    ref_poly      = []
    for i in range(len(index)):
        if index[i] in alleles_temp:
            allele_number.append(index[i])
            ref_poly.append(ref_full[i])
    allele_number = np.array(allele_number)
    allele_new    = []
    for i in range(len(allele_number)):
        for j in range(q):
            allele_new.append(str(allele_number[i]) + '-' + NUC[j])
    
    allele_number = np.array(allele_number)
    
    selection         = []
    selection_nocovar = []
    error_bars        = []
    t_infer           = []
    # arrays that give the position of the mutations in each region in the list of all mutations
    alleles_sorted = np.argsort(allele_number)
    positions_all  = [np.searchsorted(allele_number[alleles_sorted], i) for i in mutant_sites_tot]
    
    # Inference calculation at each time
    A_t  = np.zeros((L * q, L * q))
    b_t  = np.zeros(L * q)
    for sim in range(len(filepaths)):
        if timed > 1:
            sim_start = timer()
        # get data for the specific location
        print(filepaths[sim])
        times        = dates[sim]
        times_full   = dates_full[sim]
        mutant_sites = mutant_sites_tot[sim]

        #RHS          = RHS_full[sim]
        region       = regions_full[sim]
        counts       = counts_full[sim]
        positions    = positions_all[sim]
        positions    = alleles_sorted[positions]
        
        if infDate > times[-1]:
            date_idx = -1
        elif infDate <= times[0]:
            continue
        else:
            date_idx = list(times).index(infDate)
        RHS      = counts[date_idx] - counts[0]
        """
        elif date > times[-1]:
            RHS      = counts[-1] - counts[0]
        else:
            RHS_zero = True
            RHS      = np.zeros(len(mutant_sites) * q)
        """
            
        # covariance
        covar_file = covar_paths[sim]
        covar      = np.load(covar_file, allow_pickle=True)['covar']
        print(np.shape(RHS))
        print(np.shape(covar))
                                  
        if timed > 1:
            sim_load = timer()
            print2(f'loading the data for region {locations[sim]} took {sim_load - sim_start} seconds')
                
        for i in range(len(mutant_sites)):
            b_t[positions[i] * q : (positions[i] + 1) * q] += RHS[i * q : (i + 1) * q]
            #if not RHS_zero:
                #b_t[positions[i] * q : (positions[i] + 1) * q] += RHS[i * q : (i + 1) * q]
            A_t[positions[i] * q : (positions[i] + 1) * q, positions[i] * q : (positions[i] + 1) * q] += covar[i * q : (i + 1) * q, i * q : (i + 1) * q]
            for k in range(i + 1, len(mutant_sites)):
                A_t[positions[i] * q : (positions[i] + 1) * q, positions[k] * q : (positions[k] + 1) * q] += covar[i * q : (i + 1) * q, k * q : (k + 1) * q]
                A_t[positions[k] * q : (positions[k] + 1) * q, positions[i] * q : (positions[i] + 1) * q] += covar[k * q : (k + 1) * q, i * q : (i + 1) * q]
                    
        if timed > 1:
            sim_add_arrays = timer()
            print2(f'adding the RHS and the covariance in region {locations[sim]} took {sim_add_arrays - sim_load} seconds')
        
    # save covariance and RHS at this time
    #A_t *= coefficient / 5
    b_t *= coefficient 
        
    # regularize
    for i in range(L * q):
        A_t[i, i] += g1
    selection_temp = linalg.solve(A_t, b_t, assume_a='sym')
        
    if timed > 1:
        system_solve_time  = timer()
        print2(f'solving the system of equations took {system_solve_time - save_covar_time} seconds')
        
    error_bars_temp        = 1 / np.sqrt(np.absolute(np.diag(A_t)))
    selection_nocovar_temp = b_t / np.diag(A_t)
        
    # Normalize selection coefficients so reference allele has selection coefficient of zero
    selection_temp         = np.reshape(selection_temp, (L, q))
    selection_nocovar_temp = np.reshape(selection_nocovar_temp, (L, q))
    s_new = np.zeros((L, q))
    s_SL  = np.zeros((L, q))
    for i in range(L):
        if q==4 or q==5: idx = NUC.index(ref_poly[i])
        elif q==2:       idx = 0
        else: print2(f'number of states at each site is {q}, which cannot be handeled by the current version of the code')
        temp_s    = selection_temp[i]
        temp_s    = temp_s - temp_s[idx]
        temp_s_SL = selection_nocovar_temp[i]
        temp_s_SL = temp_s_SL - temp_s_SL[idx]
        s_new[i, :] = temp_s
        s_SL[i, :]  = temp_s_SL
    print2(f'normalizing selection coefficients completed')
    selection = s_new.flatten()
    selection_nocovar = s_SL.flatten()
        
    if timed > 1:
        normalization_time = timer()
        print2(f'normalizing the selection coefficients took {normalization_time - system_solve_time} seconds')
    
    
    # Change labels of sites to include the nucleotide mutation
    
    mutant_sites_new  = []
    allele_new        = []
    for i in range(len(allele_number)):
        for j in range(q):
            allele_new.append(str(allele_number[i]) + '-' + NUC[j])
    for i in range(len(mutant_sites_tot)):
        mut_temp = []
        for j in range(len(mutant_sites_tot[i])):
            for k in range(q):
                mut_temp.append(str(mutant_sites_tot[i][j]) + '-' + NUC[k])
        mutant_sites_new.append(mut_temp)
    allele_number    = allele_new
    mutant_sites_tot = mutant_sites_new
        
    # save the solution  
    if out_str[-4:]=='.npz':
        out_str = out_str[:-4]
        
    g = open(out_str+'.npz', mode='w+b') 
    np.savez_compressed(g, error_bars=error_bars, selection=selection, traj=traj_full, 
                        mutant_sites=mutant_sites_tot, allele_number=allele_number, locations=locations, 
                        times=dates, selection_independent=selection_nocovar, time_infer=t_infer,
                        inflow_tot=inflow_full, times_full=dates_full, delta_tot=RHS_full)
    g.close()
    
    

    
    
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    
    
    

