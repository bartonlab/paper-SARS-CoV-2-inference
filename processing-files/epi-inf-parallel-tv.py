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
    parser.add_argument('--refFile',     type=str,    default='ref-index.csv',         help='the file containing the reference sequence indices and nucleotides')
    
    arg_list = parser.parse_args(args)
    
    out_str       = arg_list.o
    g1            = arg_list.g1
    q             = arg_list.q
    timed         = arg_list.timed
    directory_str = arg_list.data
    mask_site     = arg_list.mask_site
    decay_rate    = arg_list.decay_rate
    delay         = arg_list.delay
    
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
    
    # Loading and organizing the data from the different locations. Combining it into new arrays.
    scratch_dir = os.path.join(os.getcwd(), 'scratch')
    if not os.path.exists(scratch_dir):
        os.mkdir(scratch_dir)
        
    allele_number     = []
    ref_sites         = []
    mutant_sites_full = []
    dates             = []
    dates_full        = []
    filepaths         = []
    k_full            = []
    R_full            = []
    N_full            = []
    locations         = []
    RHS_full          = []
    traj_full         = []
    counts_full       = []
    regions_full      = []
    inflow_full       = []    # The correction to the first moment due to migration into the population
    directory         = os.fsencode(directory_str)
    for file in sorted(os.listdir(directory)):
        
        # Load data
        filename  = os.fsdecode(file)
        filepath  = os.path.join(directory_str, filename)
        if filename.find('___') == -1 and os.path.isfile(filepath):
            location  = filename[:-4]
            print(f'\tloading location {location}')
            data = np.load(filepath, allow_pickle=True)  # Genome sequence data
            
            ref_sites.append(data['ref_sites'])    
            mutant_sites_full.append(np.array(data['allele_number']))
            dates_full.append(data['times_full'])
            locations.append(location)
            filepaths.append(filepath)
            k_full.append(data['k'])
            R_full.append(data['R'])
            N_full.append(data['N'])
            inflow_full.append(data['inflow'])
            counts_full.append(data['counts'])
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
    
    print(dates)
    print(dates_full)
            
    if out_str[-5:]=='north':
        out_str = out_str[:-5] + region
    if out_str[-1]=='-':
        out_str = out_str + region
    print2('outfile is', out_str)
    
    """
    data_after  = []
    data_before = []
    for i in range(len(regions_full)):
        mask      = np.array(regions_full)==regions_full[i]
        reg_idxs  = np.nonzero(np.array(regions_full)==regions_full[i])[0]    # new code starts here
        if len(reg_idxs)==1:
            data_after.append(False) 
            data_before.append(False)       
            continue                                                          # new code ends here
        dates_reg = np.array(dates_full)[mask]
        times     = dates_full[i]
        reg_idxs  = [reg_idxs[i] for i in range(len(reg_idxs)) if not np.all(dates_reg[i]==times)]    # new code line
        dates_reg = [i for i in dates_reg if not np.all(i==times)]
        after     = False
        before    = False
        t_diff        = 0     # the gap between the current time series and the one in the same region immediately follwing it
        reg_after_idx = 0     # the index of the time-series in the same region that immediately follows this one
        for j in range(len(dates_reg)):
            if times[0] - dates_reg[j][-1] in [0, 1, -1, -2, -3, -4, -5]:
                before = True
            if dates_reg[j][0] - times[-1] in [0, 1, -1, -2, -3, -4, -5]:
                after  = True
                t_diff        = dates[reg_idxs[j]][0] - dates[i][-1] - 1          # new code starts here
                reg_after_idx = reg_idxs[j]
                for k in range(t_diff):
                    counts_full[i].append(counts_full[i][-1])                     
            if t_diff != 0:
                dates[i] = np.arange(dates[i][0], dates[reg_after_idx][0])        # new code ends here
        data_after.append(after)
        data_before.append(before)
        
    #print2(regions_full)
    print2(data_after)
    print2(data_before)
    print2(dates)
    """
    
    print2('data loaded')
    #regions_cont      = []
    #continuous_full   = []
    #regions_unique    = np.unique(regions_full)
    #region_dates_full = []
    #for region in regions_unique:
        #mask       = regions_full==region
        #dates_reg  = np.array(dates_full)[mask]
        #region_dates_full.append(dates_reg)
        #min_dates  = [i[0] for i in dates_reg]
        #dates_sort = dates_reg[np.argsort(min_dates)]
        #continuous = []
        #for d in range(len(dates_sort)-1):
            #if dates_sort[d+1] - dates_sort[d] <= 1:
                #continuous.append(True)
            #else:
                #continuous.append(False)
        #continuous_full.append(all(continuous))
    #print(continuous_full)
    
    last_covar_times = []    # the last time-point for which there is a covariance matrix in each time series
    for sim in range(len(locations)):
        print(locations[sim])
        times    = dates[sim]
        final_t  = times[-1]
        cov_path = os.path.join(covar_dir, f'{locations[sim]}___{final_t}.npz')
        while not os.path.exists(cov_path):
            final_t -= 1
            cov_path = os.path.join(covar_dir, f'{locations[sim]}___{final_t}.npz')
            if final_t == 0:
                break
        last_covar_times.append(final_t)
            
    
    # Finds all sites at which there are mutations
    mutant_sites_all = np.sort(np.unique(np.array([mutant_sites_full[i][j] for i in range(len(mutant_sites_full)) for j in range(len(mutant_sites_full[i]))])))
    all_times        = np.unique([dates[i][j] for i in range(len(dates)) for j in range(len(dates[i]))])
    max_times        = [np.amax(i) for i in dates]
    min_times        = [np.amin(i) for i in dates]
    allele_number    = np.array(mutant_sites_all)
    mutant_sites_tot = ref_sites
    print2("number of inferred coefficients is {mutants} out of {mutants_all} total".format(mutants=len(allele_number), mutants_all=len(mutant_sites_all)))
    
    L   = len(allele_number)
    g1 *= np.mean(pop_size) * np.mean(k) * np.mean(R) / (np.mean(R) + np.mean(k))  # The regularization
    coefficient = np.mean(pop_size) * np.mean(k) * np.mean(R) / (np.mean(R) + np.mean(k))
    print2('number of sites:', L)
   
    if timed > 0:
        t_solve_old = timer()
        
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
    
    selection         = []
    selection_nocovar = []
    error_bars        = []
    t_infer           = []
    dates_covar = np.unique([dates_full[i][j] for i in range(len(dates_full)) for j in range(len(dates_full[i]))])
    dates_covar = np.arange(np.amin(dates_covar), np.amax(dates_covar)+1)
    
    # arrays that give the position of the mutations in each region in the list of all mutations
    alleles_sorted = np.argsort(allele_number)
    positions_all  = [np.searchsorted(allele_number[alleles_sorted], i) for i in mutant_sites_tot]
    positions_all  = [alleles_sorted[i] for i in positions_all]
    
    covar_out_dir  = out_str + 'tv-covar'
    if not os.path.exists(covar_out_dir):
        os.mkdir(covar_out_dir)
    
    # Inference calculation at each time
    for t in range(len(dates_covar)):
        A_t  = np.zeros((L * q, L * q))
        b_t  = np.zeros(L * q)
        date = dates_covar[t]
        #print2(f'time = {date}')
        if date < arg_list.start_time or date > arg_list.end_time:
            continue
        for sim in range(len(filepaths)):
            if timed > 1:
                sim_start = timer()
            # get data for the specific location
            times        = dates[sim]
            times_full   = dates_full[sim]
            mutant_sites = mutant_sites_tot[sim]
            if date < times_full[0]:
                #RHS = np.zeros(len(mutant_sites) * q)
                #print(f'skipping {date} due to lack of data')
                continue
            #RHS          = RHS_full[sim]
            region       = regions_full[sim]
            counts       = counts_full[sim] * coefficient
            positions    = positions_all[sim]
            
            # right-hand side
            RHS_zero = False
            #print2(f'\tregion is {locations[sim]}')
            """
            if date in times:
                date_idx = list(times).index(date)
                if data_before[sim]:
                    RHS  = counts[date_idx]
                else:
                    RHS  = counts[date_idx] - counts[0]
            elif date > times[-1]:
                if data_before[sim]:
                    if data_after[sim]:
                        RHS      = np.zeros(len(mutant_sites) * q)
                        RHS_zero = True
                    else:
                        RHS = counts[-1]
                elif data_after[sim]:
                    RHS = - counts[0]
                else:
                    RHS = counts[-1] - counts[0]
            else:
                RHS      = np.zeros(len(mutant_sites) * q)
                RHS_zero = True
            """
            if date in times:
                date_idx = list(times).index(date)
                RHS      = counts[date_idx] - counts[0]
            elif date > times[-1]:
                RHS      = counts[-1] - counts[0]
            else:
                RHS_zero = True
                RHS      = np.zeros(len(mutant_sites) * q)
            
           # if not RHS_zero:
           #     print('number of nonzero entries in the RHS is', len(np.nonzero(RHS)[0]))
            
            # covariance
            if date in times_full:
                #print2(f'{locations[sim]}___{date}.npz')
                covar_file     = os.path.join(covar_dir, f'{locations[sim]}___{date}.npz')
                if not os.path.exists(covar_file):
                    print2(f'{date} in times file but corresponding covariance not found')
                    covar_file = os.path.join(covar_dir, f'{locations[sim]}___{last_covar_times[sim]}.npz')
                    continue
                covar          = np.load(covar_file, allow_pickle=True)['covar']
            elif date > times_full[-1]:
                #print2(f'{locations[sim]}___{last_covar_times[sim]}.npz')
                covar_file     = os.path.join(covar_dir, f'{locations[sim]}___{last_covar_times[sim]}.npz')
                covar          = np.load(covar_file, allow_pickle=True)['covar']
            else:
                continue
            #print2('shapes:', np.shape(covar), len(mutant_sites) * q)
            
            if not np.all(np.isfinite(covar)):
                print(f'covariance for region {region} at time {date} has nonfinite values')
                continue
            if not np.all(np.isfinite(RHS)):
                print(f'RHS for region {region} at time {date} has nonfinite values')
                continue
                
            if np.count_nonzero(RHS)==0:
                print(f'RHS is zero at time {date} for region {locations[sim]}')
                continue
            
            if timed > 1:
                sim_load = timer()
                print2(f'loading the data for region {locations[sim]} took {sim_load - sim_start} seconds')
                
            #RHS_zero = False
            #if len(np.nonzero(RHS)[0])==0:
            #    RHS_zero = True
                
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
        
        
        if timed > 1:
            save_covar_time = timer()
            print2(f'saving the big covariance matrix at time {date} took {save_covar_time - sim_add_arrays} seconds')
        
        # skip loop if there is no frequency change data
        if np.all(b_t==np.zeros(L * q)):
            print(f'right hand side is zero for time {date}')
            continue
        
        # regularize
        for i in range(L * q):
            A_t[i, i] += g1
        selection_temp = linalg.solve(A_t, b_t, assume_a='sym')
        
        if timed > 1:
            system_solve_time  = timer()
            print2(f'solving the system of equations took {system_solve_time - save_covar_time} seconds')
        
        #error_bars_temp        = np.sqrt(np.absolute(np.diag(np.linalg.inv(A_t))))
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
        print2(f'normalizing selection coefficients for time {date} completed')
        selection.append(np.array(s_new).flatten())
        selection_nocovar.append(np.array(s_SL).flatten())
        error_bars.append(np.array(error_bars_temp))
        t_infer.append(date)
        s_new = s_new.flatten()
        s_SL  = s_SL.flatten()
        
        if timed > 1:
            normalization_time = timer()
            print2(f'normalizing the selection coefficients took {normalization_time - system_solve_time} seconds')
        
        print2(out_str + f'---{date}.npz')
        np.savez_compressed(out_str + f'---{date}.npz', selection=s_new, selection_independent=s_SL, error_bars=error_bars_temp, allele_number=allele_new)
        #np.savez_compressed(os.path.join(out_str, f'covar---{date}.npz'), covar=A_t, RHS=b_t)
    
        if timed > 0:
            t_solve_new = timer()
            print2(f"calucluating the selection coefficients for time {date} out of {dates_covar[-1]}", t_solve_new - t_solve_old)
            t_solve_old = timer()
    
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
    if not mask_site:
        np.savez_compressed(g, error_bars=error_bars, selection=selection, traj=traj_full, 
                            mutant_sites=mutant_sites_tot, allele_number=allele_number, locations=locations, 
                            times=dates, selection_independent=selection_nocovar, time_infer=t_infer,
                            inflow_tot=inflow_full, times_full=dates_full, delta_tot=RHS_full)
    else:
        np.savez_compressed(g, selection=selection, allele_number=allele_number)
    g.close()
    
    

    
    
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    
    
    

