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
import data_processing as dp

#REF_TAG = 'EPI_ISL_402125'
NUC     = ['-', 'A', 'C', 'G', 'T']
NUMBERS = list('0123456789')


def main(args):
    """Infer selection coefficients from individual files containing the covariance, trajectory, and single site frequencies for different regions"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',        type=str,    default='inferred-coefficients', help='output file')
    parser.add_argument('-q',        type=int,    default=5,                       help='number of mutant alleles per site')
    parser.add_argument('--mu',      type=float,  default=1e-4,                    help='the mutation rate')
    parser.add_argument('--data',    type=str,    default=None,                    help='directory to .npz files containing the counts, sequences, times, and mutant_sites for different locations')
    parser.add_argument('--g1',      type=float,  default=40,                      help='regularization restricting the magnitude of the selection coefficients')
    parser.add_argument('--record',  type=int,    default=1,                       help='number of generations between samples')
    parser.add_argument('--timed',   type=int,    default=0,                       help='if 0, wont print any time information, if 1 will print some information')
    parser.add_argument('--regPCT',  type=float,  default=None,                    help='the percent of regions to randomly use for inference')
    parser.add_argument('--timePCT', type=float,  default=None,                    help='the percent of total time (summed over regions) to randomly use for inference')
    parser.add_argument('--refFile', type=str,    default='ref-index.csv',         help='the file containing the site indices and reference sequence')
    parser.add_argument('--trajectory',     action='store_true',  default=False,  help='whether or not to save the trajectories in the different regions')
    parser.add_argument('--eliminateNC',    action='store_true',  default=False,  help='whether or not to eliminate non-coding sites')
    parser.add_argument('--saveCovar',      action='store_true',  default=False,  help='if true, save the covariance matrix')
    
    arg_list = parser.parse_args(args)
    
    out_str       = arg_list.o
    g1            = arg_list.g1
    mu            = arg_list.mu
    q             = arg_list.q
    record        = arg_list.record
    timed         = arg_list.timed
    directory_str = arg_list.data
    simulations   = len([name for name in os.listdir(directory_str)]) # number of locations to be combined 
    
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

    if timed > 0:
        t_elimination = timer()
        print2("starting")
    
    # loading reference sequence index
    ref_index = pd.read_csv(arg_list.refFile)
    index     = list(ref_index['ref_index'])
    index     = [str(i) for i in index]
    ref_full  = list(ref_index['nucleotide'])
    
    # Loading and organizing the data from the different locations. Combining it into new arrays.
    mutant_sites_full = []
    ref_sites         = []
    dates             = []
    dates_full        = []
    filepaths         = []
    k_full            = []
    R_full            = []
    N_full            = []
    locations         = []
    regions_full      = []
    directory         = os.fsencode(directory_str)
    for file in sorted(os.listdir(directory)):
        
        # Load data
        filename = os.fsdecode(file)
        filepath = os.path.join(directory_str, filename)
        location = filename[:-4]
        print2(f'\tloading location {location}')
        data     = np.load(filepath, allow_pickle=True)  # Genome sequence data
        
        alleles = np.array(data['allele_number'])
        mutant_sites_full.append([str(i) for i in alleles])
        ref_sites.append([str(i) for i in data['ref_sites']])
        dates.append(data['times'])
        dates_full.append(data['times_full'])
        locations.append(location)
        filepaths.append(filepath)
        k_full.append(data['k'])
        R_full.append(data['R'])
        N_full.append(data['N'])
        #counts_full.append(data['counts'])
        
        region_temp = location.split('-')
        if region_temp[3].find('2')==-1:
            region = '-'.join(region_temp[:3])
        else:
            region = '-'.join(region_temp[:2])
        if region[-1]=='-':
            region = region[:-1]
            
        region_temp = region.split('-')
        while any([i in region_temp[-1] for i in NUMBERS]):
            region_temp = region_temp[:-1]
        region = '-'.join(region_temp)
        
        regions_full.append(region)
        print(region)
    
    # picking the regions to subsample randomly
    def mask_fun(x, idxs):
        """returns only the elements of x with index in idxs"""
        return [x[i] for i in idxs]
    
    if arg_list.regPCT or arg_list.timePCT:
 
        regs_short  = [dp.find_location(i) for i in regions_full]
        regs_unique = np.unique(regs_short)
        alleles     = mutant_sites_full
        rng         = np.random.default_rng()

        # subsample number of regions
        if arg_list.regPCT:
            reg_fraction = arg_list.regPCT / 100
        #regs_unique  = np.unique(regions_full)
        #num_regs    = int(len(locations) * pct)
        #reg_idxs    = np.random.choice(np.arange(len(regs_unique)), num_regs, replace=False)
            regs_chosen = regs_unique[np.random.choice(np.arange(len(regs_unique)), int(len(regs_unique) * reg_fraction), replace=False)]
            reg_idxs    = [i for i in range(len(regions_short)) if regions_short[i] in regs_chosen]
        # Subsample as a percent of total time in all regions
        elif arg_list.timePCT:
            time_to_use = (arg_list.timePCT / 100) * np.sum([len(i) for i in dates]) # the amount of time to include in the analysis
            total_time  = 0
            regs_chosen = []
            # add regions until the total time surpasses time_to_use
            while total_time <= time_to_use:
                regs_unused = [i for i in regs_unique if i not in regs_chosen]
                reg_idx     = rng.choice(len(regs_unused), 1)[0]
                reg         = regs_unused[reg_idx]
                regs_chosen.append(reg)
                for i in range(len(regs_short)):
                    if regs_short[i]==reg:
                        total_time += len(dates[i])
            reg_idxs = [i for i in range(len(regs_short)) if regs_short[i] in regs_chosen]

        
        def mask_func(x):
            return mask_fun(x, reg_idxs)
        
        # mask data based on regions
        #mask_func  = lambda x: list(np.array(x)[np.array(reg_idxs)])    # function for masking arrays
        alleles    = mask_func(alleles)
        ref_sites  = mask_func(ref_sites)
        dates      = mask_func(dates)
        dates_full = mask_func(dates_full)
        locations  = mask_func(locations)
        k_full     = mask_func(k_full)
        R_full     = mask_func(R_full)
        N_full     = mask_func(N_full)
        filepaths  = mask_func(filepaths)
        mutant_sites_full = mask_func(mutant_sites_full)
        
    """
    regions_unique      = np.unique(regions_full)
    regional_init_muts  = dict()
    regional_final_muts = dict()
     
    # determining which locations have been broken up into separate files but have continuous data
    data_after  = []
    data_before = []
    last, first = [], []
    for i in range(len(regions_full)):
        mask     = np.array(regions_full)==regions_full[i]
        reg_idxs = np.nonzero(np.array(regions_full)==regions_full[i])[0]    # new code starts here
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
        #t_diff        = 0     # the gap between the current time series and the one in the same region immediately follwing it
        #reg_after_idx = 0     # the index of the time-series in the same region that immediately follows this one
        for j in range(len(dates_reg)):
            if times[0] - dates_reg[j][-1] in [0, 1, -1, -2, -3, -4, -5]:
                before = True
            if dates_reg[j][0] - times[-1] in [0, 1, -1, -2, -3, -4, -5]:
                after  = True
                #t_diff        = dates[reg_idxs[j]][0] - dates[i][-1] - 1          # new code starts here
                #reg_after_idx = reg_idxs[j]
                #for k in range(t_diff):
                    #counts_full[i].append(counts_full[i][-1])                     
            #if t_diff != 0:
                #dates[i] = np.arange(dates[i][0], dates[reg_after_idx][0])        # new code ends here
        #data_after.append(after)
        #data_before.append(before)
        if after and before:
            last.append(True)
            first.append(True)
        elif after and not before:
            last.append(False)
            first.append(True)
            regional_init_muts[regions_full[i]]  = mutant_sites_full[i]
        elif before and not after:
            last.append(True)
            first.append(False)
            regional_final_muts[regions_full[i]] = mutant_sites_full[i]
        else:
            last.append(False)
            first.append(False)
    assert list(regional_init_muts)==list(regional_final_muts)
    """
    
    # Finds all sites at which there are mutations
    mutant_sites_all = np.sort(np.unique(np.array([mutant_sites_full[i][j] for i in range(len(mutant_sites_full)) for j in range(len(mutant_sites_full[i]))])))
    mutant_sites_tot = ref_sites
    alleles_temp  = list(np.unique([ref_sites[i][j] for i in range(len(ref_sites)) for j in range(len(ref_sites[i]))]))
    allele_number = []
    ref_poly      = []
    for i in range(len(index)):
        if index[i] in alleles_temp:
            allele_number.append(index[i])
            ref_poly.append(ref_full[i])
    
    L = len(allele_number)
    A = np.zeros((L * q, L * q))     # The matrix on the left hand side of the SOE used to infer selection
    b = np.zeros(L * q)         # The vector on the right hand side of the SOE used to infer selection
    co  = np.mean(N_full) * np.mean(k_full) * np.mean(R_full) / (np.mean(R_full) + np.mean(k_full))
    g1 *= co  # The regularization
    print2('number of sites:\t', L)
    
    """
    # TEST THIS
    # subtract a 1 in the RHS for each reference mutation that is present in the final data file for that region but not the initial
    # this is to correct the delta x term, since the frequency for the reference nucleotide was initially near 1 but is not in the data
    for reg in list(regional_final_muts):
        absent_muts = [i for i in regional_final_muts[reg] if i not in regional_init_muts[reg]]
        site_idxs   = [allele_number.index(i) for i in absent_muts]
        nuc_idxs    = [NUC.index(ref_poly[i]) for i in site_idxs]
        for i in range(len(site_idxs)):
            b[q * site_idxs[i] + nuc_idxs[i]] -= co
    """
   
    if timed > 0:
        tracker   = 0
        t_combine = 0
        t_add_old = timer()
        t_pre_combine = timer()
        print(f'loading the data for all regions took {t_add_old - t_elimination} seconds')
    inflow_full            = []    # The correction to the first moment due to migration into the population
    traj_nosmooth          = []    # The unsmoothed frequency trajectories
    single_freq_infectious = []
    single_newly_infected  = []
    delta_tot              = []
    covar_full             = []
    traj_full              = []
    times_full             = []
    for sim in range(len(filepaths)):
        # get data for the specific location
        data         = np.load(filepaths[sim], allow_pickle=True)  # Genome sequence data
        covar        = data['covar']
        counts       = data['counts'] * co
        inflow       = data['inflow']
        times        = data['times']
        mutant_sites = mutant_sites_tot[sim]
        
        if 'traj' in data and arg_list.trajectory:
            traj = data['traj']
        else:
            traj = []
            
        times_full.append(times)
        traj_full.append(traj)
        inflow_full.append(inflow)
        region = regions_full[sim]
        
        if timed > 0:
            t_traj = timer()
            if tracker == 0:
                print2("eliminating sites whose frequency is too low", t_traj - t_elimination)
            else:
                print2(f"calculating the integrated covariance and the delta x term for location {tracker-1}, {locations[tracker-1]}", t_traj - t_combine)
            t_combine = timer()
            tracker += 1
        
        # determine where in the list of all sites the regional sites are located
        alleles_sorted = np.argsort(allele_number)
        positions      = np.searchsorted(np.array(allele_number)[alleles_sorted], ref_sites[sim])
        positions      = alleles_sorted[positions]
        
        # Add the delta x term from this region
        """
        if not data_before[sim]:
            for i in range(len(mutant_sites)):
                b[positions[i] * q : (positions[i] + 1) * q] -= counts[0][i * q : (i + 1) * q]
        if not data_after[sim]:
            for i in range(len(mutant_sites)):
                b[positions[i] * q : (positions[i] + 1) * q] += counts[-1][i * q : (i + 1) * q]
        """
        
        """
        if first[sim]:
            for i in range(len(mutant_sites)):
                b[positions[i] * q : (positions[i] + 1) * q] -= counts[0][i * q : (i + 1) * q]
        elif last[sim]:
            for i in range(len(mutant_sites)):
                b[positions[i] * q : (positions[i] + 1) * q] += counts[-1][i * q : (i + 1) * q]
        else:
            for i in range(len(mutant_sites)):
                b[positions[i] * q : (positions[i] + 1) * q] += counts[-1][i * q : (i + 1) * q] - counts[0][i * q : (i + 1) * q]
        """
        for i in range(len(mutant_sites)):
            b[positions[i] * q : (positions[i] + 1) * q] += counts[-1][i * q : (i + 1) * q] - counts[0][i * q : (i + 1) * q]
        
        # Subtract migration term
        for i in range(len(mutant_sites)):
            b[positions[i] * q : (positions[i] + 1) * q] -= inflow[i * q : (i + 1) * q]
        
        # Add the integrated covariance from this region
        for i in range(len(mutant_sites)):
            A[positions[i] * q : (positions[i] + 1) * q, positions[i] * q : (positions[i] + 1) * q] += covar[i * q : (i + 1) * q, i * q : (i + 1) * q]
            for k in range(i + 1, len(mutant_sites)):
                A[positions[i] * q : (positions[i] + 1) * q, positions[k] * q : (positions[k] + 1) * q] += covar[i * q : (i + 1) * q, k * q : (k + 1) * q]
                A[positions[k] * q : (positions[k] + 1) * q, positions[i] * q : (positions[i] + 1) * q] += covar[k * q : (k + 1) * q, i * q : (i + 1) * q]
        
    if timed > 0:
        t_solve_system = timer()
        print2(f"calculating the integrated covariance and the delta x term for location {tracker-1}, {locations[tracker-1]}", t_solve_system - t_combine)
        print2(f'combining all of the covariances and delta x terms for all of the locations took {t_solve_system - t_pre_combine} seconds')
    
    if arg_list.eliminateNC:
        prots = [dp.get_label(i).split('-')[0] for i in allele_number]
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
    
    # Apply the regularization
    for i in range(L * q):
        A[i,i] += g1
    print2('regularization applied')
    if timed > 0:
        t_preerror = timer()
    error_bars = np.sqrt(np.absolute(np.diag(linalg.inv(A))))
    print2('error bars found')
    if timed > 0:
        t_posterror = timer()
        print(f'Took {t_posterror - t_preerror} seconds to calculate the error bars')
    if timed > 0:
        t_presolve = timer()
    selection      = linalg.solve(A, b, assume_a='sym')
    print2('selection coefficients found')
    if timed > 0:
        t_postsolve   = timer()
        print2(f'Took {t_postsolve - t_presolve} seconds to solve the system of equations')
    selection_nocovar = b / np.diag(A)
    
    allele_new = []
    for i in range(len(allele_number)):
        for j in range(q):
            allele_new.append(str(allele_number[i]) + '-' + NUC[j])
    
    #np.savez_compressed(out_str + '-unnormalized.npz', selection=selection, allele_number=allele_new)
    
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
        
    # remove the regularization from the covariance
    for i in range(L * q):
        A[i,i] -= g1
    if not arg_list.saveCovar:
        A = []
        
    # save the solution     
    g = open(out_str+'.npz', mode='w+b') 
    np.savez_compressed(
        g, 
        error_bars=error_bars, 
        selection=selection, 
        #traj=traj_full,   
        #mutant_sites=mutant_sites_tot, 
        allele_number=allele_number, 
        locations=locations, 
        #times=dates, 
        selection_independent=selection_nocovar, 
        covar_int=A, 
        #inflow_tot=inflow_full, 
        #times_full=dates_full,
        numerator=b
    )
    g.close()
        
    # write the covariance to a csv
    cov_out = open(out_str + '-covar.csv', 'w')
    header = ','.join(allele_number)
    cov_out.write(f'{header}\n')
    for line in A:
        for i in range(len(line)):
            cov_out.write(f'{line[i]:.6f}')
        cov_out.write('\n')
    cov_out.close()
    
    # write inferred selection, errors, and numerator to a csv
    data = {
        'sites'         : allele_number,
        'selection'     : selection,
        'error_bars'    : error_bars,
        'selection ind' : selection_nocovar,
        'numerator'     : b
    }
    df = pd.DataFrame.from_dict(data)
    df.to_csv(out_str + '.csv', index=False)

    
    
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    
    
    

