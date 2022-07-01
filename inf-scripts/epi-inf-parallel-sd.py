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
    
def get_codon_start_index(i, d=5):
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
    parser.add_argument('--mask_site',   type=int,    default=None,                    help='the site to mask (change to WT) when inferring the other coefficients, in order to check effect of linkage (write)')
    parser.add_argument('--mask_group',  type=str,    default=None,                    help='the .npy file containing a list of sites (as nucleotide numbers) to mask for the inference')
    parser.add_argument('--remove_sites',type=str,    default=None,                    help='the sites to eliminate when inferring coefficients')
    parser.add_argument('--decay_rate',  type=float,  default=0,                       help='the exponential decay rate used to correct for infection lasting multiple generations')
    parser.add_argument('--nm_popsize',  type=str,    default=None,                    help='.csv file containing the population sizes used to correct for infection lasting multiple generations')
    parser.add_argument('--delay',       type=int,    default=None,                    help='the delay between the newly infected individuals and the individuals that ')
    parser.add_argument('--start_time',  type=int,    default=0,                       help='the time relative to january 2021 at which to start the inference')
    parser.add_argument('--end_time',    type=int,    default=1000000,                 help='the time relative to january 2021 at which to stop the inference')
    parser.add_argument('--refFile',     type=str,    default='ref-index.csv',         help='file containing the reference sequence indices and nucleotides')

    arg_list = parser.parse_args(args)
    
    out_str       = arg_list.o
    g1            = arg_list.g1
    q             = arg_list.q
    timed         = arg_list.timed
    directory_str = arg_list.data
    mask_site     = arg_list.mask_site
    decay_rate    = arg_list.decay_rate
    delay         = arg_list.delay
    
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
        
    if arg_list.remove_sites:
        remove_sites = np.load(arg_list.remove_sites)
    
    if arg_list.start_time!=0:
        status_name = f'sd-inference-status-{arg_list.start_time}.csv'
    else:
        status_name = f'sd-inference-status.csv'
        
    covar_out_dir  = out_str + 'tv-covar'
    if not os.path.exists(covar_out_dir):
        os.mkdir(covar_out_dir)
        
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
    ref_sites        = []
    mutant_sites_tot = []
    dates            = []
    filenames        = []
    k_full           = []
    R_full           = []
    N_full           = []
    locations        = []
    regions_full     = []
    directory        = directory_str
    for file in sorted(os.listdir(directory)):
        # Load data if it is not a directory
        filename  = file
        filepath  = os.path.join(directory_str, filename)
        if os.path.isdir(filepath):
            dates.append(int(file))
            continue
            
        location  = filename[:-4]
        region_temp = location.split('-')
        if region_temp[3].find('2')==-1:
            region = '-'.join(region_temp[:4])
        else:
            region = '-'.join(region_temp[:3])
        if region[-1]=='-':
            region = region[:-1]
            
        print(f'\tloading location {location}')
        data = np.load(filepath, allow_pickle=True)  # Genome sequence data
            
        ref_sites.append([str(i) for i in data['ref_sites']])    
        mutant_sites_tot.append(np.array(data['allele_number']))
        locations.append(location)
        filenames.append(file)
        k_full.append(data['k'])
        R_full.append(data['R'])
        N_full.append(data['N'])
        regions_full.append(region)
            
    print2('outfile is', out_str)
    print2('data loaded')
    
    # Finds all submission times
    unique_times = np.unique(dates)
    #max_times    = [np.amax(i) for i in dates]
    #min_times    = [np.amin(i) for i in dates]
    #dates_full   = np.arange(np.amin(min_times), np.amax(max_times) + 1)
    min_time     = np.amin(dates)
    max_time     = np.amax(dates)
    dates_full   = np.arange(min_time, max_time+1)
    #print(dates)
    #print(dates_full)
    coefficient  = np.mean(pop_size) * np.mean(k) * np.mean(R) / (np.mean(R) + np.mean(k))
    g1 *= coefficient  # The regularization
    
    # Find final times for each location file 
    """
    final_times = []    # the last submission date for which a region has data
    for file in filenames:
        time = dates_full[0]
        i = 0
        while os.path.exists(os.path.join(directory_str, str(time), file)):
            i += 1
            time = dates_full[i]
        final_times.append(time - 1)     
    """
    final_times = []
    for file in filenames:
        times_file = np.array([t for t in dates_full if os.path.exists(os.path.join(directory_str, str(t), file))])
        final_times.append(times_file[-1])
    print2('final times:', final_times)
        
        
    # loading reference sequence index and labeling sites
    mutant_sites_full = mutant_sites_tot
    ref_index = pd.read_csv(arg_list.refFile)
    index     = list(ref_index['ref_index'])
    index     = [str(i) for i in index]
    ref_full  = list(ref_index['nucleotide'])
    mutant_sites_all = np.sort(np.unique(np.array([mutant_sites_full[i][j] for i in range(len(mutant_sites_full)) for j in range(len(mutant_sites_full[i]))])))
    mutant_sites_tot = ref_sites
    alleles_temp  = list(np.unique([ref_sites[i][j] for i in range(len(ref_sites)) for j in range(len(ref_sites[i]))]))
    allele_number = []
    ref_poly      = []
    for i in range(len(index)):
        if index[i] in alleles_temp:
            allele_number.append(index[i])
            ref_poly.append(ref_full[i])
            
    allele_number = np.array(allele_number)
    L = len(allele_number)
    print2('number of sites:', L)
    
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
    
    # making arrays to save inference data
    selection         = []
    selection_nocovar = []
    #error_bars        = []
    t_infer           = []
    alleles_sorted = np.argsort(allele_number)
    positions_all  = [np.searchsorted(allele_number[alleles_sorted], i) for i in mutant_sites_tot]    # indices of the regional mutations in the list of all mutations
    positions_all  = [alleles_sorted[i] for i in positions_all]
    
    if timed > 0: t_solve_old = timer()
    
    # Inference calculation at each time
    t_infer = []
    for t in range(len(dates_full)):
        A_t  = np.zeros((L * q, L * q))
        b_t  = np.zeros(L * q)
        date = dates_full[t]
        if date < arg_list.start_time or date > arg_list.end_time:
            continue
            
        for i in range(len(filenames)):
            if timed > 1:
                sim_start = timer()
                
            # get data for the specific location
            t_load = min(date, final_times[i])
            print2(os.path.join(directory, str(t_load), filenames[i]))
            if (not os.path.exists(os.path.join(directory, str(t_load), filenames[i]))) and date<final_times[i]: 
                continue
            data   = np.load(os.path.join(directory, str(t_load), filenames[i]), allow_pickle=True)
            counts = data['counts']
            RHS    = np.array(counts[-1] - counts[0])
            covar  = data['covar']
            mutant_sites = data['allele_number']
            positions = positions_all[i]
            print(np.shape(RHS))
            print(np.shape(counts))
            
            if timed > 1:
                sim_load = timer()
                print2(f'loading the data for region {locations[sim]} took {sim_load - sim_start} seconds')
                
            for j in range(len(mutant_sites)):
                b_t[positions[j] * q : (positions[j] + 1) * q] += RHS[j * q : (j + 1) * q]
                A_t[positions[j] * q : (positions[j] + 1) * q, positions[j] * q : (positions[j] + 1) * q] += covar[j * q : (j + 1) * q, j * q : (j + 1) * q]
                for k in range(j + 1, len(mutant_sites)):
                    A_t[positions[j] * q : (positions[j] + 1) * q, positions[k] * q : (positions[k] + 1) * q] += covar[j * q : (j + 1) * q, k * q : (k + 1) * q]
                    A_t[positions[k] * q : (positions[k] + 1) * q, positions[j] * q : (positions[j] + 1) * q] += covar[k * q : (k + 1) * q, j * q : (j + 1) * q]
                    
            if timed > 1:
                sim_add_arrays = timer()
                print2(f'adding the RHS and the covariance in region {locations[sim]} took {sim_add_arrays - sim_load} seconds')
        
        # save covariance and RHS at this time
        b_t *= coefficient 
        #np.savez_compressed(os.path.join(covar_out_dir, f'covar---{date}.npz'), covar=A_t, RHS=b_t)
        print(f'max b = {np.amax(b_t)}')
        print(f'max covar = {np.amax(A_t)}')
        print(f'mean b = {np.mean(b_t)}')
        print(f'mean covar = {np.mean(A_t)}')
        
        if timed > 1:
            save_covar_time = timer()
            print2(f'saving the big covariance matrix at time {date} took {save_covar_time - sim_add_arrays} seconds')
        
        # regularize
        for i in range(L * q):
            A_t[i, i] += g1
        selection_temp = linalg.solve(A_t, b_t, assume_a='sym')
        
        if timed > 1:
            system_solve_time  = timer()
            print2(f'solving the system of equations took {system_solve_time - save_covar_time} seconds')
        
        #error_bars_temp        = np.sqrt(np.absolute(np.diag(np.linalg.inv(A_t))))
        #error_bars_temp        = 1 / np.sqrt(np.absolute(np.diag(A_t)))
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
        #error_bars.append(np.array(error_bars_temp))
        t_infer.append(date)
        s_new = s_new.flatten()
        s_SL  = s_SL.flatten()
        t_infer.append(date)
        
        if timed > 1:
            normalization_time = timer()
            print2(f'normalizing the selection coefficients took {normalization_time - system_solve_time} seconds')
        
        print2(out_str + f'---{date}.npz')
        np.savez_compressed(out_str + f'---{date}.npz', selection=s_new, selection_independent=s_SL, allele_number=allele_new)
    
        if timed > 0:
            t_solve_new = timer()
            print2(f"calucluating the selection coefficients for time {date} out of {dates_full[-1]}", t_solve_new - t_solve_old)
            t_solve_old = timer()
        
    # save the solution  
    if out_str[-4:]=='.npz':
        out_str = out_str[:-4]
    g = open(out_str+'.npz', mode='w+b') 
    np.savez_compressed(
        g, 
        selection=selection,
        mutant_sites=mutant_sites_new, 
        allele_number=allele_new, 
        locations=locations, 
        times=dates_full, 
        selection_independent=selection_nocovar
    )
    g.close()
    
    

    
    
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    
    
    

