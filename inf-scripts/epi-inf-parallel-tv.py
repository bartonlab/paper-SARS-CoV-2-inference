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
    parser.add_argument('-R',            type=float,  default=0,                       help='the basic reproduction number')
    parser.add_argument('-k',            type=float,  default=0.1,                     help='parameter determining shape of distribution of new infected')
    parser.add_argument('-q',            type=int,    default=5,                     help='number of mutant alleles per site')
    parser.add_argument('--mu',          type=float,  default=1e-4,                    help='the mutation rate')
    parser.add_argument('--data',        type=str,    default=None,                    help='directory to .npz files containing the counts, sequences, times, and mutant_sites for different locations')
    parser.add_argument('--g1',          type=float,  default=1,                       help='regularization restricting the magnitude of the selection coefficients')
    parser.add_argument('--freq_cutoff', type=float,  default=0,                       help='if a mutant frequency never rises above this number, it will not be used for inference')
    parser.add_argument('--record',      type=int,    default=1,                       help='number of generations between samples')
    parser.add_argument('--timed',       type=int,    default=0,                       help='if 0, wont print any time information, if 1 will print some information')
    parser.add_argument('--pop_size',    type=int,    default=0,                       help='the population size')
    parser.add_argument('--ss_pop_size', type=str,    default=None,                    help='.npy file containing the simulation specific population size')
    parser.add_argument('--ss_k',        type=str,    default=None,                    help='.npy file containing the simulation specific k')
    parser.add_argument('--ss_R',        type=str,    default=None,                    help='.npy file containing the simulation specific R')
    parser.add_argument('--ss_record',   type=str,    default=None,                    help='.npy file containing the simulation specific record')
    parser.add_argument('--outflow',     type=str,    default=None,                    help='.npz file containing the counts and sequences that outflow')
    parser.add_argument('--inflow',      type=str,    default=None,                    help='.npz file containing the counts and sequences that inflow, and the locations')
    #parser.add_argument('--link_tol',    type=float,  default=1,                       help='the tolerance for correlation check used for determining linked sites')
    parser.add_argument('--end_cutoff',  type=int,    default=0,                       help='the number of days at the end of the simulations to ignore due to bad sampling')
    parser.add_argument('--start_cutoff',type=int,    default=0,                       help='the number of days at the end of the simulations to ignore due to bad sampling')
    parser.add_argument('--window',      type=int,    default=15,                      help='the number of days over which to take the moving average')
    parser.add_argument('--clip_end',    type=int,    default=29700,                   help='the last site to clip the genomes to')
    parser.add_argument('--clip_start',  type=int,    default=150,                     help='the first site to clip the genomes to')
    parser.add_argument('--delta_t',     type=int,    default=0,                       help='the amount of dates at the beginning and the end of the time series to use to calculate delta_x')
    parser.add_argument('--mut_type',    type=str,    default=None,                    help='.npz file containing the location along the genome of a mutation under kw locations, and the type of mutation under kw types')
    parser.add_argument('--c_directory', type=str,    default='Archive-tv-int',        help='directory containing the c++ scripts')
    parser.add_argument('--mask_site',   type=int,    default=None,                    help='the site to mask (change to WT) when inferring the other coefficients, in order to check effect of linkage (write)')
    parser.add_argument('--mask_group',  type=str,    default=None,                    help='the .npy file containing a list of sites (as nucleotide numbers) to mask for the inference')
    parser.add_argument('--remove_sites',type=str,    default=None,                    help='the sites to eliminate when inferring coefficients')
    parser.add_argument('--final_t',     type=int,    default=None,                    help='the last time point to be considered, cutoff all time series after this time (in days after 01-01-2020)')
    parser.add_argument('--decay_rate',  type=float,  default=0,                       help='the exponential decay rate used to correct for infection lasting multiple generations')
    parser.add_argument('--nm_popsize',  type=str,    default=None,                    help='.csv file containing the population sizes used to correct for infection lasting multiple generations')
    parser.add_argument('--delay',       type=int,    default=None,                    help='the delay between the newly infected individuals and the individuals that ')
    parser.add_argument('--start_time',  type=int,    default=0,                       help='the time relative to january 2021 at which to start the inference')
    parser.add_argument('--end_time',    type=int,    default=100000,                  help='the time relative to january 2021 at which to stop the inference')
    parser.add_argument('--mutation_on',    action='store_true', default=False,  help='whether or not to include mutational term in inference')
    parser.add_argument('--cutoff_on',      action='store_true', default=False,  help='whether or not to cutoff frequencies below freq_cutoff')
    parser.add_argument('--find_linked',    action='store_true', default=False,  help='whether or not to find the sites that are (almost) fully linked')
    parser.add_argument('--tv_inference',   action='store_true', default=False,  help='if true then the coefficients are inferred at every time point')
    parser.add_argument('--no_traj',        action='store_true', default=False,  help='if true then the trajectories will not be saved')
    
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
    clip_start    = arg_list.clip_start
    mask_site     = arg_list.mask_site
    final_time    = arg_list.final_t
    decay_rate    = arg_list.decay_rate
    delay         = arg_list.delay
    
    covar_dir   = os.path.join(directory_str, 'tv_covar')
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
    
    if arg_list.start_time!=0:
        status_name = f'tv-inference-status-{arg_list.start_time}.csv'
    else:
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
            data      = np.load(filepath, allow_pickle=True)  # Genome sequence data
                
            mutant_sites_full.append(np.array(data['allele_number']))
            dates.append(data['times'])
            dates_full.append(data['times_full'])
            locations.append(location)
            filepaths.append(filepath)
            k_full.append(data['k'])
            R_full.append(data['R'])
            N_full.append(data['N'])
            RHS_full.append(data['RHS'])
            inflow_full.append(data['inflow'])
            counts_full.append(list(data['counts']))
            #counts_full.append(freqs_from_counts(data['counts'], window=delta_t))
            if not arg_list.no_traj:
                if 'traj' in data:
                    traj_full.append(data['traj'])
            
            region_temp = location.split('-')
            if region_temp[3].find('2')==-1:
                region = '-'.join(region_temp[:4])
            else:
                region = '-'.join(region_temp[:3])
            regions_full.append(region)

    
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
        
    print2(regions_full)
    print2(data_after)
    print2(data_before)
    print2(dates)
    
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
    
    allele_number    = mutant_sites_all
    mutant_sites_tot = mutant_sites_full
    print2("number of inferred coefficients is {mutants} out of {mutants_all} total".format(mutants=len(allele_number), mutants_all=len(mutant_sites_all)))
    
    allele_new = []
    for i in range(len(allele_number)):
        for j in range(q):
            allele_new.append(str(allele_number[i]) + '-' + NUC[j])

    #new_mut_types = np.zeros(1)
    
    L   = len(allele_number)
    g1 *= np.mean(N_full) * np.mean(k_full) / (1 + (np.mean(k_full) / np.mean(R_full)))  # The regularization
    
    coefficient = (1 / ((1 / (np.mean(N_full) * np.mean(k_full))) + 
                        ((np.mean(k_full) / np.mean(R_full)) / (np.mean(N_full) * np.mean(k_full) - 1))))
    
    print2('number of sites:', L)
   
    if timed > 0:
        t_solve_old = timer()
    
    ref_seq, ref_tag  = get_MSA(REF_TAG +'.fasta')
    ref_seq           = list(ref_seq[0])
    ref_poly          = np.array(ref_seq)[allele_number]
    selection         = []
    selection_nocovar = []
    error_bars        = []
    t_infer           = []
    dates_covar = np.unique([dates_full[i][j] for i in range(len(dates_full)) for j in range(len(dates_full[i]))])
    dates_covar = np.arange(np.amin(dates_covar), np.amax(dates_covar)+1)
    
    # arrays that give the position of the mutations in each region in the list of all mutations
    alleles_sorted = np.argsort(allele_number)
    positions_all  = [np.searchsorted(allele_number[alleles_sorted], i) for i in mutant_sites_tot]
    
    covar_out_dir  = out_str + 'tv-covar'
    if not os.path.exists(covar_out_dir):
        os.mkdir(covar_out_dir)
    
    # Inference calculation at each time
    for t in range(len(dates_covar)):
        A_t  = np.zeros((L * q, L * q))
        b_t  = np.zeros(L * q)
        date = dates_covar[t]
        print2(f'time = {date}')
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
                continue
            #RHS          = RHS_full[sim]
            region       = regions_full[sim]
            counts       = counts_full[sim]
            positions    = positions_all[sim]
            positions    = alleles_sorted[positions]
            
            # right-hand side
            RHS_zero = False
            print2(f'\tregion is {locations[sim]}')
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
            
            # covariance
            if date in times_full:
                print2(f'{locations[sim]}___{date}.npz')
                covar_file     = os.path.join(covar_dir, f'{locations[sim]}___{date}.npz')
                if not os.path.exists(covar_file):
                    print2(f'{date} in times file but corresponding covariance not found')
                    covar_file = os.path.join(covar_dir, f'{locations[sim]}___{last_covar_times[sim]}.npz')
                covar          = np.load(covar_file, allow_pickle=True)['covar']
            elif date > times_full[-1]:
                print2(f'{locations[sim]}___{last_covar_times[sim]}.npz')
                covar_file     = os.path.join(covar_dir, f'{locations[sim]}___{last_covar_times[sim]}.npz')
                covar          = np.load(covar_file, allow_pickle=True)['covar']
                print2('shapes:', np.shape(covar), len(mutant_sites) * q)
                
            
            if timed > 1:
                sim_load = timer()
                print2(f'loading the data for region {locations[sim]} took {sim_load - sim_start} seconds')
                
            #RHS_zero = False
            #if len(np.nonzero(RHS)[0])==0:
            #    RHS_zero = True
                
            for i in range(len(mutant_sites)):
                #b_t[positions[i] * q : (positions[i] + 1) * q] += RHS[date_idx, i * q : (i + 1) * q]
                if not RHS_zero:
                    b_t[positions[i] * q : (positions[i] + 1) * q] += RHS[i * q : (i + 1) * q]
                A_t[positions[i] * q : (positions[i] + 1) * q, positions[i] * q : (positions[i] + 1) * q] += covar[i * q : (i + 1) * q, i * q : (i + 1) * q]
                for k in range(i + 1, len(mutant_sites)):
                    A_t[positions[i] * q : (positions[i] + 1) * q, positions[k] * q : (positions[k] + 1) * q] += covar[i * q : (i + 1) * q, k * q : (k + 1) * q]
                    A_t[positions[k] * q : (positions[k] + 1) * q, positions[i] * q : (positions[i] + 1) * q] += covar[k * q : (k + 1) * q, i * q : (i + 1) * q]
                    
            if timed > 1:
                sim_add_arrays = timer()
                print2(f'adding the RHS and the covariance in region {locations[sim]} took {sim_add_arrays - sim_load} seconds')
        
        # save covariance and RHS at this time
        A_t *= coefficient / 5
        b_t *= coefficient 
        #np.savez_compressed(os.path.join(covar_out_dir, f'covar---{date}.npz'), covar=A_t, RHS=b_t)
        
        if timed > 1:
            save_covar_time = timer()
            print2(f'saving the big covariance matrix at time {date} took {save_covar_time - sim_add_arrays} seconds')
        
        # skip loop if there is no frequency change data
        if np.all(b_t==np.zeros(L * q)):
            continue
        
        # regularize
        for i in range(L * q):
            A_t[i, i] += g1
        selection_temp         = linalg.solve(A_t, b_t, assume_a='sym')
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
        
        np.savez_compressed(out_str + f'---{date}.npz', selection=s_new, selection_independent=s_SL, error_bars=error_bars_temp, allele_number=allele_new)
        #np.savez_compressed(os.path.join(out_str, f'covar---{date}.npz'), covar=A_t, RHS=b_t)
    
        if timed > 0:
            t_solve_new = timer()
            print2(f"calucluating the selection coefficients for time {date} out of {len(dates_covar)}", t_solve_new - t_solve_old)
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
    

    
    
    
    
    
    
    

