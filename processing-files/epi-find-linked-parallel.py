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
import data_processing as dp
#cwd = os.getcwd()
#os.chdir('/Users/brianlee/SARS-CoV-2-Data/Processing-files')
#import data_processing as dp
#os.chdir(cwd)

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


def find_linked_pairs(single, double, alleles, tol=0.8, q=5, ref_poly=None, min_counts=10):
    """ Takes the single and double site counts and finds pairs of mutations that are linked."""
    
    L = len(single)
    linked_pairs = []
    #linked_pairs = set()
    for i in range(L):
        if single[i]<min_counts:
            continue
        # ignore nucleotide if it is the reference nucleotide at this site
        site_num1 = int(i / q)
        nuc_idx1  = i % q
        if q==5:
            if ref_poly[site_num1]==NUC[nuc_idx1]:
                continue
        elif q==2:
            if i%2==0:
                continue
            
        for j in range(i+1, L):
            if single[j]<min_counts:
                continue
            site_num2 = int(j / q)
            nuc_idx2  = j % q
            if q==5:
                if ref_poly[site_num2]==NUC[nuc_idx2]:
                    continue
            elif q==2:
                if j%2==0:
                    continue
            if (double[i,j]/single[i])>1 or (double[i,j]/single[j])>1:
                print(f'ISSUE: double counts greater than single counts for sites {alleles[i]} and {alleles[j]}')
            if (double[i,j]/single[i])>tol and (double[i,j]/single[j])>tol:
                linked_pairs.append([alleles[i], alleles[j]])
    return linked_pairs


def find_linked_pairs_region(single, double, alleles, linked_pairs, ref_poly=None, tol=0.8, freq_tol=0.01, q=5, min_counts=5):
    """ Takes the single and double site counts and finds pairs of mutations that are linked
    NEEDS TESTING. IS IT FASTER?"""
    
    L = len(single)
    for i in range(L):
        #if single[i]<min_counts:
        #    continue
        # ignore nucleotide if it is the reference nucleotide at this site
        site_num1 = int(i / q)
        nuc_idx1  = i % q
        if q==5:
            if ref_poly[site_num1]==NUC[nuc_idx1]:
                continue
        elif q==2:
            if i%2==0:
                continue
                
        for j in range(i+1, L):
            if single[j]<min_counts:
                continue
            site_num2 = int(j / q)
            nuc_idx2  = j % q
            if q==5:
                if ref_poly[site_num2]==NUC[nuc_idx2]:
                    continue
            elif q==2:
                if j%2==0:
                    continue
                    
            if double[i,j]/single[i]>tol and double[i][j]/single[j]>tol:
                if [alleles[i], alleles[j]] not in linked_pairs:
                    linked_pairs.append([alleles[i], alleles[j]])
                #if ([alleles[i], alleles[j]] not in linked_pairs) and ([alleles[j], alleles[i]] not in linked_pairs):
                    #linked_pairs.append([alleles[i], alleles[j]])
    return linked_pairs


def find_linked_groups(single, double, alleles, tol=0.8):
    """ Takes the single and double site counts and finds groups of linked sites. 
    A potential issue arises in skipping terms that that have already been added to a group. 
    This is because if a and b are linked and b and c are linked, a and c might not be linked.
    So if you skip b after finding that a and b are linked, c will not be added to the group.
    NEEDS TESTING"""
    
    L = len(single)
    linked_groups = []
    sites_used    = []
    for i in range(L):
        new = True
        if alleles[i] in sites_used:
            continue
        else:
            group = [alleles[i]]
            sites_used.append(alleles[i])
            for j in range(i+i, L):
                if double[i,j]/single[i]>tol and double[i][j]/single[j]>tol:
                    group.append(alleles[j])
                    sites_used.append(alleles[j])
            if len(group)>1:
                linked_groups.append(group)
    return linked_groups


def find_linked_groups_alt(single, double, alleles, tol=0.8):
    """ Takes the single and double site counts and finds groups of linked sites. 
    Attempts to fix the issue that the version above has.
    NEEDS TESTING"""
    
    L = len(single)
    linked_groups = []
    sites_used    = []
    for i in range(L):
        #print(f'finding sites linked to coefficient {i} out of {L} total sites.')
        new = True
        if alleles[i] in sites_used:
            idx   = np.nonzero([alleles[i] in j for j in linked_groups])[0][0]
            group = linked_groups[idx]
            new   = False
        else:
            group = [alleles[i]]
            #sites_used.append(alleles[i])
        for j in range(i+i, L):
            if double[i,j]/single[i]>tol and double[i][j]/single[j]>tol:
                if alleles[j] not in sites_used:
                    group.append(alleles[j])                    
                    sites_used.append(alleles[j])
                else:
                    idx2 = np.nonzero([alleles[j] in k for k in linked_groups])[0][0]
                    for k in linked_groups[idx2]:
                        group.append(k)
                    if new:
                        linked_groups[idx2] = group
                        new = False
                    else:
                        linked_groups.pop(idx2)
        if len(group)>1 and new:
            linked_groups.append(group)
    return linked_groups


def combine_linked_sites2(linked_sites):
    """ Combines the pairs of linked sites into lists of mutually linked sites.
    FUTURE: Try making a set of all sites linked to a site for each site. 
    For sets that have nonzero intersection, take their union (consider alternative approaches to this step)."""
    
        
    linked_full = list(np.unique(np.array([linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]))) # all sites linked to any other sites
    linked_new = []
    for site in linked_full:
        linked_specific = []   # list of all sites linked to this specific site
        for i in range(len(linked_sites)):
            if site in linked_sites[i]:
                loc = linked_sites[i].index(site)
                linked_specific.append(linked_sites[i][1-loc])
        new = True
        if len(linked_new) != 0:
            #all_new = [linked_new[i][j] for i in range(len(linked_new)) for j in range(len(linked_new[i]))]
            all_new = set([linked_new[i][j] for i in range(len(linked_new)) for j in range(len(linked_new[i]))])
            if site in all_new:
                new = False
                counter = 0
                for i in range(len(linked_new)):
                    if site in linked_new[i]:
                        #new = False
                        break
                    else: 
                        counter += 1
                new_series = linked_new[counter]                
            elif any(linked_specific[i] in all_new for i in range(len(linked_specific))):
                """
                for i in range(len(linked_specific)):
                    if linked_specific[i] in all_new:
                        counter = 0
                        for j in range(len(linked_new)):
                            if linked_specific[i] in linked_new[j]:
                                new = False
                                break
                            else:
                                counter += 1
                new_series = linked_new[counter]
                """
                new = False
                indices_present = []
                for i in range(len(linked_specific)):
                    if linked_specific[i] in all_new:
                        indices_present.append(i)
                counter = 0
                for j in range(len(linked_new)):
                    if linked_specific[indices_present[0]] in linked_new[j]:
                        break
                    else:
                        counter += 1
                new_series = linked_new[counter]
            else:
                new_series = []
        else:
            new_series = []
        if site not in new_series:
            new_series.append(site)
        for site_linked in linked_specific:
            if site_linked not in new_series:
                new_series.append(site_linked)
        if new:
            linked_new.append(new_series)
    return linked_new
    
    
def find_frequencies_linked(linked_sites, single, double, alleles):
    """ Find and return the single and the double site frequencies for all sites in each group of linked sites"""
    
    nucs        = [dp.get_label_orf(i) for i in alleles]
    single_freq = []
    double_freq = []
    new_sites   = []
    for group in linked_sites:
        single_temp = []
        double_temp = []
        sites_temp  = []
        for i in range(len(group)):
            if group[i] not in nucs:
                print(group[i])
            else:
                single_temp.append(single[nucs.index(group[i])])
                sites_temp.append(group[i])
                for j in range(i+1, len(group)):
                    if group[j] in nucs:
                        double_temp.append(double[nuc.index(group[i]), nuc.index(group[j])])
        single_freq.append(single_temp)
        double_freq.append(double_temp)
        new_sites.append(sites_temp)
    
    return new_sites, single_freq, double_freq


def main(args):
    """Find the sites that are almost fully linked."""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',            type=str,    default='linked-sites',          help='output file')
    parser.add_argument('--out_dir',     type=str,    default=None,                    help='directory to write linked files to')
    parser.add_argument('--data',        type=str,    default=None,                    help='directory to .npz files containing the counts, sequences, times, and mutant_sites for different locations')
    parser.add_argument('--g1',          type=float,  default=1,                       help='regularization restricting the magnitude of the selection coefficients')
    parser.add_argument('--freq_cutoff', type=float,  default=0.05,                    help='if a mutant frequency never rises above this number, it will not be used for inference')
    parser.add_argument('--link_tol',    type=float,  default=0.9,                     help='the tolerance for correlation check used for determining linked sites')
    parser.add_argument('--minCounts',   type=int,    default=10,                      help='the minimum number of times a mutation must be observed in order to be added to groups of linked sites')
    parser.add_argument('--c_directory', type=str,    default='Archive-alt2',          help='directory containing the c++ scripts')
    parser.add_argument('--timed',       type=int,    default=0,                       help='whether or not the program is timed')
    parser.add_argument('-w',            type=int,    default=10,                      help='the window size')
    parser.add_argument('-q',            type=int,    default=1,                       help='the number of states at each site')
    parser.add_argument('--refFile',     type=str,    default=None,                    help='the file containing the reference sequence and site indices')
    #parser.add_argument('--find_linked_anywhere', action='store_true', default=False,  help='whether or not to find sites that are linked in any region')
    parser.add_argument('--findLinkedRegional', action='store_true', default=False,  help='whether or not to find sites are linked in each region')
    parser.add_argument('--multisite',            action='store_true', default=False,  help='whether or not to to use multiple states at the same site')
    
    arg_list = parser.parse_args(args)
    
    out_str       = arg_list.o
    freq_cutoff   = arg_list.freq_cutoff
    link_tol      = arg_list.link_tol
    directory_str = arg_list.data
    c_directory   = arg_list.c_directory
    g1            = arg_list.g1
    timed         = arg_list.timed
    window        = arg_list.w
    q             = arg_list.q
    multisite     = arg_list.multisite
    #find_linked_anywhere = arg_list.find_linked_anywhere
    find_linked_anywhere = False
    find_linked_regional = arg_list.findLinkedRegional
    if arg_list.out_dir:
        out_dir = arg_list.out_dir
        link_outfile = os.path.join(arg_list.out_dir, out_str.split('/')[-1])
        if not os.path.exists(arg_list.out_dir):
            os.mkdir(arg_list.out_dir)
    else:
        #link_outfile = out_str.split('/')[-1]
        link_outfile = out_str
        out_dir      = ''
    
    simulations = len([name for name in os.listdir(directory_str)]) # number of locations to be combined
    
    if os.getcwd().split('/')[1]=='rhome':
        status_name = '/rhome/blee098/linked-sites-status.csv'
    else:
        status_name = '/Users/brianlee/Desktop/linked-sites-status.csv'
    stat_file   = open(status_name, 'w')
    stat_file.close()
    
    def print2(*args):
        stat_file = open(status_name, 'a+')
        line   = [str(i) for i in args]
        string = ','.join(line)
        stat_file.write(string+'\n')
        stat_file.close()
        print(string)
    
    
    def allele_counter(nVec, sVec, mutant_sites_samp, q=q):
        """ Calculates the counts for each allele. """
        if q == 1:
            single = np.zeros((len(mutant_sites_samp)))
            for i in range(len(mutant_sites_samp)):
                single[i] += np.sum([np.sum([nVec[t][j] * sVec[t][j][i] for j in range(len(sVec[t]))]) for t in range(len(nVec))])
        else:
            single = np.zeros(len(mutant_sites_samp) * q)
            for t in range(len(nVec)):
                for i in range(len(mutant_sites_samp)):
                    for j in range(len(nVec[t])):
                        single[i * q + sVec[t][j][i]] += nVec[t][j]
        return single
    
    
    def max_freq_counter(nVec, sVec, mutant_sites_samp, q=q):
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
                for j in range(len(nVec[t])):
                    for i in range(len(mutant_sites_samp)):
                        single[i * q + sVec[t][j][i]] += nVec[t][j]
        return single
    
    
    def max_freq(sVec, nVec, mutant_sites_samp, allele_number, window):
        """ Finds the maximum frequency that a mutation reaches in a particular population, 
        and if it is greater than the cutoff frequency, the name of the mutation site is appended
        to the allele_number variable. """
        
        Q = np.ones(len(nVec))
        for t in range(len(nVec)):
            if len(nVec[t]) > 0:
                Q[t] = np.sum(nVec[t])
        max_freq = np.zeros(len(mutant_sites_samp))
        for i in range(len(mutant_sites_samp)):
            for k in range(window):
                max_freq[i] += np.sum([nVec[k][j] * sVec[k][j][i] for j in range(len(sVec[k])) ]) / (window * Q[k])
        for t in range(1, len(nVec) - window + 1):
            new_freq = np.zeros(len(mutant_sites_samp))
            for i in range(len(mutant_sites_samp)):
                for k in range(window):
                    if Q[t] > 0:
                        new_freq[i] += np.sum([nVec[t+k][j] * sVec[t+k][j][i] for j in range(len(sVec[t+k]))]) / (window * Q[t+k])
            max_freq = np.maximum(max_freq, new_freq)
        for i in range(len(mutant_sites_samp)):
            if (max_freq[i] > freq_cutoff) and (mutant_sites_samp[i] not in allele_number):
                allele_number.append(mutant_sites_samp[i])
                
                
    def max_freq2(sVec, nVec, mutant_sites_samp, allele_number, window):
        """ Finds the maximum frequency that a mutation reaches in a particular population, 
        and if it is greater than the cutoff frequency, the name of the mutation site is appended
        to the allele_number variable. """
        
        Q = np.ones(len(nVec))
        for t in range(len(nVec)):
            if len(nVec[t]) > 0:
                Q[t] = np.sum(nVec[t])
        max_freq = np.zeros(len(mutant_sites_samp))
        for i in range(len(mutant_sites_samp)):
            for k in range(window):
                max_freq[i] += np.sum([nVec[k][j] * sVec[k][j][i] for j in range(len(sVec[k])) ]) / (window * Q[k])
        new_freq = np.zeros(len(mutant_sites_samp))
        for t in range(1, len(nVec) - window):
            for i in range(len(mutant_sites_samp)):
                new_freq[i] += np.sum([nVec[t+window][j] * sVec[t+window][j][i] for j in range(len(sVec[t+window]))]) / (window * Q[t+window])
                new_freq[i] -= np.sum([nVec[t][j] * sVec[t][j][i] for j in range(len(sVec[t]))]) / (window * Q[t])
            max_freq = np.maximum(max_freq, new_freq)
        for i in range(len(mutant_sites_samp)):
            if (max_freq[i] > freq_cutoff) and (mutant_sites_samp[i] not in allele_number):
                allele_number.append(mutant_sites_samp[i])
                
    
    def filter_sites_alternative(sVec_full, nVec_full, mutant_sites, allele_number):
        """ eliminate sites whose frequency is too small """
        
        sVec_full_new    = []
        mutant_sites_new = []
        for sim in range(simulations):
            #nVec    = nVec_full[sim]
            sVec = sVec_full[sim]
            mutants = mutant_sites[sim]
            #nVec_new    = []
            sVec_new = []
            mask = [(mutants[i] == allele_number).any() for i in range(len(mutants))]
            for t in range(len(sVec)):
                sVec_t = []
                for seq in sVec[t]:
                    sVec_t.append(np.array(np.array(seq)[mask]))
                sVec_new.append(np.array(sVec_t))
            sVec_full_new.append(np.array(sVec_new))
            mutant_sites_new.append(np.array(mutants)[mask])
        return mutant_sites_new, sVec_full_new
    

    if timed > 0:
        t_elimination = timer()
        print("starting")
        
    # loading reference sequence index
    ref_index = pd.read_csv(arg_list.refFile)
    index     = list(ref_index['ref_index'])
    index     = [str(i) for i in index]
    ref_full  = list(ref_index['nucleotide'])
    
    # Loading and organizing the data from the different locations. Combining it into new arrays.
    mutant_sites_full = []
    filepaths = []
    ref_sites = []
    locations_tot = []
    directory = os.fsencode(directory_str)
    for file in sorted(os.listdir(directory)):
        
        # Load data
        filename = os.fsdecode(file)
        filepath = os.path.join(directory_str, filename)
        location = filename[:-4]
        filepaths.append(filepath)
        print2(f'\tloading location {location}')
        data      = np.load(filepath, allow_pickle=True)  # Genome sequence data
        labels    = data['allele_number']                # The genome locations of mutations
        ref_sites.append(np.array(data['ref_sites'], dtype=str))
        locations_tot.append(location)
        
        # Append location specific information to lists.
        mutant_sites_full.append(np.array(labels, dtype=str))
    
    # Finds all sites at which there are mutations
    #mutant_sites_all = np.sort(np.unique(np.array([mutant_sites_full[i][j] for i in range(len(mutant_sites_full)) for j in range(len(mutant_sites_full[i]))])))
    
    # Filter sites with too low of a frequency in every population. 
    #mutant_sites_all = np.sort(np.unique(np.array([mutant_sites_full[i][j] for i in range(len(mutant_sites_full)) for j in range(len(mutant_sites_full[i]))])))
    mutant_sites_all = np.sort(np.unique(np.array([mutant_sites_full[i][j] for i in range(len(mutant_sites_full)) for j in range(len(mutant_sites_full[i]))])))
    mutant_sites_tot = ref_sites
    alleles_temp  = list(np.unique([ref_sites[i][j] for i in range(len(ref_sites)) for j in range(len(ref_sites[i]))]))
    allele_number = []
    ref_poly      = []
    for i in range(len(index)):
        if index[i] in alleles_temp:
            allele_number.append(index[i])
            ref_poly.append(ref_full[i])
    print2("number of inferred coefficients is {mutants} out of {mutants_all} total".format(mutants=len(allele_number), mutants_all=len(mutant_sites_all)))
    
    L             = len(allele_number)
    single_counts = np.zeros(L * q)     # The number of genomes with a mutation at each site across all populations
    double_counts = np.zeros((L * q, L * q)) # The double counts
    
    #ref_seq, ref_tag  = get_MSA(REF_TAG +'.fasta')
    #ref_seq  = list(ref_seq[0])
    #ref_poly = np.array(ref_seq)[allele_number]
   
    if timed > 0:
        tracker = 0
        t_combine = 0
    link_pairs_anywhere = []
    link_sites_regional = []
    for sim in range(simulations):
        
        # get data for the specific location  
        mutant_sites = mutant_sites_tot[sim]    # The simulation specific site labels
        data         = np.load(filepaths[sim], allow_pickle=True)  # Genome sequence data
        single_temp  = data['single_counts']
        double_temp  = data['double_counts']
        print(np.shape(double_temp))
        #print(double_temp)
        if 'max_freq' in data:
            freq_max = data['max_freq']
                
        # Find the frequency trajectories for this location
        if timed > 0:
            t_start_combo = timer()
        
        # Combine the single and double counts for this location into the overall ones
        alleles_sorted = np.argsort(allele_number)
        positions      = np.searchsorted(np.array(allele_number)[alleles_sorted], ref_sites[sim])
        positions      = alleles_sorted[positions]
        for i in range(len(mutant_sites)):
            single_counts[positions[i] * q : (positions[i] + 1) * q] += single_temp[i * q : (i + 1) * q]
            double_counts[positions[i] * q : (positions[i] + 1) * q, positions[i] * q : (positions[i] + 1) * q] += double_temp[i * q : (i + 1) * q, i * q : (i + 1) * q]
            for k in range(i + 1, len(mutant_sites)):
                double_counts[positions[i] * q : (positions[i] + 1) * q, positions[k] * q : (positions[k] + 1) * q] += double_temp[i * q : (i + 1) * q, k * q : (k + 1) * q]
                double_counts[positions[k] * q : (positions[k] + 1) * q, positions[i] * q : (positions[i] + 1) * q] += double_temp[k * q : (k + 1) * q, i * q : (i + 1) * q]
        """
        for i in range(len(mutant_sites)):
            for j in range(q):
                single_counts[positions[i] * q + j] += single_temp[i * q + j]
                for k in range(i + 1, len(mutant_sites)):
                    for l in range(q):
                        double_counts[positions[i] * q + j, positions[k] * q + l] += double_temp[i * q + j, k * q + l]
                        double_counts[positions[k] * q + l, positions[i] * q + j] += double_temp[i * q + j, k * q + l]
        """
        
        """
        for i in range(len(mutant_sites)):
            loc = list(allele_number).index(mutant_sites[i])
            for k in range(q):
                single_counts[loc * q + k] += single_temp[i * q + k]
                for j in range(1, len(mutant_sites)):
                    loc2 = list(allele_number).index(mutant_sites[j])
                    for l in range(q):
                        double_counts[loc  * q + k, loc2 * q + l] += double_temp[i * q + k, j * q + l]
                        double_counts[loc2 * q + l, loc  * q + k] += double_temp[i * q + k, j * q + l]
        """
                        
        if timed > 0:
            t_end_combo = timer()
            print2(f"took {t_end_combo - t_start_combo} seconds to add the counts from region {filepaths[sim].split('/')[-1]} to the overall counts")
                
        # Find pairs linked in this region
        if find_linked_anywhere or find_linked_regional:
            ref_poly_region = []
            for i in range(len(index)):
                if index[i] in mutant_sites:
                    ref_poly_region.append(ref_full[i])
            assert(len(ref_poly_region)==len(mutant_sites))
        if multisite:
            mutant_sites_new = []
            for i in range(len(mutant_sites)):
                for j in range(q):
                    mutant_sites_new.append(dp.get_label_new(mutant_sites[i] + '-' + NUC[j]))
            mutant_sites = mutant_sites_new
        if find_linked_anywhere or find_linked_regional:
            link_pairs_regional = find_linked_pairs_region(single_temp, double_temp, mutant_sites_new, link_pairs_anywhere, ref_poly=ref_poly_region, q=q, tol=link_tol) 
            if find_linked_anywhere:
                for pair in link_pairs_regional:
                    if pair not in link_pairs_anywhere:
                        link_pairs_anywhere.append(pair)
            if find_linked_regional:
                linked_temp = combine_linked_sites2(link_pairs_regional)
                for i in range(len(linked_temp)):
                    link_sites_regional.append(linked_temp[i])
                    
        if timed > 0:
            t_linked_anywhere = timer()
            print2(f"took {t_linked_anywhere - t_end_combo} seconds to find all of the linked sites in region {filepaths[sim].split('/')[-1]}")
                
    print('single count mean:', np.mean(single_counts))
    print('double count mean:', np.mean(double_counts))
    if timed > 0:
        t_counts = timer()
                
    # Finding linked pairs, then groups of linked sites
    if multisite:
        allele_new = []
        for i in range(len(allele_number)):
            for j in range(q):
                allele_new.append(str(allele_number[i]) + '-' + NUC[j])
        allele_number = allele_new
    
    linked_full = find_linked_pairs(single_counts, double_counts, allele_number, ref_poly=ref_poly, tol=link_tol, q=q, min_counts=arg_list.minCounts)
    print2('number of nonunique linked pairs:', len(linked_full))
    if timed > 0:
        t_pairs = timer()
        print2(f'Took {t_pairs - t_counts} seconds to find all linked pairs')
    
    linked_unique = linked_full
        
    linked_muts   = combine_linked_sites2(linked_unique)
    if timed > 0:
        t_groups = timer()
        print2(f'Took {t_groups - t_pairs} seconds to combine linked pairs into groups')
    
    #linked_muts  = find_linked_groups_alt(single_counts, double_counts, allele_number, tol=link_tol)
    if not multisite:
        linked_sites = [[dp.get_label(i) for i in linked_muts[j]] for j in range(len(linked_muts))]
    
    if find_linked_anywhere:
        linked_groups_anywhere = combine_linked_sites2(link_pairs_anywhere)
        if not multisite:
            linked_anywhere_sites  = [[dp.get_label(i[:-2]) + '-' + i[-1] for i in linked_groups_anywhere[j]] for j in range(len(linked_groups_anywhere))]
            np.save(link_outfile+'-anywhere-alleles.npy', linked_groups_anywhere)
            np.save(link_outfile+'-anywhere.npy', linked_anywhere_sites)
        else:
            np.save(link_outfile+'-anywhere.npy', linked_groups_anywhere)
    
    if find_linked_regional:
        np.savez_compressed(link_outfile + '-regional.npz', link_sites_regional=link_sites_regional, locations=locations_tot)
        
    # save the solution 
    if not multisite:
        np.save(link_outfile+'-alleles.npy', linked_muts)
        np.save(link_outfile+'.npy', linked_sites)
    else:
        np.save(link_outfile+'-alleles.npy', linked_muts)
        linked_sites = []
        for group in linked_muts:
            new_group = []
            for j in group:
                new_site = dp.get_label_new(j)
                new_group.append(new_site)
            linked_sites.append(new_group)
        np.save(link_outfile+'.npy', linked_sites)
    
    #np.savez_compressed(os.path.join(out_dir, 'nucleotide-counts.npz'), allele_number=allele_number, counts=single_counts)
    
    if timed > 0:
        end_time = timer()
        print2(f'total time = {end_time - t_elimination}')
    
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    
    
    

