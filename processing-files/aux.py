#!/usr/bin/env python
# coding: utf-8

# Auxiliary functions for sequence analysis

import os
import sys

import numpy as np
import datetime as dt
import pandas as pd


NUC = ['-', 'A', 'C', 'G', 'T']


# COLUMN NAMES

COL_SEQ    = 'sequence'   # name of the column for sequences
COL_SNV    = 'mutation'   # input column for SNVs
COL_TIME   = 'time'       # input/output column storing time
COL_S      = 'selection'  # input/output column for selection coef
COL_REGION = 'location'   # region column in selection file

COL_REF_IDX  = 'ref_index'   # reference sequence index
COL_REF_SITE = 'ref_sites'   # reference site index
COL_REF_NUC  = 'nucleotide'  # reference sequence nucleotide

COL_UPLOAD  = 'submission_date'  # name of the column for upload date
COL_COLLECT = 'date'             # name of the column for collection date

COL_GROUP  = 'group'                # linked groups
COL_ASSOC  = 'variants_associated'  # variant association (weak)
COL_MEMBER = 'variants_member'      # variant association (strong)
COL_MATCH  = 'variants_match'       # closest variant association by Hamming distance

COL_THRESH = 'threshold'       # linked group selection threshold
COL_TP     = 'true_positive'   # true positive column
COL_FP     = 'false_positive'  # false positive column


# ___ FUNCTIONS FOR FINDING LINKED GROUPS ___ #

def find_seq_counts(df_recent, last_seqs=100):
    """Finds the unique sequences and adds them to a dictionary as the keys. 
    The number of appearances of the sequence is the value."""
    r_seq_counts = {}
    for i in range(min(last_seqs, len(df_recent))):
        if df_recent.iloc[i][COL_SEQ] in r_seq_counts:
            r_seq_counts[str(df_recent.iloc[i][COL_SEQ])] += 1
        else:
            r_seq_counts[str(df_recent.iloc[i][COL_SEQ])]  = 1
    return r_seq_counts
    

def find_freqs(seq_counts, q=5):
    """Calculates the single and double site frequencies"""
    keys   = [i for i in seq_counts]
    L      = len(keys[0])
    single = np.zeros(L*q)
    double = np.zeros((L*q,L*q))
    for key in keys:
        seq   = np.array(list(key), dtype=int)
        count = seq_counts[key]
        for i in range(L):
            single[i*q+seq[i]] += count
            for j in range(i+1, L):
                double[i*q+seq[i], j*q+seq[j]] += count
                double[j*q+seq[j], i*q+seq[i]] += count
    return single, double


def find_sites(ref_file, site_file):
    """Given a reference sequence file and a file containing the sites that correspond to sequences in a region, 
    find the reference nucleotides at those sites"""
    ref_index = pd.read_csv(ref_file)
    index     = list(ref_index[COL_REF_IDX])
    index     = [str(i) for i in index]
    ref_full  = list(ref_index[COL_REF_NUC])
    
    sites_data    = pd.read_csv(site_file)
    ref_sites     = [str(i) for i in list(sites_data[COL_REF_SITE])]
    allele_number = []
    ref_poly      = []
    for i in range(len(index)):
        if index[i] in ref_sites:
            allele_number.append(index[i])
            ref_poly.append(ref_full[i])
    return allele_number, ref_poly


def find_linked_pairs(single, double, alleles, tol=0.8, q=5, ref_poly=None, min_counts=1):
    """ Takes the single and double site counts and finds pairs of mutations that are linked."""
    labels = []
    for site in alleles:
        for i in range(q):
            labels.append(site + '-' + NUC[i])
    alleles = labels
    
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


def combine_linked_pairs(linked_sites):
    """ Combines the pairs of linked sites into lists of mutually linked sites.
    FUTURE: Try making a set of all sites linked to a site for each site. 
    For sets that have nonzero intersection, take their union (consider alternative approaches to this step)."""
    
    #linked_sites = np.unique(linked_sites)
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


def get_linked_groups(df, ref_file=None, site_file=None, last_seqs=100, min_counts=1):
    """Takes a dataframe containing recent sequences ordered by upload date and finds linked groups"""
    if ref_file is None:
        print('no ref_file keyword passed')
    if site_file is None:
        print('no site_file keyword passed')
    seq_counts       = find_seq_counts(df, last_seqs=last_seqs)
    alleles, ref_seq = find_sites(ref_file, site_file)
    single, double   = find_freqs(seq_counts)
    linked_pairs     = find_linked_pairs(single, double, alleles, ref_poly=ref_seq, min_counts=min_counts)
    linked_groups    = combine_linked_pairs(linked_pairs)
    #linked_groups   = [[dp.get_label_new(i) for i in linked_muts[j]] for j in range(len(linked_muts))]
    return linked_groups


# ^ ^ ^ END OF FUNCTIONS FOR FINDING LINKED GROUPS ^ ^ ^ #


# AUXILIARY FUNCTION: MAP FROM NUMBER TO NUCLEOTIDE
def num2nuc(number):
    return NUC[number]


# AUXILIARY FUNCTION: GET SELECTION COEFFICIENTS
def get_s(seqs, df_s):
    s_values = []
    for seq in seqs:
        #temp_seq = np.array(seq.split(), dtype=int)
        temp_s   = 0
        for i in range(len(seq)):
            #if seq[i]!=wt_seq[i]:
            #temp_s += df_s[df_s[COL_SNV]==(str(df_map.iloc[i][COL_MAP])+'-'+num2nuc(int(seq[i])))][COL_S]
            #print(df_s[df_s[COL_SNV]==seq[i]])
            #print(df_s[df_s[COL_SNV]==seq[i]][COL_S].iloc[0])
            temp_s += df_s[df_s[COL_SNV]==seq[i]][COL_S].iloc[0]
        s_values.append(temp_s)
    return s_values


def get_s_alt(seqs, df_s, muts):
    s_values = []
    s_dict   = {}
    #print(muts)
    print(np.unique(list(df_s[COL_SNV])))
    if len(df_s)==0:
        s_values = np.zeros(len(seqs))
        return s_values
    for site in muts:
        for n in NUC:
            mut = site + '-' + n
            if mut not in np.unique(list(df_s[COL_SNV])):
                print(f'mutation {mut} not in inference file at this time')
                print(mut)
                #print(np.unique(list(df_s[COL_SNV])))
                s_dict[mut] = 0
            else:
                s_dict[mut] = df_s[df_s[COL_SNV]==mut][COL_S].iloc[0]
    for seq in seqs:
        temp_seq = np.array(list(seq), dtype=int)
        temp_seq = [muts[i] + '-' + NUC[temp_seq[i]] for i in range(len(temp_seq))]
        temp_s   = 0
        for i in range(len(seq)):
            #if seq[i]!=wt_seq[i]:
            #temp_s += df_s[df_s[COL_SNV]==(str(df_map.iloc[i][COL_MAP])+'-'+num2nuc(int(seq[i])))][COL_S]
            #print(df_s[df_s[COL_SNV]==seq[i]])
            #print(df_s[df_s[COL_SNV]==seq[i]][COL_S].iloc[0])
            #site = muts[i] + '-' + NUC[temp_seq[i]]
            #if site not in s_dict:
            if temp_seq[i] not in s_dict:
                print(f'site {seq[i]} not in selection coefficients for this time')
                continue
            temp_s += s_dict[temp_seq[i]]
            #temp_s += df_s[df_s[COL_SNV]==site][COL_S].iloc[0]
        s_values.append(temp_s)
    return s_values

    
    
# AUXILIARY FUNCTION: GET SELECTION COEFFICIENT FOR SEQUENCE AS SET
def get_s_set(seq_set, df_s):
    temp_s = 0
    for mut in seq_set:
        temp_s += df_s[df_s[COL_SNV]==mut][COL_S].iloc[0]
    return temp_s


# AUXILIARY FUNCTION: GET VARIANT ASSOCIATION
def get_var(seqs, df_var, var_min_muts=1, get_closest=False):
    variants  = df_var.columns.values[2:]
    mutations = list(df_var[COL_SNV])
    associated_vars = []    # variants that have any mutations in common with a given group
    member_vars     = []    # variants that have at least VAR_MIN_MUT mutations in common with a given group
    closest_vars    = []    # best matching variant
    #num_mutations   = np.array(df_var.drop(labels=[COL_SNV, 'label'], axis=1).sum())  # total number of mutations for each variant
    var_muts        = []
    for col in variants:
        var_muts.append(set(np.array(mutations)[df_var[col]==True]))
    for group in seqs:
        group_idxs = [mutations.index(i) for i in group if i in mutations]
        if len(group_idxs)==0:
            associated_vars.append([])
            member_vars.append([])
            closest_vars.append([])
            continue
        group_df   = df_var.iloc[group_idxs]                           # rows for SNVs in group that are in variants
        group_df   = group_df.drop(labels=[COL_SNV, 'label'], axis=1)  # drop unneeded columns
        var_counts = list(group_df.sum())                              # sum columns to get total number of SNVs assoc w/ each variant
        #print(var_counts)
        #print(var_counts)
        group_vars = []
        main_vars  = []
        for i in range(len(var_counts)):
            if var_counts[i]>0:
                group_vars.append(variants[i])
            if var_counts[i]>=var_min_muts:
                main_vars.append(variants[i])
        associated_vars.append(group_vars)
        member_vars.append(main_vars)
        
        # Check which variant is closest, measured by fraction overlapping (seq/var)
        if get_closest:
            group_muts = set([i for i in group if i in mutations])
            hamm_dists = [len(group_muts.symmetric_difference(i)) for i in var_muts]
            closest_vars.append(variants[np.argmin(hamm_dists)])
            #overlap = np.array(var_counts)/num_mutations
            #closest_vars.append(variants[np.argmax(overlap)])
            #if variants[np.argmax(overlap)] in main_vars:
            #    closest_vars.append(variants[np.argmax(overlap)])
            #else:
            #    closest_vars.append('')
            
    if get_closest:
        return associated_vars, member_vars, closest_vars
    else:
        return associated_vars, member_vars


# AUXILIARY FUNCTION: REMOVE MUTATIONS THAT HAVE ALREADY BEEN DETECTED FROM GROUPS OF MUTATIONS
def remove_detected_muts(detected_muts, groups):
    new_groups = []
    for group in groups:
        new_group = [i for i in group if i not in detected_muts]
        if len(new_group)>0:
            new_groups.append(new_group)
    return new_groups


# AUXILIARY FUNCTION: ADD MUTATIONS THAT ARE PART OF DETECTED GROUPS TO THE SET OF DETECTED MUTATIONS
def add_detected_muts(detected_muts, groups, groups_s, min_sel=0):
    for i in range(len(groups)):
        if groups_s[i] > min_sel:
            for mut in groups[i]:
                detected_muts.add(mut)


# AUXILIARY FUNCTION: FIND REGION FROM FILENAME
def find_region(file):
    numbers = list('0123456789')
    temp = os.path.split(file)[-1]
    temp = temp[:temp.find('.')]
    if '---' in temp:
        temp = temp[:temp.find('---')]
    temp = temp.split('-')
    while any([i in list(temp[-1]) for i in numbers]):
        temp = temp[:-1]
    return '-'.join(temp)


# AUXILIARY FUNCTION: MAKE DICTIONARY OF SELECTION COEFFICIENTS FROM DF
def make_selection_dict(group_df, df_s):
    groups    = [i.split() for i in list(group_df[COL_GROUP])]
    mutations = np.unique([groups[i][j] for i in range(len(groups)) for j in range(len(groups[i]))])
    selection = {i : {} for i in mutations}
    for mut in mutations:
        df_mut = df_s[df_s[COL_SNV]==mut]
        if len(df_mut)==0:
            print(f'mutation {mut} has no corresponding selection information')
        times = np.unique(list(df_mut[COL_TIME]))
        for t in times:
            selection[mut][t] = df_mut[df_mut[COL_TIME]==t][COL_S].iloc[0]
    return selection
            

# AUXILIARY FUNCTION: GET SELECTION FROM DICTIONARY
def get_s_dict(seq, s_dict, time):
    #print([s_dict[i] for i in seq])
    return np.sum([s_dict[i][time] for i in seq if time in s_dict[i]])


# AUXILIARY FUNCTION: MAKE SET OF ALL NUCLEOTIDES AT SITES FROM A SET OF INDIVIDUAL NUCLEOTIDES
def add_all_nucs(group):
    sites = [i[:-1] for i in group]
    new_group = []
    for site in sites:
        for nuc in NUC:
            new_group.append(site + nuc)
    return set(new_group)

    

