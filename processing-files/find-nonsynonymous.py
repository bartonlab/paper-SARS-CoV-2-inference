#!/usr/bin/env python
# coding: utf-8
# %%

# %%

# Check what the above means

import sys
import argparse
import numpy as np                          # numerical tools
import os
import datetime as dt
import pandas as pd
import data_processing as dp
from copy import deepcopy


NUC = ['-', 'A', 'C', 'G', 'T']
ALPHABET = list('abcdefghijklmnopqrstuvwxyz')
PROTEINS = ['NSP1', 'NSP2', 'NSP3', 'NSP4', 'NSP5', 'NSP6', 'NSP7', 'NSP8','NSP9', 'NSP10', 'NSP12', 'NSP13', 'NSP14', 
            'NSP15', 'NSP16', 'S', 'ORF3a', 'E', 'M', 'ORF6', 'ORF7a', 'ORF7b', 'ORF8', 'N', 'ORF10']
PROTEIN_LENGTHS = {'ORF3a' : 828, 'E' : 228, 'ORF6' : 186, 'ORF7a' : 365, 'ORF7b' : 132, 'S' : 3822,
                   'N' : 1260, 'M' : 669, 'ORF8' : 336, 'ORF10' : 117, 'NSP1' : 539, 'NSP2' : 1914,
                   'NSP3' : 5834, 'NSP4' : 1500, 'NSP5' : 917, 'NSP6' : 870, 'NSP7' : 249, 'NSP8' : 594,
                   'NSP9' : 339, 'NSP10' : 417, 'NSP12' :2795, 'NSP13' : 1803, 'NSP14' : 1582, 
                   'NSP15' : 1038, 'NSP16' : 894}


def find_nonsynonymous_all(ref_seq, ref_index, data_folder, out_file, status_name='nonsynonymous_status.csv', simple=True):
    """ Finds the sites that are nonsynonymous and synonymous given the reference sequence and the folder containing the sequences
    and the polymorphic indices.
    If simple=True, then the mutated site is placed in the reference genome and it is checked whether or not the resulting codon 
    codes for the same amino acid."""
    ### FUTURE: when there are multiple mutations in a codon, consider whether reverting the mutation to the background 
    ### that it arose on makes it nonsynonymous, instead of just comparing with the reference sequence.
    ### Could do this by looking at the sequences that have the mutation and seeing the most frequent nucleotides to
    ### appear at the other sites in the codon as well.
    
    f = open(status_name, 'w')
    f.close()
    
    def print2(*args):
        stat_file = open(status_name, 'a+')
        line   = [str(i) for i in args]
        string = '\t'.join(line)
        stat_file.write(string+'\n')
        stat_file.close()
        print(string)
        
    # Determining whether mutations are nonsynonymous or synonymous
    ref_index  = list(ref_index)
    types      = []    # S or NS depending on if a mutation is synonymous or nonsynonymous
    aa_changes = []
    new_sites  = []
    nuc_nums   = []
    for z in range(len(ref_seq)):
        i = ref_index[z]
        if i[0]=='-' or i[-1] in ALPHABET:
            continue
        for j in NUC:
            lab = dp.get_label_new(i + '-' + j).split('-')
            # need a way to identify what the new amino acid will be for insertions
            #if i[-1] in ALPHABET:
                #aa_changes.append('->' + j)
                #if j=='-':
                #    types.append('S')
                #else:
                #    types.append('NS')
            #    continue
            new_sites.append(dp.get_label_new(i + f'-{j}'))
            nuc_nums.append(str(i) + f'-{j}')
            if lab[0] == 'NC':
                types.append('S')
                aa_changes.append(f'NC-{i}')
                continue
            start_idx      = dp.get_codon_start_index(int(i))
            codon_idxs     = np.arange(start_idx,start_idx+3)
            pos            = list(codon_idxs).index(int(i))    # the position of the mutation in the codon
            ref_codon      = ref_seq[ref_index.index(str(start_idx)):ref_index.index(str(start_idx))+3]
            ref_aa         = dp.codon2aa(ref_codon)
            if int(i) > 22200:
                print(ref_codon, start_idx, pos)
                print('')
            if j == '-' and ref_seq[z] != '-':
                types.append('NS')
                aa_changes.append(f'{ref_aa}>-')
            else:
                mut_codon      = deepcopy(ref_seq[ref_index.index(str(start_idx)):ref_index.index(str(start_idx))+3])
                mut_codon[pos] = j
                mut_aa         = dp.codon2aa(mut_codon)
                if ref_aa == mut_aa:
                    types.append('S')
                else:
                    types.append('NS')
                aa_changes.append(ref_aa + '>' + mut_aa)

    h = open(out_file + '.npz', 'wb')
    np.savez_compressed(h, types=types, locations=new_sites, aa_changes=aa_changes, nuc_index=nuc_nums)
    h.close()    


def main(args):
    """Infer time-varying selection coefficients from the results of a Wright-Fisher simulation"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('--data',    type=str,    default=None,                    help='folder containing the data files')   
    parser.add_argument('--refFile', type=str,    default=None,                    help='file containing the reference sequence and index')
    parser.add_argument('-o',        type=str,    default='synonymous-prot',       help='the name of the output file')
    parser.add_argument('--multisite', action='store_true', default=False,         help='whether or not the ')
    
    arg_list  = parser.parse_args(args)
    multisite = arg_list.multisite
    data      = arg_list.data
    ref_df    = pd.read_csv(arg_list.refFile)
    ref_seq   = list(ref_df['nucleotide'])
    ref_index = list(ref_df['ref_index'])
    
    #out_file = os.path.join(data, 'synonymous')
    out_file = os.path.join(data, arg_list.o)
    find_nonsynonymous_all(
        ref_seq,
        ref_index,
        data,
        out_file
    )
                    
                    
        
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    

