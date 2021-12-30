#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
import os
import subprocess
import shutil
import data_processing as dp

REF_TAG = 'EPI_ISL_402125'
NUC     = ['-', 'A', 'C', 'G', 'T']

def filter_sites(data_dir, max_file, out_name=None, freq_cutoff=0.05):
    """ Given the maximum frequency that any mutation achieves in any of the regions (given in max_file), 
    eliminate all sites that are never greater than freq_cutoff in any region and resave the data.
    out_name specifies the name of the output directory"""
    
    if out_name==None:
        out_name=f'freq_{freq_cutoff}'
    if out_name.find('/')!=-1:
        print('error with the name of the output directory')
        out_dir = out_name
    else:
        out_dir = os.path.join(os.path.split(data_dir)[0], out_name)
    out_dir = os.path.join(out_dir, f'freq_{freq_cutoff}')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    # load maximum frequency data
    freq_data     = np.load(max_file, allow_pickle=True)
    site_names    = freq_data['allele_number']
    max_freqs     = freq_data['max_freq']
    allele_number = site_names[max_freqs>=freq_cutoff]
    print(len(allele_number))
    
    # eliminate sites and resave data
    for file in sorted(os.listdir(data_dir)):
        filepath  = os.path.join(data_dir, file)
        data_temp = np.load(filepath, allow_pickle=True)
        nVec      = data_temp['nVec']
        sVec      = data_temp['sVec']
        muts      = data_temp['mutant_sites']
        times     = data_temp['times']
        mask      = np.isin(muts, allele_number)
        #print(mask)
        #print(muts)
        sVec_new  = []
        for i in range(len(sVec)):
            sVec_t = []
            for seq in sVec[i]:
                sVec_t.append(np.array(seq)[mask])
            sVec_new.append(np.array(sVec_t))
        muts_new = muts[mask]
        #print(muts_new)
        print(len(muts_new))
        print(len(sVec[0][0]))
            
        out_file = os.path.join(out_dir, file)
        f = open(out_file, mode='wb')
        np.savez_compressed(f, nVec=nVec, sVec=sVec_new, times=times, mutant_sites=muts_new)
        f.close()

def main(args):
    """ determine the maximum frequency that each mutant site reaches in a given region"""
    
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',             type=str,    default=None,           help='output directory')
    parser.add_argument('--max_file',     type=str,    default=None,           help='file containing maximum frequencies for each site')
    parser.add_argument('--input_dir',    type=str,    default=None,           help='input directory containing sequence files')
    parser.add_argument('--max_freq',     type=float,  default=0.05,           help='the maximum frequency at which to cutoff sites')
    
    arg_list = parser.parse_args(args)
    
    out_dir   = arg_list.o        
    max_file  = arg_list.max_file
    input_dir = arg_list.input_dir
    max_freq  = arg_list.max_freq
    
    filter_sites(input_dir, max_file, out_name=out_dir, freq_cutoff=max_freq)    
    
if __name__ == '__main__': 
    main(sys.argv[1:])