#!/usr/bin/env python
# coding: utf-8
# %%

# %%

import sys
import argparse
import os
import shutil
import numpy as np

REF_TAG = 'EPI_ISL_402125'


def region_name(file):
    """ Find the region name given the file name"""
    numbers   = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    file_temp = file[:file.find('---')]
    split     = file_temp.split('-')
    while any([i in list(split[-1]) for i in numbers]):
        split = split[:-1]
    new_file  = '-'.join(split)
    return new_file


def main(args):
    """ Formats the regional sequence data so that it can be analyzed by the smoothing deconvolution method. """
    
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',             type=str,    default=None,     help='output directory')
    parser.add_argument('--input',        type=str,    default=None,     help='input directory')
    
    arg_list = parser.parse_args(args)
    
    out_dir  = arg_list.o        
    in_dir   = arg_list.input
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    # Find different regions
    locs_all = []
    for file in os.listdir(in_dir):
        filepath = os.path.join(in_dir, file)
        if os.path.isfile(filepath):
            locs_all.append(region_name(file))
    locs_unique = np.unique(locs_all)
        
    # Create new directories for each region
    for region in locs_unique:
        new_dir   = os.path.join(out_dir, region)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        covar_dir = os.path.join(new_dir, 'tv_covar')
        if not os.path.exists(covar_dir):
            os.mkdir(covar_dir)
        
    # Transfer files from old directories into new regional directories
    
    # move data files
    for file in os.listdir(in_dir):
        filepath = os.path.join(in_dir, file)
        if os.path.isfile(filepath):
            location = region_name(file)
            outfile  = os.path.join(out_dir, location, file)
            shutil.copyfile(filepath, outfile)
            
    # move covariance files
    covar_in_dir = os.path.join(in_dir, 'tv_covar')
    for file in os.listdir(covar_in_dir):
        filepath       = os.path.join(covar_in_dir, file)
        location       = region_name(file)
        covar_out_file = os.path.join(out_dir, location, 'tv_covar', file)
        shutil.copyfile(filepath, covar_out_file)
        
            
    
if __name__ == '__main__': 
    main(sys.argv[1:])


