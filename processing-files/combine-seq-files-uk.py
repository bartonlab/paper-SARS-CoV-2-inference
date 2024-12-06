# coding: utf-8

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
import os
import math
import datetime as dt
import subprocess
import pandas as pd
import shutil
import gzip

REF_TAG = 'EPI_ISL_402125'
NUC     = ['-', 'A', 'C', 'G', 'T']

def find_site_index_file(filepath):
    """ Given a sequence file find the correct corresponding file that has the site names"""
    directory, file = os.path.split(filepath)
    return filepath[:-4] + '-sites.csv'


def get_data(file, get_seqs=True):
    """ Given a sequence file, get the sequences, dates, and mutant site labels"""
    #print('sequence file\t', file)
    #data = pd.read_csv(file, engine='python', converters={'sequence': lambda x: str(x)})
    data = pd.read_csv(file, engine='python', dtype={'sequence': str})
    #if not get_seqs:
    #    sequences = None
    #else:
    #    sequences   = np.array([np.array(list(str(i)), dtype=int) for i in list(data['sequence'])])
    #    seq_lens, seq_counts = np.unique([len(i) for i in sequences], return_counts=True)
    #    print(f'sequence lengths are {seq_lens}')
    #    print(f'number of sequences with each length {seq_counts}')
    #    common_len = seq_lens[np.argmax(seq_counts)]
    #    mask  = [i for i in range(len(sequences)) if len(sequences[i])==common_len]
    dates     = list(data['date'])
    sub_dates = list(data['submission_date'])
    index_file = find_site_index_file(file)
    index_data = pd.read_csv(index_file)
    ref_sites  = list(index_data['ref_sites'])
    if get_seqs:
        sequences   = np.array([np.array(list(str(i))) for i in list(data['sequence'])])
        seq_lens, seq_counts = np.unique([len(i) for i in sequences], return_counts=True)
        #print(f'sequence lengths are {seq_lens}')
        #print(f'number of sequences with each length {seq_counts}')
        common_len = seq_lens[np.argmax(seq_counts)]
        mask       = [i for i in range(len(sequences)) if len(sequences[i])==common_len]
        dates      = list(np.array(dates)[mask])
        sub_dates  = list(np.array(sub_dates)[mask])
        sequences  = sequences[mask]
        assert(len(ref_sites)==common_len)
    else:
        sequences = None
    dic = {
        'ref_sites' : ref_sites,
        'sequences' : sequences,
        'dates' : dates,
        'submission_dates' : sub_dates
    }
    return dic


def find_unique_regions(files):
    """ Find the unique regions over all files"""
    regs = [i[:-4] for i in files if 'sites' not in i]
    regs = [i[:i.find('---')].split('-') for i in regs]
    new_regs = []
    for reg in regs:
        while np.any(np.isin(list('0123456789'), list(reg[-1]))):
            reg = reg[:-1]
        new_regs.append('-'.join(reg))
    unique_regs = np.unique(new_regs)
    return unique_regs


def separate_time_series(files, max_dt=20, max_seqs=1000000000000, subdate=True):
    """For a specific region, collect files whose time-series are close together and separate ones that are far apart"""
    if not subdate:
        dates = [list(pd.read_csv(i, engine='python', usecols=['date'], dtype={'date' : int})['date']) for i in files]
    else:
        dates = [list(pd.read_csv(i, engine='python', usecols=['submission_date'], dtype={'submission_date' : int})['submission_date']) for i in files]
    sorter      = np.argsort([i[0] for i in dates])
    dates       = np.array(dates)[sorter]
    files_sort  = np.array(files)[sorter]
    init_dates  = [i[0] for i in dates]
    final_dates = [i[-1] for i in dates]
    groups      = [[files_sort[0]]]
    #print(dates)
    for i in range(1, len(files_sort)):
        #if init_dates[i] - final_dates[i - 1] <= max_dt:
        groups[-1].append(files_sort[i])
        #else:
        #    groups.append([files_sort[i]])
            
    # make new groups of files such that the number of sequences in each group is less than max_seqs
    new_groups = []
    for i in range(len(groups)):
        n_seqs        = [len(pd.read_csv(groups[i][j], engine='python', usecols=['date'])) for j in range(len(groups[i]))]
        running_count = n_seqs[0]
        new_group     = [groups[i][0]]
        for j in range(1, len(groups[i])):
            if running_count<=max_seqs:
                running_count += n_seqs[j]
                new_group.append(groups[i][j])
            else:
                new_groups.append(new_group)
                running_count = n_seqs[j]
                new_group = [groups[i][j]]
        new_groups.append(new_group)
        
    print(f'number of files in old groups = {np.sum([len(i) for i in groups])}')
    print(f'number of files in new groups = {np.sum([len(i) for i in new_groups])}')
    n_old = np.sum([len(i) for i in groups])
    n_new = np.sum([len(i) for i in new_groups])
    assert n_old==n_new
                
    return new_groups


def main(args):
    """Calculate the covariance matrix and the trajectories for a single region from SARS-CoV-2 data"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',            type=str,    default='parallel',              help='output directory')
    parser.add_argument('--data',        type=str,    default=None,                    help='.csv files containing the sequence data at different times')
    parser.add_argument('--window',      type=int,    default=10,                      help='the number of days over which to take the moving average')
    parser.add_argument('--minSeqs',     type=int,    default=15,                      help='the minimum number of sequences to use at the end at the beginning to calculate delta_x.')
    parser.add_argument('--refFile',     type=str,    default='ref-index.csv',         help='the file containing the site indices and reference sequence')

    arg_list = parser.parse_args(args)
    
    out_str       = arg_list.o
    window        = arg_list.window
    data          = arg_list.data
    min_seqs      = arg_list.minSeqs
    
    if not os.path.exists(out_str):
        os.mkdir(out_str)

    filepath    = data
    location    = os.path.split(filepath)[-1][:-4]
    
    status_name = f'combine-files.csv'
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
        
    # loading reference sequence index
    ref_index = pd.read_csv(arg_list.refFile)
    index     = list(ref_index['ref_index'])
    index     = [str(i) for i in index]
    ref_full  = list(ref_index['nucleotide'])
    
    # combine separate data files for the same region
    
    unique_regs  = find_unique_regions(os.listdir(data))
    #print2(unique_regs)
    #print2('\n')
    
    # find files for each region
    seq_files    = [file for file in os.listdir(data) if 'sites' not in file]
    files_by_reg = []
    reg_names    = []
    for reg in unique_regs:
        file_list = []
        for file in seq_files:
            if reg in file:
                file_list.append(os.path.join(data, file))
        reg_names.append(reg)
        files_by_reg.append(file_list)
        
    #print2(files_by_reg)
    #print2('\n')
    #file_groups = [separate_time_series(i) for i in files_by_reg]
    file_groups = []
    new_names   = []
    for i in range(len(files_by_reg)):
        new_groups = separate_time_series(files_by_reg[i])
        counter    = 1
        for j in new_groups:
            file_groups.append(j)
            new_names.append(f'{reg_names[i]}-{counter}')
            counter += 1
            
    #print2(file_groups)
        
    #for reg in unique_regs:
        #print2(f'working on region {reg}')
        # make output directory for this region
        #region_dir = os.path.join(out_str, reg)
        #if not os.path.exists(region_dir):
        #    os.mkdir(region_dir)
    
        
        
        # find all files that have sequence data for this region
        #region_files = []    # the set of all sequence files that correspond to this region
        #for file in os.listdir(data):
        #    if reg in file and 'sites' not in file:
        #        region_files.append(os.path.join(data, file)) 
    for z in range(len(file_groups)):
        print2(f'working on group {file_groups[z]}')
        print2(f'file name {new_names[z]}')
        region_files = file_groups[z]
        reg          = new_names[z]
        #reg = find_unique_regions(regions_files)[0]
        if len(region_files)==1:
            shutil.copyfile(region_files[0], os.path.join(out_str, os.path.split(region_files[0])[-1]))
            sites_file = region_files[0][:-4] + '-sites.csv'
            shutil.copyfile(sites_file, os.path.join(out_str, os.path.split(sites_file)[-1]))
            continue
            
        # find all of the sites in each of the files so that sequence data can be combined
        region_sites = []    # the set of all mutations in each of the files
        for file in region_files:
            index_file = find_site_index_file(file)
            index_data = pd.read_csv(index_file)
            ref_sites  = [str(i) for i in list(index_data['ref_sites'])]
            region_sites.append(ref_sites)
        sites_temp = list(np.unique([region_sites[i][j] for i in range(len(region_sites)) for j in range(len(region_sites[i]))]))
        
        # find reference nucleotides so that these can be inserted into sequences
        all_sites  = []
        ref_seq    = []
        for i in range(len(index)):
            if index[i] in sites_temp:
                all_sites.append(index[i])
                ref_seq.append(str(NUC.index(ref_full[i])))
        print2(f'length of sites accross all files is {len(all_sites)}')
        if len(all_sites)==0:
            print2(region_sites)
            
                
        # save new ref_sites file
        out_file = os.path.join(out_str, reg)
        sites_df = pd.DataFrame(data={'ref_sites' : all_sites})
        sites_df.to_csv(out_file + '-sites.csv', index=False)
        
        # combine sequences from each of the files into one, inserting reference nucleotides at sites that were not present
        f = open(out_file + '.csv', 'w')
        f.write('date,submission_date,sequence\n')
        for file in region_files:
            reg_data  = get_data(file)
            dates     = reg_data['dates']
            sub_dates = reg_data['submission_dates']
            ref_sites = np.array(list(reg_data['ref_sites']))
            seqs      = [list(i) for i in reg_data['sequences']]
            
            sites_sorted = np.argsort(all_sites)
            positions    = np.searchsorted(np.array(all_sites)[sites_sorted], ref_sites)
            positions    = sites_sorted[positions]
            for i in range(len(dates)):
                seq     = seqs[i]
                new_seq = [i for i in ref_seq]
                for j in range(len(ref_sites)):
                    new_seq[positions[j]] = seq[j]
                new_seq = ''.join(new_seq)
                f.write(f'{dates[i]},{sub_dates[i]},{new_seq}\n')
        f.close()

   

        


if __name__ == '__main__': 
    main(sys.argv[1:])

