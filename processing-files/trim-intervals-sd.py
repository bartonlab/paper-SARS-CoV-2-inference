#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
import os
import shutil
import datetime
import pandas as pd

def find_site_index_file(filepath):
    """ Given a sequence file find the correct corresponding file that has the site names"""
    directory, file = os.path.split(filepath)
    if file.find('---')==-1:
        return filepath[:-4] + '-sites.csv'
    else:
        return filepath[:filepath.find('---')] + '-sites.csv' 
    
    
def slice_time_series(file, times_new, out_file):
    """ Cuts off the sequences at the beginning and end and resaves the data. """
    df = pd.read_csv(file)
    df_new = df[(df['date'] >= times_new[0]) & (df['date'] <= times_new[-1])]
    df_new.to_csv(out_file + '.csv', index=False)
    
    
def find_intervals(in_file, window=5, min_seqs=20, max_dt=5, min_range=20, end_cutoff=3):
    """ Determine the time intervals that have good enough sampling to be used for inference. 
    Expects an input .csv file."""
    df = pd.read_csv(in_file)
    df = df.sort_values(by=['date'])
    vc = df['date'].value_counts(sort=False)
    times  = np.array(list(vc.axes[0]))
    counts = np.array(list(vc))
    mask   = np.argsort(times)
    times  = times[mask]
    counts = counts[mask]
    print(times)
    print(counts)
    #assert sorted(times) == times \
    new_times  = np.arange(np.amin(times), np.amax(times) + 1)
    new_counts = np.zeros(len(new_times))
    for i in range(len(new_times)):
        if new_times[i] in times:
            idx = list(times).index(new_times[i])
            new_counts[i] = counts[idx]
    times  = new_times
    counts = new_counts
    
    cumulative_counts = 0
    last_time_used    = 0
    n_intervals       = 0
    new_times = []
    recording = False
    
    for t in range(len(times)):
        print(cumulative_counts)
        cumulative_counts += counts[t]
        if t > window - 1:
            cumulative_counts -= counts[t - window]
        if cumulative_counts >= min_seqs:
            if not recording:
                if t - window + 1 > 0:
                    if counts[t - window + 1]!=0 and (t - window + 1)>=last_time_used:
                        recording  = True
                        zero_dt    = 0
                        times_temp =  list(times[t - window + 1 : t + 1])
                    else:
                        continue
                else:
                    recording  = True
                    zero_dt    = 0
                    times_temp = list(times[:t+1])
                    print(times_temp)
            else:
                times_temp.append(times[t])
            if counts[t]==0:
                zero_dt += 1
                if zero_dt >= max_dt:
                    recording = False
                    cumulative_counts = np.sum(counts[t - window + 1:t + 1])
                    zero_dt = 0
                    if len(times_temp) - max_dt - end_cutoff >= min_range:
                        new_times.append(times_temp[:-max_dt-end_cutoff])
                        last_time_used = list(times).index(new_times[-1][-1])
                        n_intervals += 1
            else:
                zero_dt = 0
        else:
            if recording:
                recording = False
                cumulative_counts = np.sum(counts[t - window + 1:t + 1])
                if len(times_temp) - end_cutoff - zero_dt >= min_range:
                    if zero_dt > 0:
                        new_times.append(times_temp[:-zero_dt-end_cutoff])
                        last_time_used = list(times).index(new_times[-1][-1])
                    elif end_cutoff!=0:
                        new_times.append(times_temp[:-end_cutoff])
                        last_time_used = new_times[-1][-1]
                    else: 
                        new_times.append(times_temp)
                        last_time_used = list(times).index(new_times[-1][-1])
                    zero_dt      = 0
                    n_intervals += 1
        if times[t] == times[-1] and recording and len(times_temp) >= min_range:
            new_times.append(times_temp)
            n_intervals +=1
            
    return n_intervals, new_times



def trim_time_series(input_file, out_folder, window=5, max_dt=5, min_seqs=20, min_range=20, end_cutoff=0, ignore_short=True):
    """finds the intervals that have good sampling in a region, cuts the data array to these 
    intervals, and resaves the file."""
    ref_date = datetime.date(2020, 1, 1)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
        
    sites_filepath = find_site_index_file(input_file)
    site_dir, site_file = os.path.split(sites_filepath)
    shutil.copy(sites_filepath, os.path.join(out_folder, site_file))
    
    filename = os.path.split(input_file)[1]
    print(f'working on file {filename}')
    df = pd.read_csv(input_file)
    full_times = list(df['date'])
    if filename.find('None') != -1:
        location = filename[:filename.find('None') - 1]
    else:
        location = filename[:np.amin([filename.find(i) for i in [str(j) for j in np.arange(10)]]) - 1]
    if location.find('.') != -1:
        location = location[:location.find('.')]
    print(f'location is {location}')
    regions = filename.split('-')
    #if location.split('-')[1] in ['united kingdom', 'united_kingdom'] and ignore_short:
    print(f'dataframe length {len(df)}')
    print(f'number of times is {len(np.arange(np.amin(full_times), np.amax(full_times)+1))}')
    if len(np.arange(np.amin(full_times), np.amax(full_times)+1)) < 20 and len(df) > 1000:
        n_intervals = 1
        #data        = np.load(input_file, allow_pickle=True)
        times       = [np.arange(np.amin(full_times), np.amax(full_times)+1)]
    else:
        n_intervals, times = find_intervals(
            input_file, max_dt=max_dt, window=window,
            min_seqs=min_seqs, min_range=min_range, 
            end_cutoff=end_cutoff
        )
    for i in times:
        print(i)
    print(times)
    print(f'number of intervals is {n_intervals}')
    n_intervals = len(times)
    print(f'number of intervals is {n_intervals}')
        
    for i in range(n_intervals):
        times_temp = times[i]
        start_year, end_year   = (ref_date + datetime.timedelta(int(times_temp[0]))).year,   (ref_date + datetime.timedelta(int(times_temp[-1]))).year
        start_month, end_month = (ref_date + datetime.timedelta(int(times_temp[0]))).month,  (ref_date + datetime.timedelta(int(times_temp[-1]))).month
        start_day, end_day     = (ref_date + datetime.timedelta(int(times_temp[0]))).day,    (ref_date + datetime.timedelta(int(times_temp[-1]))).day
        out_file = location + '---' + str(start_year) + '-' + str(start_month) + '-' + str(start_day) + '-' + str(end_year) + '-' + str(end_month) + '-' + str(end_day)
        print(f'out file is {out_file}')
        slice_time_series(input_file, times_temp, os.path.join(out_folder, out_file))

        
def main(args):
    """ Trims time-series to dates that have good sampling. """
    
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',             type=str,    default=None,     help='output directory')
    parser.add_argument('--input'     ,   type=str,    default=None,     help='input directory')
    parser.add_argument('--seq_dir',      type=str,    default=None,     help='the directory with the sequences if this is different from genome-data')
    parser.add_argument('--max_dt',       type=int,    default=5,        help='the maximum number of consecutive days with zero sequences allowed')
    parser.add_argument('--window',       type=int,    default=5,        help='the number of days within which min_seqs must be found')
    parser.add_argument('--minSeqs',      type=int,    default=20,       help='the minimum number of sequences that must be found in each window days')
    parser.add_argument('--minRange',     type=int,    default=20,       help='the shortest time-series allowed')
    parser.add_argument('--trimDir',      type=str,    default=None,     help='the directory to write the trimmed files to')
    parser.add_argument('--trim_exists',  default=False, action='store_true', help='whether or not the trimmed directory has already been made')
    
    arg_list = parser.parse_args(args)
    
    out_dir   = arg_list.o        
    in_file   = arg_list.input
    max_dt    = arg_list.max_dt
    window    = arg_list.window
    min_seqs  = arg_list.minSeqs
    min_range = arg_list.minRange
    seq_dir   = arg_list.seq_dir
    
    used_files = 'used-files'
    if not os.path.exists(used_files):
        os.mkdir(used_files)
    
    if arg_list.trimDir:
        trim_dir = os.path.join(out_dir, arg_list.trimDir)
    else:
        trim_dir = os.path.join(out_dir, 'genome-trimmed')
    if not arg_list.trim_exists:
        print('restricting intervals')
        trim_time_series(in_file, trim_dir, window=window, max_dt=max_dt, min_seqs=min_seqs, min_range=min_range)
        f = open(os.path.join(used_files, os.path.split(in_file)[1]), mode='w')
        f.write('used')
        f.close()
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
