#!/usr/bin/env python
# coding: utf-8

import sys
import os
import argparse

import aux  # auxiliary functions

import numpy as np
import pandas as pd
import datetime as dt
from timeit import default_timer as timer


def main(args):
    """Find sites that are almost fully linked."""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',           type=str,    default=None, help='output directory')
    parser.add_argument('--inSeqs',     type=str,    default=None, help='input sequence file')
    #parser.add_argument('--inInf',     type=str,    default=None, help='input inference file containing inferred selection coefficients over time for the region')
    #parser.add_argument('--infDir',      type=str,    default=None,                    help='input inference directory containing inference in each region at every time')
    parser.add_argument('--inVar',      type=str,    default=None, help='input file for associating mutations with variants')
    parser.add_argument('--refFile',    type=str,    default=None, help='the file containing the reference sequence and site indices')
    parser.add_argument('-q',           type=int,    default=5,    help='the number of states at each site')
    parser.add_argument('--linkTol',    type=float,  default=0.8,  help='the tolerance for correlation check used for determining linked sites')
    parser.add_argument('--minMuts',    type=int,    default=5,    help='the minimum number of mutations that a group must share with a variant to be considered strongly associated')
    parser.add_argument('--detectTol',  type=float,  default=0,    help='the minimum selection coefficient in order for a group of mutations to be considered detected')
    parser.add_argument('--lastSeqs',   type=int,    default=100,  help='the number of most recent sequences to consdier when finding linked groups')
    parser.add_argument('--minCounts',  type=int,    default=1,    help='the minimum number of sequences that a mutation must appear on in order to be added to the linked groups')
    parser.add_argument('--mutsIgnore', type=str,    default=None, help='a file containing mutations to eliminate from linked groups')
    parser.add_argument('--startDate',  type=int,    default=0,    help='the date to start from if provided')
    parser.add_argument('--useSelection', default=False, action='store_true', help='whether or not to eliminate muations using the selection file')
    
    arg_list = parser.parse_args(args)
    
    link_tol   = arg_list.linkTol
    min_muts   = arg_list.minMuts
    min_sel    = arg_list.detectTol
    last_seqs  = arg_list.lastSeqs
    min_counts = arg_list.minCounts
    start_date = arg_list.startDate
    
    out_str  = arg_list.o
    ref_file = arg_list.refFile
    seq_file = arg_list.inSeqs
    var_file = arg_list.inVar
    #inf_file = arg_list.inInf
    #inf_dir  = arg_list.infDir
    q        = arg_list.q

    use_selection = arg_list.useSelection
    
    site_file = seq_file[:-4] + '-sites.csv'
    region    = aux.find_region(seq_file)
    #inf_file  = os.path.join(inf_dir, region + '.csv')
    out_file  = os.path.join(out_str, os.path.split(seq_file)[-1])
    if arg_list.mutsIgnore:
        muts_ignore = np.loadtxt(arg_list.mutsIgnore, dtype=str)
    else:
        muts_ignore = []
    
    status_name = f'status-linked-groups-selection-{os.path.split(seq_file)[-1][:-4]}.csv'
    stat_file   = open(status_name, 'w')
    stat_file.close()
    
    def print2(*args):
        stat_file = open(status_name, 'a+')
        line   = [str(i) for i in args]
        string = ','.join(line)
        stat_file.write(string+'\n')
        stat_file.close()
        print(string)
        
    # loading reference sequence index
    ref_index = pd.read_csv(ref_file)
    index     = list(ref_index[aux.COL_REF_IDX])
    index     = [str(i) for i in index]
    ref_full  = list(ref_index[aux.COL_REF_NUC])
    
    # Open selection file and get coefficients
    #df_s = pd.read_csv(inf_file, memory_map=True)
    
    ### RESTRICT THE DATAFRAME TO CONTAIN ONLY THE ENTRIES FOR THE GIVEN REGION  (ALT: MAKE SURE INFERENCE FILENAMES MATCH SEQUENCE FILENAMES)
    #df_s = df_s[df_s['location']==region]
    
    # Open variant association file containing map b/w SNVs and variants
    df_var = pd.read_csv(var_file, memory_map=True)
    
    # Open seq/metadata and sort it by upload date (early to late) and 
    # by collection date (late to early)
    df = pd.read_csv(seq_file, memory_map=True, dtype={aux.COL_SEQ : str}, engine='python')
    df.sort_values(by = [aux.COL_UPLOAD, aux.COL_COLLECT], ascending = [True, False],
                   inplace = True)  # sorting in place to replace unsorted df

    # Get upload dates -- dates should be integer
    upload_dates = np.unique(np.array(df[aux.COL_UPLOAD], int))

    # Open output file, write column names, and initialize
    f = open(out_file, 'w')
    f.write('%s,%s,%s,%s,%s\n' % (aux.COL_TIME, aux.COL_GROUP, aux.COL_ASSOC, aux.COL_MEMBER, aux.COL_MATCH))

    up_date    = upload_dates[0]                        # first upload date
    idx_upload = df[df[aux.COL_UPLOAD]==up_date].index  # find first upload
    df_upload  = df.loc[idx_upload]                     # get first upload
    df.drop(idx_upload, inplace=True)                   # drop from big df

    df_recent = df_upload.head(last_seqs)  # get most recent seqs

    t_min = np.min(np.array(df_recent[aux.COL_UPLOAD])) # get earliest current time

    groups = aux.get_linked_groups(df_recent, ref_file=ref_file, site_file=site_file, last_seqs=last_seqs, min_counts=min_counts)
    if arg_list.mutsIgnore:
        groups = aux.remove_detected_muts(muts_ignore, groups)


    # Get group selection coefficients
    #temp_df_s = df_s[df_s[aux.COL_TIME]==up_date]  # get s at this time only
    #if len(temp_df_s)==0:
    #    print(f'no selection information for time {up_date}')
    #else:
    #    groups_s = get_s(groups, temp_df_s)           # get selection coefficients
    #    if use_selection:
    #         aux.add_detected_muts(muts_detected, groups, groups_s, min_sel=min_sel)
    
    # Get variant-association for each group
    group_var, strong_var, closest_var = aux.get_var(groups, df_var, var_min_muts=min_muts, get_closest=True)

    # Write data to file
    #if len(temp_df_s)!=0:
    for i in range(len(groups)):
        f.write('%d,%s,%s,%s,%s\n' % (up_date, ' '.join(groups[i]), ' '.join(group_var[i]), ' '.join(strong_var[i]), closest_var[i]))
    #else:
    #    print(f'no selection data for time {t_min}')

    # Loop through remaining times
    for up_date in upload_dates[1:]:
        if up_date < start_date:
            continue
        print2(f'working on time {up_date}')

        # Get next uploads
        idx_upload = df[(df[aux.COL_UPLOAD]==up_date) &
                        (df[aux.COL_COLLECT]>=t_min)].index  # next upload RECENT only
        df_upload  = df.loc[idx_upload]             # get next sequences
        df.drop(idx_upload, inplace=True)           # drop from big df

        # Get most recent
        df_recent = pd.concat([df_recent, df_upload])
        df_recent = df_recent.sort_values(by=aux.COL_COLLECT, ascending=False).head(last_seqs)

        # Reset t_min so that sequences too far in the past can be auto dropped
        t_min = np.min(np.array(df_recent[aux.COL_COLLECT]))

        # Get sequence counts (MERGE WITH get_linked_groups WHEN WRITTEN)
        if len(df_recent)<last_seqs:
            print(f'there are only {len(df_recent)} sequences uploaded before {up_date}')
            continue

        # Get linked groups
        groups = aux.get_linked_groups(df_recent, ref_file=ref_file, site_file=site_file, last_seqs=last_seqs, min_counts=min_counts)
        if arg_list.mutsIgnore:
            groups = aux.remove_detected_muts(muts_ignore, groups)

        
        # Get selection and variant association
        #temp_df_s = df_s[df_s[aux.COL_TIME]==up_date]  # get s at this time only
        #if len(temp_df_s)!=0:
        #    groups_s  = aux.get_s(groups, temp_df_s)       # get selection coefficients
        group_var, strong_var, closest_var = aux.get_var(groups, df_var, var_min_muts=min_muts, get_closest=True)
        #    if use_selection:
        #        aux.remove_detected_muts(muts_detected, groups)
        #        aux.add_detected_muts(muts_detected, groups, groups_s, min_sel=min_sel)
        for i in range(len(groups)):
            f.write('%d,%s,%s,%s,%s\n' % (up_date, ' '.join(groups[i]), ' '.join(group_var[i]), ' '.join(strong_var[i]), closest_var[i]))
        #else:
        #    print(f'no selection data for time {t_min}')

    # Close output file
    f.close()
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
