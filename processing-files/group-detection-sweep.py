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
    """Get detected linked groups over a range of threshold values."""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',           type=str,   default=None,    help='output string')
    #parser.add_argument('--inSeqs',     type=str,   default=None,    help='input sequence file')
    parser.add_argument('--inGroups',   type=str,   default=None,    help='input linked groups file')
    #parser.add_argument('--inInf',      type=str,   default=None,    help='input inference file containing inferred selection coefficients over time for the region')
    parser.add_argument('--infDir',     type=str,   default=None,    help='inference directory containing inferred selection coefficient file over time for each region')
    parser.add_argument('-q',           type=int,   default=5,       help='the number of states at each site')
    parser.add_argument('--threshMin',  type=float, default=0.005,   help='minimum detection threshold')
    parser.add_argument('--threshMax',  type=float, default=0.250,   help='maximum detection threshold')
    parser.add_argument('--threshStep', type=float, default=0.005,   help='detection threshold step size')
    parser.add_argument('--variants',   type=str,   default='alpha', help='list of variants to determine first detection times, enclosed in quotes and separated by spaces')
    parser.add_argument('--saveGroups', default=False, action='store_true', help='whether or not to save groups of mutations that are detected')
    parser.add_argument('--elimSites',  default=False, action='store_true', help='whether or not to eliminate all nucleotide mutations at a site once any of them have been detected')
    parser.add_argument('--closestVar', default=False, action='store_true', help='whether or not to use the closest associated variant for a group instead of all groups that it is strongly associated with')
    parser.add_argument('--allGroups',  default=False, action='store_true', help='if true, saves all groups, detected or not. if false, only saves detected groups')

    
    arg_list = parser.parse_args(args)
    
    # Selection threshold parameters
    thresh_min  = arg_list.threshMin
    thresh_max  = arg_list.threshMax
    thresh_step = arg_list.threshStep
    thresh_vals = np.arange(thresh_min, thresh_max+thresh_step, thresh_step)
    
    # List of variants
    variants = [v.lower() for v in arg_list.variants.split()]
    
    # Input/output files
    out_str  = arg_list.o
    #seq_file = arg_list.inSeqs
    grp_file = arg_list.inGroups
    inf_dir  = arg_list.infDir
    #inf_file = arg_list.inInf
    q        = arg_list.q

    save_groups = arg_list.saveGroups
    
    # Get region
    region   = aux.find_region(grp_file)
    inf_file = os.path.join(inf_dir, region + '.csv')
    out_file = os.path.join(out_str, os.path.split(grp_file)[-1])
    
    print(region)
    
    # Open selection file and get coefficients
    df_s = pd.read_csv(inf_file, memory_map=True)
    
    ### RESTRICT THE DATAFRAME TO CONTAIN ONLY THE ENTRIES FOR THE GIVEN REGION  (ALT: MAKE SURE INFERENCE FILENAMES MATCH SEQUENCE FILENAMES)
    #df_s = df_s[df_s['location']==region]
    
    # Open linked groups file and list of times
    df = pd.read_csv(grp_file, memory_map=True, dtype={aux.COL_MEMBER : str})
    
    print(aux.COL_TIME)
    print(df.columns)
    
    times     = np.unique(np.array(list(df[aux.COL_TIME]), int))
    selection = aux.make_selection_dict(df, df_s)
    s_times   = np.unique(df_s[aux.COL_TIME])

    # Open output file, write column names, and initialize
    f = open(out_file, 'w')
    f.write('%s,%s,%s,%s\n' % (aux.COL_THRESH, aux.COL_TP, aux.COL_FP, ','.join(variants)))

    if save_groups:
        grps_out = open(out_file[:-4] + '-groups.csv', 'w')
        grps_out.write(f'positive,{aux.COL_THRESH},{aux.COL_GROUP},{aux.COL_S},{aux.COL_MATCH},{aux.COL_ASSOC},{aux.COL_MEMBER},time\n')
    
    # ITERATE
    for thresh in thresh_vals:
        print(f'working on threshold {thresh}')
    
        muts_detected  = set()                                 # detected mutations
        first_detected = [9999 for i in range(len(variants))]  # first detection times
        true_pos       = 0
        false_pos      = 0
        
        for t in times:
            if t not in s_times:
                print(f'no selection information for time {t}')
                missing_data = True
            else:
                missing_data = False
            temp_df   = df[df[aux.COL_TIME]==t]
            #temp_df_s = df_s[df_s[aux.COL_TIME]==t]
            
            for idx, row in temp_df.iterrows():
                temp_group = set(str(row[aux.COL_GROUP]).split()).difference(muts_detected)
                #temp_s     = aux.get_s_set()
                temp_s     = aux.get_s_dict(temp_group, selection, t)
                
                # Check for detection -- GOOD PLACE FOR VERBOSE OUTPUT
                if temp_s>thresh:
                    
                    # False positive -- blank variant association
                    if pd.isnull(row[aux.COL_ASSOC]):
                        false_pos += 1

                        if arg_list.allGroups:
                            #try:
                            #    vars_temp = [v.lower() for v in row[aux.COL_MEMBER].split()]
                            #except:
                            #    print(row[aux.COL_MEMBER])

                            if arg_list.closestVar:
                                closest_var = row[aux.COL_MATCH]
                                vars_temp   = [closest_var]

                            col_assoc = row[aux.COL_ASSOC]
                            col_mem   = row[aux.COL_MEMBER]
                            grps_out.write(f"True,{thresh:.4f},{' '.join(list(temp_group))},{temp_s:.6f},{closest_var},{col_assoc},{col_mem},{t}\n")

                        
                    # True positive
                    else:
                        true_pos += 1


                        # Check for tight association to record first detection time
                        if not pd.isnull(row[aux.COL_MEMBER]):
                            vars_temp = [v.lower() for v in row[aux.COL_MEMBER].split()]

                            if arg_list.closestVar:
                                closest_var = row[aux.COL_MATCH]
                                vars_temp   = [closest_var]

                            if save_groups:
                                col_assoc = row[aux.COL_ASSOC]
                                col_mem   = row[aux.COL_MEMBER]
                                grps_out.write(f"True,{thresh:.4f},{' '.join(list(temp_group))},{temp_s:.6f},{closest_var},{col_assoc},{col_mem},{t}\n")

                            for v in vars_temp:
                                if v in variants:
                                    idx = variants.index(v)
                                    if t<first_detected[idx]:
                                        first_detected[idx] = t

                        elif save_groups:
                            grps_out.write(f"True,{thresh:.4f},{' '.join(list(temp_group))},{temp_s:.6f},None,None,None,{t}\n")
                    
                    # Add to detected mutations
                    if arg_list.elimSites:
                        full_group = aux.add_all_nucs(temp_group)
                    muts_detected  = muts_detected.union(full_group)

                # Negatives
                elif arg_list.allGroups:
                    if arg_list.closestVar:
                        closest_var = row[aux.COL_MATCH]

                    if save_groups:
                        col_assoc = row[aux.COL_ASSOC]
                        col_mem   = row[aux.COL_MEMBER]
                        grps_out.write(f"False,{thresh:.4f},{' '.join(list(temp_group))},{temp_s:.6f},{closest_var},{col_assoc},{col_mem},{t}\n")

                    
        # Record results for this threshold value
        print(thresh, true_pos, false_pos, ','.join([str(i) for i in first_detected]))
        f.write('%s,%s,%s,%s\n' % (thresh, true_pos, false_pos, ','.join([str(i) for i in first_detected])))

        if missing_data:
            print(f'selection times: {s_times}\nlinked groups times: {times}')

    # Close output file
    f.close()
    if save_groups:
        grps_out.close()
    
    
if __name__ == '__main__': 
    main(sys.argv[1:])
