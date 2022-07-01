# LIBRARIES

import numpy as np
import pandas as pd
import datetime as dt
import os
import sys
import argparse
from timeit import default_timer as timer   # timer for performance
#import data_processing as dp
import lmdb

# GLOBAL VARIABLES

NUC = ['-', 'A', 'C', 'G', 'T']
PRO = ['-', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
       'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
REF = NUC[0]
REF_TAG     = 'EPI_ISL_402125'
ALPHABET    = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
NUMBERS     = '0123456789'
TIME_INDEX  = 1

alph_temp   = list(ALPHABET)
alph_new    = []
for i in range(len(alph_temp)):
    for j in range(len(alph_temp)):
        alph_new.append(alph_temp[i] + alph_temp[j])
ALPHABET = alph_new

# SEQUENCE PROCESSING GLOBAL VARIABLES

START_IDX    = 150
END_IDX      = 29690
MAX_GAP_NUM  = 20000
MAX_GAP_FREQ = 0.95
MIN_SEQS     = 0
MAX_DT       = 9999
MAX_AMBIGUOUS = 298


def eliminate_ref_gaps(ref_seq):
    """ Get the reference sequence from the MSA"""

    idxs_keep = []
    ref_index = 0
    ref_alpha = 0
    ref_seq   = list(ref_seq)
    for i in range(len(ref_seq)):
        ref_str = 'NA'
        if ref_seq[i]!='-':
            idxs_keep.append(i)
            ref_index += 1
        elif ref_index==22204:
            idxs_keep.append(i)
        else:
            continue
    idxs_keep = np.array(idxs_keep)
    ref_poly  = np.array(ref_seq)[idxs_keep]
    print(idxs_keep)
    idxs_keep = idxs_keep.astype('int32')
    
    return idxs_keep, ref_poly


def index_ref_seq(ref_seq, out_file):
    """Finds an index for the sites on the unfiltered reference sequence so that they can be found after regions are combined"""
    ref_seq   = list(ref_seq)
    ref_index = 0
    ref_alpha = 0
    f = open(out_file + '.csv')
    f.write('index,ref_index\n')
    for i in range(len(reg_seq)):
        ref_str = 'NA'
        if ref_seq[i]!='-':
            ref_str = '%d' % ref_index
            ref_index += 1
            ref_alpha  = 0
        else:
            ref_str = '%d%s' % (ref_index-1, ALPHABET[ref_alpha])
            ref_alpha += 1
        ref_sites.append(ref_str)
        f.write(f'{i},{ref_str}\n')
    f.close()


def fromisoformat(isodate):
    """ Transforms a date in the YYYY-MM-DD format into a datetime object."""
    year  = int(isodate[:4])
    month = int(isodate[5:7])
    day   = int(isodate[8:])
    return dt.date(year, month, day)


def main():
    """ Given a region to be analyzed, collects the sequences from that regions and in the given time frames
    and then filters out sites that have too many gaps."""
    
    parser = argparse.ArgumentParser(description='filtering out sequences and sites with too many gaps')
    parser.add_argument('--input_file',  type=str,    default=None,                       help='metadata file')
    parser.add_argument('--lmdb',        type=str,    default=None,                       help='path to the database where the sequence information is stored')
    parser.add_argument('-o',            type=str,    default='regional',                 help='output destination')
    parser.add_argument('--regions',     type=str,    default='regional-data.csv.gz',     help='the regions and times to extract data from')
    
    arg_list = parser.parse_args(sys.argv[1:])
    out_folder    = arg_list.o
    input_file    = arg_list.input_file
    lmdb_dir      = arg_list.lmdb
    
    start_time = timer()
    if lmdb_dir is None:
        print('no path to a sequence database was passed. Pass the "--lmdb" flag.')
        sys.exit()
    
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    #genome_dir = os.path.join(out_folder, "genome-data")
    #if not os.path.exists(genome_dir):
    #    os.mkdir(genome_dir)
    out_dir = out_folder
    
    # Choose sequence subsets for analysis
    ref_date   = fromisoformat('2020-01-01')  # reference starting date ('day 0')
    identifier = out_folder
    selected   = np.load(arg_list.regions, allow_pickle=True)
    
    reg_name    = '-'.join(selected[0][:2])
    status_name = f'genome-status-{reg_name}.csv'
    stat_file   = open(status_name, 'w')
    stat_file.close()

    def print2(*args):
        """ Print the status of the processing and save it to a file."""
        stat_file = open(status_name, 'a+')
        line   = [str(i) for i in args]
        string = '\t'.join(line)
        stat_file.write(string+'\n')
        stat_file.close()
        print(string)
        
    #df = dd.read_csv(input_file)
    df = pd.read_csv(input_file)
    print2('loading MSA finished')
    print2('number of genomes in original MSA', len(list(df['accession'])))

    # Iterate over selection criteria
    conds       = selected[0]
    s_continent = conds[0]
    s_country   = conds[1]
    s_region    = conds[2]
    s_subregion = conds[3]
    date_lower  = conds[4]
    date_upper  = conds[5]
    df_temp     = df.copy()
    
    # Sort by country/state/sub
    print2(f'collecting sequences for region {conds}')
    for s_cond, col in [[s_country, 'country'], [s_region, 'region'], [s_subregion, 'subregion']]:
        if s_cond!=None:
            if isinstance(s_cond, str):
                df_temp = df_temp[df_temp[col]==s_cond]
            elif isinstance(s_cond, type([])):
                df_temp = df_temp[df_temp[col].isin(s_cond)]
            else:
                print('\tUnexpected %s type ' % col, s_cond)
        print(f'number of sequences in region {s_cond}:', len(list(df_temp['accession'])))
    
    # Sort by date
    df_date = [fromisoformat(val) for val in list(df_temp['date'])]
    df_pass = [True for i in range(len(df_temp))]
    
    if date_lower!=None:
        if isinstance(date_lower, str):
            date_lower = fromisoformat(date_lower)
        if isinstance(date_lower, dt.date):
            for i in range(len(df_temp)):
                if df_date[i]-date_lower < dt.timedelta(0):
                    df_pass[i] = False
        else:
            print('\tUnexpected date type ', date_lower)
    
    if date_upper!=None:
        if isinstance(date_upper, str):
            date_upper = fromisoformat(date_upper)
        if isinstance(date_upper, dt.date):
            for i in range(len(df_temp)):
                if date_upper-df_date[i] < dt.timedelta(0):
                    df_pass[i] = False
        else:
            print('\tUnexpected date type ', date_upper)
            
    df_locs = [i                               for i in range(len(df_pass)) if df_pass[i] and int((df_date[i]-ref_date).days)>0]
    df_time = [int((df_date[i]-ref_date).days) for i in range(len(df_pass)) if df_pass[i] and int((df_date[i]-ref_date).days)>0]
    df_temp = df_temp.iloc[df_locs]
    df_temp['date'] = df_time
    
    df_sub_date = [fromisoformat(val) for val in list(df_temp['submission_date'])]
    df_sub_date = [int((df_sub_date[i]-ref_date).days) for i in range(len(df_sub_date))]
    df_temp['submission_date'] = df_sub_date
    
    collection_time = timer()
    print2('number of genomes in df_temp', len(list(df_temp['accession'])))
    print2('seconds to collect the relevant sequences:\t', collection_time - start_time)
    
    # Find databases and open enviornments
    db_paths   = []
    db_handles = []    # the actual enviornment where the database is stored
    for file in os.listdir(lmdb_dir):
        path = os.path.join(lmdb_dir, file)
        if os.path.isdir(path):
            db_paths.append(path)
            env = lmdb.open(path, map_size=int(1E12), max_dbs=100, max_readers=2000)
            db_handles.append(env)
            
    extraction_start_time = timer()
    
    sub_label = os.path.split(arg_list.regions)[1]
    out_file  = os.path.join(out_dir, str(sub_label)[:-4] + '.csv')
    f         = open(out_file, mode='w')
    f.write('accession,date,submission_date,sequence\n')
    
    # get reference sequence
    #temp_msa  = []
    #temp_acc  = []
    #temp_date = []
    tags_set    = set()
    ref_present = False
    ref_poly    = ''
    idxs_keep   = []
    for db in db_handles:
        with db.begin() as txn:
            ref = txn.get(REF_TAG.encode())
            if ref:
                #temp_msa.append(ref.decode('utf-8'))
                ref = ref.decode('utf-8')
                idxs_keep, ref = eliminate_ref_gaps(ref)
                f.write(f'{REF_TAG},2020-01-01,2020-01-01,{ref}\n')
                #temp_tag.append(REF_TAG)
                tags_set.add(REF_TAG)
                #temp_date.append('2020-01-01')
                ref_present = True
                break
    if not ref_present:
        print2('reference sequence not found in any database')
    ref_seq = ''.join(list(ref))
    
    # make tags and dictionary mapping accessions to tags
    acc_to_date = {}    # keys are accession IDs and values are tags
    acc_to_subdate = {}
    for i, entry in df_temp.iterrows():
        temp_accession = str(entry['accession'])
        tag = '.'.join([temp_accession, str(entry['date'])])
        if tag not in tags_set:
            tags_set.add(tag)
            acc_to_date[temp_accession] = str(entry['date'])
            acc_to_subdate[temp_accession] = str(entry['submission_date'])
        else:
            print('\tUnexpected duplicate sequence %s' % tag)
    
    # extract sequences from database and compile associated metadata
    keys     = [i.encode() for i in acc_to_date]
    accs_set = set()
    for db in db_handles:
        with db.begin() as txn:
            with txn.cursor() as crs:
                db_results = crs.getmulti(keys)
                db_accs = [i[0].decode('utf-8') for i in db_results]
                db_seqs = [i[1].decode('utf-8') for i in db_results]
                for i in range(len(db_accs)):
                    if db_accs[i] not in accs_set:
                        accs_set.add(db_accs[i])
                        new_seq   = np.array(list(db_seqs[i]))[idxs_keep]
                        num_ambig = np.count_nonzero(new_seq=='N')
                        if num_ambig > MAX_AMBIGUOUS:
                            continue
                        new_seq = ''.join(list(new_seq))
                        f.write(f'{db_accs[i]},{acc_to_date[db_accs[i]]},{acc_to_subdate[db_accs[i]]},{new_seq}\n')
                        #temp_acc.append(db_accs[i])
                        #temp_date.append(acc_to_date[db_accs[i]])
                        #temp_msa.append(db_seqs[i])
    #temp_tag = [REF_TAG] + ['.'.join([temp_acc[i], temp_date[i]]) for i in range(len(temp_acc))]
    f.close()
    
    extraction_end_time = timer()
    print2(f'took {(extraction_end_time - extraction_start_time) / 60} minutes to get sequences from database')
    
    
    

    


if __name__ == '__main__':
    main()
