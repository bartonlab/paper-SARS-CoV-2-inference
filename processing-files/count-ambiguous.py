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
REF_TAG = 'EPI_ISL_402125'
ALPHABET    = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
NUMBERS     = '0123456789'
TIME_INDEX  = 1

alph_temp   = list(ALPHABET)
alph_new    = []
for i in range(len(alph_temp)):
    for j in range(len(alph_temp)):
        alph_new.append(alph_temp[i] + alph_temp[j])
ALPHABET    = alph_new

# SEQUENCE PROCESSING GLOBAL VARIABLES

START_IDX    = 150
END_IDX      = 29690
MAX_GAP_NUM  = 20000
MAX_GAP_FREQ = 0.95
MIN_SEQS     = 0
MAX_DT       = 9999
MAX_AMBIG    = 298


def fromisoformat(isodate):
    """ Transforms a date in the YYYY-MM-DD format into a datetime object."""
    if len(isodate)==10:
        year  = int(isodate[:4])
        month = int(isodate[5:7])
        day   = int(isodate[8:])
    else:
        temp  = isodate.split('-')
        year  = int(temp[0])
        month = int(temp[1])
        day   = int(temp[2])
    return dt.date(year, month, day)


def count_ambiguous(msa, tag):
    """count the number of ambiguous nucleotides in an msa"""
    
    #ref_df      = pd.read_csv('ref-index.csv')
    #ref_indices = list(ref_df['ref_index'])
    #ref_seq     = list(ref_df['nucleotide'])
    #ref_idx = tag.index(REF_TAG)
    #ref_seq = list(msa[ref_idx])
    
    ambig_chars = ['R', 'Y', 'K', 'M', 'S', 'W']
    num_ns      = 0
    num_ambig   = 0
    
    ref_seq = list(msa[list(tag).index(REF_TAG)])
    
    kept_indices = []
    ref_index    = 0
    for i in range(len(ref_seq)):
        if ref_seq[i]!='-':
            ref_index += 1
            if ref_index > START_IDX and ref_index < END_IDX:
                kept_indices.append(i)

            
    """
    for i in range(len(ref_seq)):
        temp_ref_index = list(ref_indices[i])
        if temp_ref_index[0]=='-':
            continue
        if temp_ref_index[-1] in list(NUMBERS):
            ref_index = int(ref_indices[i])
        elif temp_ref_index[-2] in list(NUMBERS):
            ref_index = int(ref_indices[i][:-1])
        else:
            ref_index = int(ref_indices[i][:-2])
        if ref_seq[i]=='-' and ref_index!=22203:    # eliminates sites that are gaps in the reference sequence except the insertion in Omicron
            continue
        if ref_index > START_IDX and ref_index < END_IDX:
            kept_indices.append(i)
    """
            
    gaps_seq = []
    ns_seq = []    # the number of ambiguous nucleotides in each sequence
    for i in range(len(msa)):
        temp_seq = np.array(list(msa[i]))[np.array(kept_indices)]
        for j in ambig_chars:
            count = np.count_nonzero(temp_seq==j)
            num_ambig += count
        count      = np.count_nonzero(temp_seq=='N')
        if count>MAX_AMBIG:
            continue
        num_ambig += count
        num_ns    += count
        ns_seq.append(count)
        gap_count  = np.count_nonzero(temp_seq=='-')
        gaps_seq.append(gap_count)
    
    return num_ns, num_ambig, ns_seq, gaps_seq



def main():
    """ Given a region to be analyzed, collects the sequences from that regions and in the given time frames
    and then filters out sites that have too many gaps."""
    
    parser = argparse.ArgumentParser(description='filtering out sequences and sites with too many gaps')
    parser.add_argument('--input_file',  type=str,    default=None,                       help='metadata file')
    parser.add_argument('-o',            type=str,    default='regional',                 help='output destination')
    parser.add_argument('--regions',     type=str,    default='regional-data.csv.gz',     help='the directory containing files that specify the regions and times')
    
    arg_list = parser.parse_args(sys.argv[1:])
    out_file      = arg_list.o
    input_file    = arg_list.input_file
    
    start_time = timer()
    
    # separate location information
    df = pd.read_csv(input_file)
    locs = list(df['location'])
    if 'virus_name' in df:
        virus_name = list(df['virus_name'])
        virus_name = [i.split('/') for i in virus_name]
        locs2      = []
        for i in range(len(virus_name)):
            if len(virus_name[i])>2:
                locs2.append(virus_name[i][2])
            else:
                print(len(virus_name[i]))
                locs2.append('')
        locs_new   = []
        """
        for i in range(len(locs)):
            l = locs[i].split(' / ')
            m = locs2[i].split('-')[:-1]
            m = [j.lower() for j in m]
            if len(l) == 1:
                l.append('NA')
                l.append('NA')
                l.append('/'.join(m))
            if len(l) == 2:
                l.append('NA')
                l.append('/'.join(m))
            elif len(l) == 3:
                l.append('/'.join(m))
            elif len(l) == 4:
                l[3] += '/' + '/'.join(m)
            #else:
                #print('there is too much location information', len(l), len(m))
            locs_new.append(l)
        """
        for i in range(len(locs)):
            l = locs[i].split(' / ')
            m = locs2[i].split('-')[:-1]
            m = [j.lower() for j in m]
            if len(l) == 1:
                l.append('NA')
                l.append('NA')
                l.append('NA')
                l.append('/'.join(m))
            if len(l) == 2:
                l.append('NA')
                l.append('NA')
                l.append('/'.join(m))
            elif len(l) == 3:
                #l.append('NA')    # Edited to make london sequences location appear in the correct slot. 
                l.append('/'.join(m))
            elif len(l) == 4:
                l.append('/'.join(m))
            elif len(l) == 5:
                l[4] += '/' + '/'.join(m)
            #else:
                #print('there is too much location information', len(l), len(m))
            locs_new.append(l)
    else:
        locs_new = [locs[i].split(' / ') for i in range(len(locs))]
    # separate the different location data and put it in separate columns.
    continent    = []
    country      = []
    region       = []
    subregion    = []
    subsubregion = []
    for l in locs_new:
        #l = l.split(' / ')
        if len(l)>0:
            continent.append(l[0])
        else:
            continent.append('NA')
        
        if len(l)>1:
            if l[1] == 'houston':
                country.append('usa')
                region.append('texas')
                subregion.append('houston')
                #subsubregion.append('NA')
                continue
            else:
                country.append(l[1])
        else:
            country.append('NA')
        
        if len(l)>2:
            region.append(l[2])
        else:
            region.append('NA')
        if len(l)>3:
            subregion.append(l[3])
        else:
            subregion.append('NA')
        if len(l)>4:
            subsubregion.append('/'.join(l[3:]))
        else:
            subsubregion.append('NA')
        """
        if len(l)>3:
            subregion.append('/'.join(l[3:]))
        else:
            subregion.append('NA')  
        """
    
    df  = df.drop('location', axis=1)
    df2 = pd.DataFrame.from_dict({'continent' : continent, 'country' : country, 
                                  'region' : region, 'subregion' : subregion,
                                  'subsubregion' : subsubregion})
    df = pd.concat([df, df2], axis=1)
    
    # Find regions and times that are analyzed
    ref_date   = fromisoformat('2020-01-01')  # reference starting date ('day 0')
    selected   = []
    for file in os.listdir(arg_list.regions):
        selected.append(np.load(os.path.join(arg_list.regions, file), allow_pickle=True)[0])
        
    
    status_name = f'count-ambiguous-status.csv'
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
    print2('number of genomes in original MSA', len(list(df['accession'])))
    ref_seq = df[df.accession==REF_TAG].iloc[0].sequence
    ref_row = {'accession' : REF_TAG, 'sequence' : ref_seq}

    num_ns    = 0
    num_ambig = 0
    num_seqs  = 0
    ns_seq    = []    # the number of ambiguous nucleotides in each sequence
    gaps_seq  = []
    ref_gaps  = 0
    # Iterate over selection criteria
    for conds in selected:
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
        df_temp = df_temp.append(ref_row, ignore_index=True)
        
        ns, ambig, ns_seq_temp, gaps_seq_temp  = count_ambiguous(list(df_temp['sequence']), list(df_temp['accession']))
        num_ns    += ns
        num_ambig += ambig
        num_seqs  += len(df_temp)
        for i in ns_seq_temp:
            ns_seq.append(i)
        for i in gaps_seq_temp:
            gaps_seq.append(i)
        
    f = open(out_file + '.npz', mode='wb')
    np.savez_compressed(f, number_ambiguous=num_ambig, number_ns=num_ns, number_sequences=num_seqs, ns_sequence=ns_seq, gap_distribution=gaps_seq)
    f.close()
    print2(f"number of NNN's in all data is {num_ns}")
    print2(f'total number of ambiguous nucleotides in all data is {num_ambig}')


if __name__ == '__main__':
    main()
