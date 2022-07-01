# LIBRARIES

import numpy as np
import pandas as pd
import datetime as dt
import os
import sys
import argparse
from timeit import default_timer as timer   # timer for performance
import data_processing as dp

# GLOBAL VARIABLES
REF_TAG = 'EPI_ISL_402125'
TIME_INDEX = 1


status_name = 'genome-time-intervals-status.csv'
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


def order_sequences(tag):
    """ Put sequences in time order. """

    tag = list(tag)
    
    ref_idx = tag.index(REF_TAG)
    del tag[ref_idx]
    
    temp_tag = [REF_TAG] #, CONS_TAG]
    tag, times = get_times(tag, sort=True)
    
    return np.array(temp_tag + list(tag)), np.array([str(1)] + list(times))

    
def get_times(tag, sort=False):
    """Return sequences and times collected from an input MSA and tags (optional: time order them)."""

    times = []
    for i in range(len(tag)):
        if tag[i] not in [REF_TAG]: #, CONS_TAG]:
            tsplit = tag[i].split('.')
            times.append(int(tsplit[TIME_INDEX]))
        else:
            times.append(-1)
    
    if sort:
        t_sort = np.argsort(times)
        return np.array(tag)[t_sort], np.array(times)[t_sort]

    else:
        return np.array(times)


def fromisoformat(isodate):
    """ Transforms a date in the YYYY-MM-DD format into a datetime object."""
    year  = int(isodate[:4])
    month = int(isodate[5:7])
    day   = int(isodate[8:])
    return dt.date(year, month, day)


def toisoformat(days):
    """Given a number of days of january first, print isoformat date"""
    days  = int(days)
    date  = dt.date(2020,1,1) + dt.timedelta(days)
    year  = str(date.year)
    month = str(date.month)
    day   = str(date.day)
    if len(month)==1:
        month = '0' + month
    if len(day)==1:
        day = '0' + day
    return year + '-' + month + '-' + day


def main():
    """ Given the desired regions to be analyzed, collects the sequences from those regions and in the given time frames and then filters out 
    sequences and sites that have too many gaps."""
    
    
    parser = argparse.ArgumentParser(description='filtering out sequences and sites with too many gaps')
    parser.add_argument('--input_file',  type=str,    default=None,                       help='sequence alignment file')
    parser.add_argument('-o',            type=str,    default='regions',                  help='output destination')
    parser.add_argument('--regions',     type=str,    default='regional-data.csv.gz',     help='the regions and times to extract data from')
    parser.add_argument('--maxSeqs',     type=int,    default=50000,                     help='maximum number of sequences in a file')
    parser.add_argument('--find_syn_off', default=False,  action='store_true',            help='whether or not to determine which sites are synonymous and not')
    parser.add_argument('--full_index',   default=False,  action='store_true',            help='whether or not to calculate the full index for sites')
    parser.add_argument('--no_trim',      default=False,  action='store_true',            help='whether or not to trim sites based on frequency')
    
    arg_list = parser.parse_args(sys.argv[1:])
    out_dir      = arg_list.o
    input_file   = arg_list.input_file
    find_syn_off = arg_list.find_syn_off
    full_index   = arg_list.full_index
    maxSeqs      = arg_list.maxSeqs
    
    start_time = timer()
    
    df = pd.read_csv(input_file, usecols=['accession', 'virus_name', 'date', 'location'])
    print2('loading MSA finished')
    print2('number of genomes in original MSA', len(list(df.accession)))
    
    locs     = list(df.location)
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
                subsubregion.append('NA')
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
    
    df = df.drop('location', axis=1)
    #df2 = pd.DataFrame.from_dict({'continent' : continent, 'country' : country, 
    #                                  'region' : region, 'subregion' : subregion, 'subsubregion' : subsubregion})
    df2 = pd.DataFrame.from_dict({'continent' : continent, 'country' : country, 
                                  'region' : region, 'subregion' : subregion,
                                  'subsubregion' : subsubregion})
    df = pd.concat([df, df2], axis=1)
    
    # Choose sequence subsets for analysis
    ref_date   = fromisoformat('2020-01-01')  # reference starting date ('day 0')
    selected   = np.load(arg_list.regions, allow_pickle=True)

    # Iterate over selection criteria
    counter = 0 
    for conds in selected:
        # Read in conditions
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
            print(f'number of sequences in region {s_cond}:', len(list(df_temp.accession)))
    
        # Sort by date
        df_date   = [fromisoformat(val) for val in list(df_temp['date'])]
        df_time   = [int((df_date[i]-ref_date).days) for i in range(len(df_date)) if int((df_date[i]-ref_date).days)>0]
        df_temp   = df_temp.reset_index(drop=True)
        idxs_drop = [i for i in range(len(df_date)) if int((df_date[i]-ref_date).days)<=0]
        df_temp   = df_temp.drop(index=idxs_drop)
        df_temp['date'] = df_time
    
        if counter == 0:
            df_new = df_temp.copy()
        else:
            df_new = df_new.append(df_temp, ignore_index=True)
        counter += 1
    
    collection_time = timer()
    print2('number of genomes in df_new', len(list(df_new.accession)))
    print2('seconds to collect the relevant sequences:\t', collection_time - start_time)
    # Compile sequences and tags
    temp_tag = [REF_TAG]
    tag_set  = {REF_TAG}
    idx_drop = []
    counter  = 0
    msa_time = timer()
    for i, entry in df_new.iterrows():
        counter += 1
        if counter % 100000 == 0:
            msa_time_new = timer()
            print2(f'took {msa_time_new - msa_time} seconds to add {counter} sequences out of {len(df)}')
            msa_time = timer()
        tag = '.'.join([str(entry.accession), str(entry.date)])
        if tag not in tag_set:
            temp_tag.append(tag)
            tag_set.add(tag)
        else:
            print2('\tUnexpected duplicate sequence %s' % tag)
            idx_drop.append(i)
            
    ref_row = pd.DataFrame({'date' : ref_date, 'accession' : REF_TAG, 
                           'continent' : 'asia', 'country': 'china', 'region' : 'wuhan'}, index=[0])
    
    if len(idx_drop)>0:
        df_new = df_new.drop(labels=idx_drop, axis=0)
    df_new  = pd.concat([ref_row, df_new[:]], sort=False).reset_index(drop = True)
    df      = df_new.drop(0, axis=0)
    
    old_time = timer()
    
    region_names = []
    sub_label    = []
    for conds in selected:
        # Read in conditions
        s_continent = conds[0]
        s_country   = conds[1]
        s_region    = conds[2]
        s_subregion = conds[3]
        date_lower  = conds[4]
        date_upper  = conds[5]
        df_temp     = df.copy()
        
        region_start_time = timer()
        # Data processing, time ordering, and quality checks
        ## - put sequences in time order (first entry is reference sequence)
        print2('\ttime-ordering sequences...')
        #if s_region == 'england_wales_scotland':
        #    dates = ['2020-01-02', '2020-11-03', '2021-01-19', '2021-03-09', '2021-05-13', '2021-06-26', '2021-07-23', '2021-08-16',
        #             '2021-09-05', '2021-09-24', '2021-10-13', '2021-10-31', '2021-11-15', '2021-11-28', '2021-12-09', '2021-12-21',
        #             '2022-01-04', '2022-01-13', '2022-01-22', '2022-01-31', '2022-02-08', '2022-02-17', '2022-02-28', '2022-03-09', 
        #             '2022-03-17', '2022-03-27', '2022-04-30']
        #else:
        #    temp_tag, temp_times = order_sequences(temp_tag)
        temp_tag, temp_times = order_sequences(temp_tag)
        region_timeorder_time = timer()
        print2('\ttime-ordering took %d seconds' % (region_timeorder_time - region_start_time))
        
        ## - write region names to files separated by number of sequences
        #if s_subregion == None:
        #    s_subregion=='None'
        if len(temp_tag) < maxSeqs:
            region_names.append([s_continent, s_country, s_region, s_subregion, date_lower, date_upper])
        else:
            L     = len(temp_tag)
            if not s_region == 'england_wales_scotland':
                dates = [temp_times[i*maxSeqs] for i in range(int(L / maxSeqs) + 1)] + [temp_times[-1]]
                dates = [toisoformat(i) for i in dates]
            for i in range(len(dates) - 1):
                region_names.append([s_continent, s_country, s_region, s_subregion, dates[i], dates[i+1]])
                
    for group in region_names:
        new = np.array(group)[np.array(group)!=None]
        contains_list = False
        for i in new:
            if not isinstance(i, str):
                contains_list = True
        if contains_list:
            label = ''
            for i in new:
                if isinstance(i, str):
                    label += i
                else:
                    if len(i) > 4:
                        #print(i)
                        if new[2]=='california':
                            label += 'south'
                    else:
                        label += '_'.join(i)
                if i!=new[-1]:
                    label += '-'
        else:
            label = '-'.join(new)
        f = open(os.path.join(out_dir, label + '.npy'), mode='wb')
        np.save(f, [group])
        f.close()
    
    
    


if __name__ == '__main__':
    main()