# LIBRARIES

import numpy as np
import pandas as pd
import datetime as dt
import os
import sys
import argparse
from timeit import default_timer as timer
import multiprocessing as mp
from itertools import repeat

# GLOBAL VARIABLES

NUC = ['-', 'A', 'C', 'G', 'T']
PRO = ['-', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
       'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
REF = NUC[0]
REF_TAG       = 'EPI_ISL_402124'
ALPHABET      = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
NUMBERS       = '0123456789'
TIME_INDEX    = 1

alph_temp   = list(ALPHABET)
alph_new    = []
for i in range(len(alph_temp)):
    for j in range(len(alph_temp)):
        alph_new.append(alph_temp[i] + alph_temp[j])
ALPHABET    = alph_new

# SEQUENCE PROCESSING GLOBAL VARIABLES

START_IDX     = 150
END_IDX       = 29690
MAX_AMBIGUOUS = 297
MAX_GAP_NUM   = 150
MAX_GAP_FREQ  = 0.9
MIN_SEQS      = 0
MAX_DT        = 9999


def order_sequences(msa, tag):
    """ Put sequences in time order. """

    msa = list(msa)
    tag = list(tag)
    
    ref_idx = tag.index(REF_TAG)
    ref_seq = msa[ref_idx]
    del msa[ref_idx]
    del tag[ref_idx]
    
    temp_msa = [ref_seq] #, cons_seq]
    temp_tag = [REF_TAG] #, CONS_TAG]
    msa, tag, temp = get_times(msa, tag, sort=True)
    
    return np.array(temp_msa + list(msa)), np.array(temp_tag + list(tag))


def impute_ambiguous(msa, tag, gaps=None, ref_idxs=None, start_index=0, verbose=False, impute_edge_gaps=False):
    """ Impute ambiguous nucleotides with the most frequently observed ones in the alignment. """
    
    # Find site numbers that don't have known deletions
    #mask     = ~np.isin(ref_idxs, gaps)
    #col_idxs = np.arange(len(ref_idxs))[mask]
    
    # Impute ambiguous nucleotides
    avg = np.zeros((len(msa[0]), len(NUC)))
    for i in range(len(msa)):
        for j in range(len(msa[0])):
            if msa[i][j] in NUC:
                avg[j, NUC.index(msa[i][j])] += 1
    
    #print('%d ambiguous nucleotides across %d sequences' % (np.sum(ambiguous), len(msa)))
    
    for i in range(start_index, len(msa)):
        ambiguous = np.isin(list(msa[i]), NUC, invert=True)
        idxs      = np.arange(len(msa[i]))[ambiguous]
        for j in idxs:
            orig = msa[i][j]
            new  = NUC[np.argmax(avg[j])]
            if orig=='R': # A or G
                if avg[j][NUC.index('A')]>avg[j][NUC.index('G')]:
                    new = 'A'
                else:
                    new = 'G'
            elif orig=='Y': # T or C
                if avg[j][NUC.index('T')]>avg[j][NUC.index('C')]:
                    new = 'T'
                else:
                    new = 'C'
            elif orig=='K': # G or T
                if avg[j][NUC.index('G')]>avg[j][NUC.index('T')]:
                    new = 'G'
                else:
                    new = 'T'
            elif orig=='M': # A or C
                if avg[j][NUC.index('A')]>avg[j][NUC.index('C')]:
                    new = 'A'
                else:
                    new = 'C'
            elif orig=='S': # G or C
                if avg[j][NUC.index('G')]>avg[j][NUC.index('C')]:
                    new = 'G'
                else:
                    new = 'C'
            elif orig=='W': # A or T
                if avg[j][NUC.index('A')]>avg[j][NUC.index('T')]:
                    new = 'A'
                else:
                    new = 'T'
            msa[i][j] = new
            if verbose:
                print('\texchanged %s for %s in sequence %d, site %d' % (new, orig, i, j))

    return msa


def create_index(msa, tag, ref_seq, ref_idxs, ref_start, min_seqs, max_dt, out_file, return_polymorphic=True, return_truncated=True):
    """ Create a reference to map between site indices for the whole alignment, polymorphic sites only, and the
        reference sequence. To preserve quality, identify last time point such that all earlier time points have at
        least min_seqs sequences and maximum time gap of max_dt between samples. Return the list of polymorphic sites. """

    msa = list(msa)
    tag = list(tag)
    
    ref_idx = tag.index(REF_TAG)
    ref_seq = msa[ref_idx]
    del msa[ref_idx]
    del tag[ref_idx]
    
    f = open('%s' % out_file, 'w')
    f.write('alignment,polymorphic,reference_index,reference_nucleotide\n')
    
    # Check for minimum number of sequences/maximum dt to truncate alignment
    temp_msa = msa
    temp_tag = tag

    ref_index = ref_start
    ref_alpha = 0
    polymorphic_index = 0
    polymorphic_sites = []
    ref_sites         = []
    for i in range(len(temp_msa[0])):
        
        # Index polymorphic sites
        poly_str = 'NA'
        if np.sum([s[i]==temp_msa[0][i] for s in temp_msa])<len(temp_msa):
            poly_str = '%d' % polymorphic_index
            polymorphic_index += 1
            polymorphic_sites.append(i)
        
        # Index reference sequence
        ref_str = 'NA'
        if ref_seq[i]!='-':
            ref_str = '%d' % ref_index
            ref_index += 1
            ref_alpha  = 0
        else:
            ref_str = '%d%s' % (ref_index-1, ALPHABET[ref_alpha])
            ref_alpha += 1
        ref_sites.append(ref_idxs[i])

        # Save to file
        f.write('%d,%s,%s,%s\n' % (i, poly_str, ref_str, ref_seq[i]))
    f.close()

    temp_msa = [ref_seq] + list(temp_msa)
    temp_tag = [REF_TAG] + list(temp_tag)

    # Change this to be able to deal with insertions/deletions
    ref_seq_poly = [str(NUC.index(a)) for a in np.array(ref_seq)[np.array(polymorphic_sites)]]
    ref_sites    = np.array(ref_sites)[polymorphic_sites]
    
    if return_polymorphic and return_truncated:
        return ref_sites, polymorphic_sites, temp_msa, temp_tag, ref_seq_poly
    elif return_polymorphic:
        return polymorphic_sites
    elif return_truncated:
        return temp_msa, temp_tag


def save_alignment(msa, tag, out_file, poly_sites, ref_sites):
    """ Save the sequences and metadata in a .csv file"""
    sub_time_idx  = 2
    accession_idx = 0
    
    idxs      = [i for i in range(len(msa)) if tag[i]!=REF_TAG]
    temp_msa  = np.array(msa)[idxs]
    temp_tag  = np.array(tag)[idxs]
    times     = get_times(temp_msa, temp_tag, sort=False)
    sub_times = np.array([int(i.split('.')[sub_time_idx])  for i in temp_tag])
    accession = np.array([i.split('.')[accession_idx]      for i in temp_tag])
    new_msa   = [''.join([str(NUC.index(a)) for a in temp_msa[i][poly_sites]]) for i in range(len(temp_msa))]
    ref_sites = [i for i in ref_sites]
    mut_sites = [i for i in poly_sites]
    
    data = {
        'sequence' : new_msa, 
        'date' : times, 
        'submission date' : sub_times, 
        'accession' : accession
        }
    df = pd.DataFrame(data=data)
    df.to_csv(out_file + '.csv', index=False)
    
    sites_data = {'mutant_sites' : mut_sites, 'ref_sites' : ref_sites}
    df2 = pd.DataFrame(data=sites_data)
    df2.to_csv(out_file + '-sites.csv', index=False)


def get_times(msa, tag, sort=False):
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
        return np.array(msa)[t_sort], np.array(tag)[t_sort], np.array(times)[t_sort]
    else:
        return np.array(times)


def fromisoformat(isodate):
    """ Transforms a date in the YYYY-MM-DD format into a datetime object."""
    year  = int(isodate[:4])
    month = int(isodate[5:7])
    day   = int(isodate[8:])
    return dt.date(year, month, day)


def filter_excess_gaps(msa, tag, sequence_max_gaps, site_max_gaps, verbose=True, ref_file=None):
    """ Remove sequences and sites from the alignment which have excess gaps. """
    
    msa = list(msa)
    tag = list(tag)
    
    ref_idx = tag.index(REF_TAG)
    ref_seq = list(msa[ref_idx])
    del msa[ref_idx]
    del tag[ref_idx]
    
    # get unique site index from site labels for the reference sequence
    ref_df = pd.read_csv(ref_file)
    ref_indices = list(ref_df['ref_index'])
    temp_tag    = tag
    temp_msa    = np.array([list(i) for i in msa])
    
    print('sequence lengths')
    n,c = np.unique([len(i) for i in temp_msa], return_counts=True)
    print(n,c)
    
    # Drop sites that have too many gaps
    kept_indices = []
    for i in range(len(ref_seq)):
        if np.sum(temp_msa[:,i]=='-')/len(temp_msa)<site_max_gaps:
            # determine index number on reference sequence
            temp_ref_index = list(ref_indices[i])
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
    print(f'{len(ref_seq) - len(kept_indices)} sites are being removed due to the number of sequences with gaps')
    temp_msa = [np.array(list(ref_seq))[kept_indices]] + [np.array(s)[kept_indices] for s in temp_msa]
    temp_tag = [REF_TAG] + temp_tag
    ref_idxs = np.array(ref_indices)[kept_indices]

    return temp_msa, temp_tag, ref_idxs


def make_impute_intervals(times, counts, min_seqs=50):
    """Find time intervals between which sequences should be taken to impute ambiguous nucleotides"""
    intervals = [times[0]]
    cum_sum   = counts[0]
    for i in range(1,len(times)):
        cum_sum += counts[i]
        if cum_sum >= min_seqs:
            intervals.append(times[i])
            cum_sum = 0
    intervals.pop
    intervals.append(times[-1]+1)
    return intervals


def main():
    """ Given a region to be analyzed, collects the sequences from that regions and in the given time frames
    and then filters out sites that have too many gaps."""
    
    parser = argparse.ArgumentParser(description='filtering out sequences and sites with too many gaps')
    parser.add_argument('--input_file',  type=str,    default=None,                       help='regional .csv file')
    parser.add_argument('-o',            type=str,    default='genome-data',              help='output destination')
    parser.add_argument('--refFile',     type=str,    default='ref-index.csv',            help='reference index file')
    parser.add_argument('--threads',     type=int,    default=1,                          help='the number of threads to use to impute ambiguous nucleotides')
    
    arg_list = parser.parse_args(sys.argv[1:])
    out_folder    = arg_list.o
    input_file    = arg_list.input_file
    
    sub_label = [os.path.split(input_file)[1][:-4]]
    sys.path.append(os.getcwd())
    
    start_time = timer()
    
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    genome_dir = out_folder
    if not os.path.exists(genome_dir):
        os.mkdir(genome_dir)
    out_dir = out_folder
    
    # Choose sequence subsets for analysis
    ref_date   = fromisoformat('2020-01-01')  # reference starting date ('day 0')
    
    reg_name    = sub_label[0]
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
    df_temp = pd.read_csv(input_file, engine='python', dtype={'sequence' : str})
    #df_temp = df_temp.drop(index=[0])
    ref_idx = index=list(df_temp.index[df_temp['accession']==REF_TAG])
    print(f'reference index in dataframe {ref_idx}')
    print(f'size of original dataframe {len(df_temp)}')
    if len(ref_idx)!=0:
        df_temp = df_temp.drop(index=list(df_temp.index[df_temp['accession']==REF_TAG]))    # drop reference sequence from MSA
    df_temp = df_temp.dropna(axis=0)
    print2(input_file)
    print2('loading MSA finished')
    print2('number of genomes in original MSA', len(list(df_temp['accession'])))
    start_time = timer()
    
    ref_seq  = ''.join(list(pd.read_csv(arg_list.refFile)['nucleotide']))
    ref_row  = pd.DataFrame(columns=['sequence', 'accession', 'date'])
    ref_row.loc[0] = [ref_seq, REF_TAG, 0]
    df_temp  = pd.concat([ref_row, df_temp], ignore_index=True)    
    df_temp = df_temp.reset_index(drop=True)
    
    ref_seq  = list(df_temp[df_temp.accession==REF_TAG].iloc[0].sequence)
    temp_msa = [ref_seq]
    temp_tag = [REF_TAG]
    for i in range(len(df_temp)):
        tag = '.'.join([str(df_temp.iloc[i].accession), str(df_temp.iloc[i].date), str(df_temp.iloc[i].submission_date)])
        if tag not in temp_tag and str(df_temp.iloc[i].accession)!=REF_TAG:
            temp_tag.append(tag)
            temp_msa.append(list(df_temp.iloc[i].sequence))
    temp_msa = np.array(temp_msa)
    print('msa length', len(temp_msa))
    print('tag length', len(temp_tag))
    if len(temp_msa)<=5:
        sys.exit()
    
    temp_msa, temp_tag, ref_idxs = filter_excess_gaps(temp_msa, temp_tag, MAX_GAP_NUM, MAX_GAP_FREQ, verbose=True, ref_file=arg_list.refFile)
    
    filter_time = timer()
    print2('time to filter sequences for quality:\t', filter_time - start_time)
    
    # eliminate sequences with too many gaps
    idxs_drop = []
    ref_gaps  = np.sum(np.array(list(ref_seq))=='-')
    #temp_msa  = np.array([list(i) for i in list(df_temp['sequence'])])
    for i in range(len(temp_msa)):
        if np.sum(temp_msa[i]=='-')-ref_gaps>=MAX_GAP_NUM:
            idxs_drop.append(i)
    idxs_keep = np.array([i for i in np.arange(len(temp_msa)) if i not in idxs_drop])
    if len(idxs_keep)==0:
        sys.exit()
    temp_msa  = np.array(temp_msa)[idxs_keep]
    temp_tag  = np.array(temp_tag)[idxs_keep]
    print2(f'number of sequences with too many gaps is {len(idxs_drop)}')
    
    old_time = timer()
    print2(f'seconds to create new dataframe is {old_time - filter_time}')
    
    region_start_time = timer()
    
    # Compile sequences and tags
    ### CHANGE TO LOOP THROUGH DATAFRAME VIA "for idx, entry in enumerate(df_temp)" ###
        
    # Data processing, time ordering, and quality checks
    ## - put sequences in time order (first entry is reference sequence)
    print2('\ttime-ordering sequences...')
    temp_msa, temp_tag = order_sequences(temp_msa, temp_tag)
    temp_msa = np.array([list(i) for i in temp_msa])
    if len(temp_msa)<=1:
        sys.exit()
    region_timeorder_time = timer()
    print2('\ttime-ordering took %d seconds' % (region_timeorder_time - region_start_time))
    
    ## - impute ambiguous nucleotides and gaps at start/end of alignment
    print2('\timputing ambiguous nucleotides...')
    times = get_times(temp_msa, temp_tag, sort=False)
    t_unique, counts = np.unique(times, return_counts=True)
    impute_intervals = make_impute_intervals(t_unique[1:], counts[1:])
    print(f"impute intervals", impute_intervals)
    
    #ref_seq   = temp
    #new_msa   = [ref_seq]
    #new_tag   = [REF_TAG]
    new_msa   = [temp_msa[list(temp_tag).index(REF_TAG)]]
    new_tag   = [REF_TAG]
    chunk_msa = []
    chunk_tag = []
    for i in range(len(impute_intervals) - 1):
        start_t = impute_intervals[i]
        end_t   = impute_intervals[i+1]
        mask    = np.logical_and(times>=start_t, times<end_t)
        tag_masked = np.array(temp_tag)[mask]
        print('length of tag_masked', len(tag_masked))
        msa_masked = np.array(temp_msa)[mask]
        if len(msa_masked)==0:
            continue
        chunk_msa.append(msa_masked)
        chunk_tag.append(tag_masked)
    with mp.Pool(arg_list.threads) as pool:
        #result = pool.starmap(impute_ambiguous, zip(chunk_msa, chunk_tag, repeat(ref_idxs)))
        result = pool.starmap(impute_ambiguous, zip(chunk_msa, chunk_tag))
    #print(result)
    for i in range(len(result)):
        for j in range(len(result[i])):
            new_msa.append(result[i][j])
            new_tag.append(chunk_tag[i][j])
    n = [len(i) for i in new_msa]
    print(f'length of sequences are: {np.unique(n)}')
    temp_msa = np.array(new_msa)
    temp_tag = np.array(new_tag)

    region_impute_time = timer()
    print2('\timputing ambiguous nucleotides took %d seconds' % (region_impute_time - region_timeorder_time))
    
    # Identify and record polymorphic sites
    print2("\tidentifying and recording polymorphic sites...")
    out_str = os.path.join(out_dir, '%s-index.csv' % sub_label[-1])
    ref_sites, poly_sites, temp_msa, temp_tag, ref_poly = create_index(temp_msa, temp_tag, temp_msa[list(temp_tag).index(REF_TAG)], ref_idxs,
                                                                       0, MIN_SEQS, MAX_DT, out_str, return_polymorphic=True, 
                                                                       return_truncated=True)
    
    os.remove(out_str)
    region_polymorphic_time = timer()
    print2("\tidentifying and recording polymorphic sites took %d seconds" % (region_polymorphic_time - region_impute_time))
        
    print2("\tconverting sequences to binary and calculating number of each genome...")
    temp_label = sub_label[-1].replace('-None', '')
    genome_str = os.path.join(genome_dir, temp_label)
    save_alignment(temp_msa, temp_tag, genome_str, poly_sites, ref_sites)
    
    if os.path.exists(out_str):
        os.remove(out_str)
        
    region_conversion_time = timer()
    print2('\tconverting sequences to binary took %d seconds' % (region_conversion_time - region_polymorphic_time))
        
    new_time = timer()
    print2('\tregion %s took %d seconds to process' % (sub_label[-1], new_time - old_time))

    

    


if __name__ == '__main__':
    main()
