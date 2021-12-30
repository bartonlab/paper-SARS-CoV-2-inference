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

NUC = ['-', 'A', 'C', 'G', 'T']
PRO = ['-', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
       'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
REF = NUC[0]
REF_TAG = 'EPI_ISL_402125'
ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+++++++++++++++++++++++++++'
TIME_INDEX = 1

# SEQUENCE PROCESSING GLOBAL VARIABLES

START_IDX    = 0
END_IDX      = 29800
MAX_GAP_NUM  = 10000
MAX_GAP_FREQ = 0.99
MIN_SEQS     = 0
MAX_DT       = 9999


status_name = 'genome-processing-status-multisite.csv'
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


def impute_ambiguous(msa, tag, start_index=0, verbose=True, impute_edge_gaps=False):
    """ Impute ambiguous nucleotides with the most frequently observed ones in the alignment. """
    
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
                    
    # Impute leading and trailing gaps
    #if impute_edge_gaps:
    #    for j in range(start_index, len(msa)):
    #        gap_lead = 0
    #        gap_trail = 0
            
    #        gap_idx = 0
    #        while msa[j][gap_idx]=='-':
    #            gap_lead += 1
    #            gap_idx += 1
                
    #        gap_idx = -1
    #        while msa[j][gap_idx]=='-':
    #            gap_trail -= 1
    #            gap_idx -= 1
                
    #        for i in range(gap_lead):
    #            new = NUC[1 + np.argmax(avg[i][1:])]
    #            msa[j][i] = new
                
    #        for i in range(gap_trail, 0):
    #            new = NUC[1 + np.argmax(avg[i][1:])]
    #            msa[j][i] = new
                
    #        if verbose and ((gap_lead>0) or (gap_trail<0)):
    #            print('\timputed %d leading gaps and %d trailing gaps in sequence %d' % (gap_lead, -1*gap_trail, j))

    return msa


def create_index(msa, tag, ref_seq, ref_start, min_seqs, max_dt, out_file, return_polymorphic=True, return_truncated=True):
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
    temp_msa, temp_tag, times = get_times(msa, tag, sort=True)
    u_times = np.unique(times)
    t_count = [np.sum(times==t) for t in u_times]
    
    t_allowed = [u_times[0]]
    t_last    = u_times[0]
    for i in range(1, len(t_count)):
        if t_count[i]<min_seqs:
            continue
        elif u_times[i]-t_last>max_dt:
            break
        else:
            t_allowed.append(u_times[i])
            t_last = u_times[i]
    t_max    = t_allowed[-1]
    temp_msa = temp_msa[np.isin(times, t_allowed)]
    temp_tag = temp_tag[np.isin(times, t_allowed)]
    
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
        ref_sites.append(ref_str)

        # Save to file
        f.write('%d,%s,%s,%s\n' % (i, poly_str, ref_str, ref_seq[i]))
    f.close()

    temp_msa = [ref_seq] + list(temp_msa)
    temp_tag = [REF_TAG] + list(temp_tag)
    # Change this to be able to deal with insertions/deletions
    ref_seq_poly = [str(NUC.index(a)) for a in ref_seq[polymorphic_sites]]
    ref_sites    = np.array(ref_sites)[polymorphic_sites]
    
    if return_polymorphic and return_truncated:
        return ref_sites, polymorphic_sites, temp_msa, temp_tag, ref_seq_poly
    elif return_polymorphic:
        return polymorphic_sites
    elif return_truncated:
        return temp_msa, temp_tag
    
    
def save_MPL_alignment(msa, tag, out_file, polymorphic_sites=[], return_states=True, protein=False):
    """ Save a nucleotide alignment into MPL-readable form. Optionally return converted states and times. """

    idxs = [i for i in range(len(msa)) if tag[i]!=REF_TAG]
    temp_msa = np.array(msa)[idxs]
    temp_tag = np.array(tag)[idxs]
    
    if polymorphic_sites==[]:
        polymorphic_sites = range(len(temp_msa[0]))

    poly_times = get_times(temp_msa, temp_tag, sort=False)

    poly_states = []
    if protein:
        for s in temp_msa:
            poly_states.append([str(PRO.index(a)) for a in s[polymorphic_sites]])
    else:
        for s in temp_msa:
            poly_states.append([str(NUC.index(a)) for a in s[polymorphic_sites]])

    f = open(out_file, 'w')
    for i in range(len(poly_states)):
        f.write('%d\t1\t%s\n' % (poly_times[i], ' '.join(poly_states[i])))
    f.close()

    if return_states:
        return np.array(poly_states, int), np.array(poly_times)
    
    
def freqs_from_seqs_new(seq_file, out_file, ref_seq_poly, poly_sites, ref_sites):
    """ Produces the file that can be plugged into the MPL given the file containing the .dat sequence data"""
    
    times, sequences = [], []
    for line in open(seq_file):
        time, seq = line.split("\t1\t", 1)[0], line.split("\t1\t", 1)[1][0:-1]
        times.append(time)
        sequences.append(seq)
    
    nVec = []
    sVec = []
    max_time     = np.amax(np.array(times, dtype=int))
    min_time     = np.amin(np.array(times, dtype=int))
    new_times    = np.arange(min_time, max_time + 1)
    for t in range(len(new_times)):
        seq    = []
        counts = []
        time   = new_times[t]
        for i in range(len(sequences)):
            if int(times[i]) == time:
                if len(seq) > 1:
                    if sequences[i] in seq:
                        loc  = seq.index(sequences[i])
                        counts[loc] += 1
                    else:
                        seq.append(sequences[i])
                        counts.append(1)
                elif len(seq) == 1:
                    if sequences[i] == seq[0]:
                        counts[0] += 1
                    else:
                        seq.append(sequences[i])
                        counts.append(1)
                else:
                    seq.append(sequences[i])
                    counts.append(1)
        nVec.append(np.array(counts))
        sVec.append(np.array([np.array(i.split(' '), dtype=int) for i in seq]))
    
    f = open(out_file+".npz", mode='w+b')
    np.savez_compressed(f, nVec=nVec, sVec=sVec, times=new_times, mutant_sites=poly_sites, ref_sites=ref_sites)
    f.close()
    

def codon2aa(c, noq=False):
    """ Return the amino acid character corresponding to the input codon. """
    
    # If all nucleotides are missing, return gap
    if c[0]=='-' and c[1]=='-' and c[2]=='-': return '-'
    
    # Else if some nucleotides are missing, return '?'
    elif c[0]=='-' or c[1]=='-' or c[2]=='-':
        if noq: return '-'
        else:   return '?'
    
    # If the first or second nucleotide is ambiguous, AA cannot be determined, return 'X'
    elif c[0] in ['W', 'S', 'M', 'K', 'R', 'Y'] or c[1] in ['W', 'S', 'M', 'K', 'R', 'Y']: return 'X'
    
    # Else go to tree
    elif c[0]=='T':
        if c[1]=='T':
            if    c[2] in ['T', 'C', 'Y']: return 'F'
            elif  c[2] in ['A', 'G', 'R']: return 'L'
            else:                          return 'X'
        elif c[1]=='C':                    return 'S'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'Y'
            elif  c[2] in ['A', 'G', 'R']: return '*'
            else:                          return 'X'
        elif c[1]=='G':
            if    c[2] in ['T', 'C', 'Y']: return 'C'
            elif  c[2]=='A':               return '*'
            elif  c[2]=='G':               return 'W'
            else:                          return 'X'
        else:                              return 'X'
    
    elif c[0]=='C':
        if   c[1]=='T':                    return 'L'
        elif c[1]=='C':                    return 'P'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'H'
            elif  c[2] in ['A', 'G', 'R']: return 'Q'
            else:                          return 'X'
        elif c[1]=='G':                    return 'R'
        else:                              return 'X'
    
    elif c[0]=='A':
        if c[1]=='T':
            if    c[2] in ['T', 'C', 'Y']: return 'I'
            elif  c[2] in ['A', 'M', 'W']: return 'I'
            elif  c[2]=='G':               return 'M'
            else:                          return 'X'
        elif c[1]=='C':                    return 'T'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'N'
            elif  c[2] in ['A', 'G', 'R']: return 'K'
            else:                          return 'X'
        elif c[1]=='G':
            if    c[2] in ['T', 'C', 'Y']: return 'S'
            elif  c[2] in ['A', 'G', 'R']: return 'R'
            else:                          return 'X'
        else:                              return 'X'
    
    elif c[0]=='G':
        if   c[1]=='T':                    return 'V'
        elif c[1]=='C':                    return 'A'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'D'
            elif  c[2] in ['A', 'G', 'R']: return 'E'
            else:                          return 'X'
        elif c[1]=='G':                    return 'G'
        else:                              return 'X'
    
    else:                                  return 'X'
    
    
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


def filter_excess_gaps(msa, tag, sequence_max_gaps, site_max_gaps, verbose=True):
    """ Remove sequences and sites from the alignment which have excess gaps. """
    
    msa = list(msa)
    tag = list(tag)
    
    ref_idx = tag.index(REF_TAG)
    ref_seq = msa[ref_idx]
    del msa[ref_idx]
    del tag[ref_idx]
    
    # Remove sequences with too many gaps
    temp_msa = []
    temp_tag = []
    ref_gaps = np.sum(ref_seq=='-')
    print(f'There are {ref_gaps} gaps in the reference sequence')
    for i in range(len(msa)):
        if np.sum(msa[i]=='-')-ref_gaps<sequence_max_gaps:
            temp_msa.append(list(msa[i]))
            temp_tag.append(tag[i])
    temp_msa = np.array(temp_msa)
    if verbose:
        print('\tselected %d of %d sequences with <%d gaps in excess of reference' %
              (len(temp_msa), len(msa), sequence_max_gaps))
    
    # Drop sites that have too many gaps
    kept_indices = []
    for i in range(len(ref_seq)):
        if ref_seq[i]!='-' or np.sum(temp_msa[:,i]=='-')/len(temp_msa)<site_max_gaps:
            kept_indices.append(i)
    print(f'{len(ref_seq) - len(kept_indices)} sites are being removed due to the number of sequences with gaps')
    temp_msa = [np.array(list(ref_seq))[kept_indices]] + [np.array(s)[kept_indices] for s in temp_msa]
    temp_tag = [REF_TAG] + temp_tag

    return temp_msa, temp_tag


def main():
    """ Given the desired regions to be analyzed, collects the sequences from those regions and in the given time frames and then filters out 
    sequences and sites that have too many gaps."""
    
    parser = argparse.ArgumentParser(description='filtering out sequences and sites with too many gaps')
    parser.add_argument('--input_file',  type=str,    default='regional-data.csv.gz',     help='sequence alignment file')
    parser.add_argument('-o',            type=str,    default='regional',                 help='output destination')
    parser.add_argument('--regions',     type=str,    default=None,                       help='the regions and times to extract data from')
    parser.add_argument('--find_syn_off', default=False,  action='store_true',            help='whether or not to determine which sites are synonymous and not')
    parser.add_argument('--full_index',   default=False,  action='store_true',            help='whether or not to calculate the full index for sites')
    parser.add_argument('--no_trim',      default=False,  action='store_true',            help='whether or not to trim sites based on frequency')
    
    arg_list = parser.parse_args(sys.argv[1:])
    out_folder   = arg_list.o
    input_file   = arg_list.input_file
    find_syn_off = arg_list.find_syn_off
    full_index   = arg_list.full_index
    
    start_time = timer()
    
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    genome_dir = os.path.join(out_folder, "genome-data")
    if not os.path.exists(genome_dir):
        os.mkdir(genome_dir)
    out_dir = out_folder
        
    df = pd.read_csv(input_file)
    print2('loading MSA finished')
    print2('number of genomes in original MSA', len(list(df.accession)))
    
    locs       = list(df.location)
    if 'virus_name' in df:
        virus_name = list(df.virus_name)
        virus_name = [i.split('/') for i in virus_name]
        locs2      = []
        for i in range(len(virus_name)):
            if len(virus_name[i])>2:
                locs2.append(virus_name[i][2])
            else:
                print(len(virus_name[i]))
                locs2.append('')
        locs_new   = []
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
                l.append('NA')
                l.append('/'.join(m))
            elif len(l) == 4:
                l.append('/'.join(m))
            elif len(l) == 5:
                l[4] += '/' + '/'.join(m)
            #else:
                #print('there is too much location information', len(l), len(m))
            locs_new.append(l)
        """
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
    identifier = out_folder
    selected   = np.load(arg_list.regions, allow_pickle=True)
    
    # Read in data
    ref_seq = str(df[df.accession==REF_TAG].iloc[0].sequence)

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
            
        df_locs = [i                               for i in range(len(df_pass)) if df_pass[i]]
        df_time = [int((df_date[i]-ref_date).days) for i in range(len(df_pass)) if df_pass[i]]
        df_temp = df_temp.iloc[df_locs]
    
        if counter == 0:
            df_new = df_temp.copy()
        else:
            df_new = df_new.append(df_temp, ignore_index=True)
        counter += 1
    
    collection_time = timer()
    print2('number of genomes in df_new', len(list(df_new.accession)))
    print2('seconds to collect the relevant sequences:\t', collection_time - start_time)
    # Compile sequences and tags
    temp_msa = [ref_seq]
    temp_tag = [REF_TAG]
    idx_drop = []
    for i in range(len(df_new)):
        tag = '.'.join([str(df_new.iloc[i].accession), str(df_new.iloc[i].date)])
        if tag not in temp_tag:
            temp_tag.append(tag)
            temp_msa.append(str(df_new.iloc[i].sequence))
        else:
            print('\tUnexpected duplicate sequence %s' % tag)
            idx_drop.append(i)
    
    ## - filter sequences for quality (maximum #excess gaps)
    print2('filtering sequences by gaps...')
    temp_msa, temp_tag = filter_excess_gaps(temp_msa, temp_tag, MAX_GAP_NUM, MAX_GAP_FREQ, verbose=True)
    ref_row = pd.DataFrame({'sequence' : ''.join(temp_msa[0]), 'date' : ref_date, 'accession' : REF_TAG, 'location' : 'asia / china / wuhan', 
                           'continent' : 'asia', 'country': 'china', 'region' : 'wuhan'}, index=[0])
    
    filter_time = timer()
    print2('time to filter sequences for quality:\t', filter_time - collection_time)
    
    new_msa = []
    for seq in temp_msa[1:]:
        new_msa.append(''.join(seq))
    if len(idx_drop)>0:
        df_new = df_new.drop(labels=idx_drop, axis=0)
    df_new['sequence'] = new_msa
    df_new  = pd.concat([ref_row, df_new[:]], sort=False).reset_index(drop = True)
    ref_seq = df_new[df_new.accession==REF_TAG].iloc[0].sequence
    print('reference sequence:', ref_seq)
    df      = df_new.drop(0, axis=0)
    
    old_time = timer()
    print2('seconds to classify mutations as synonymous or nonsynonymous:\t', old_time - filter_time)
    
    sub_label = []
    for conds in selected:
        # Read in conditions
        s_continent = conds[0]
        s_country   = conds[1]
        s_region    = conds[2]
        s_subregion = conds[3]
        date_lower  = conds[4]
        date_upper  = conds[5]
        df_temp     = df.copy()
    
        # Update progress
        region_start_time = timer()
        print2(f'Getting continent={s_continent}, country={s_country}, region={s_region}, subregion={s_subregion}, between {date_lower} and {date_upper}...')
        if s_region=='california':
            temp_label = f'{s_continent}-{s_country}-{s_region}-south-{date_lower}-{date_upper}'
        elif s_country != 'usa' and (type(s_region) == list or type(s_region) == np.ndarray):
            temp_label = f'{s_continent}-{s_country}-' + '_'.join(s_region) + f'-{date_lower}-{date_upper}'
        else:
            temp_label = f'{s_continent}-{s_country}-{s_region}-{s_subregion}-{date_lower}-{date_upper}'
        print2('The label for this subset is %s' % temp_label)
        sub_label.append(temp_label)
    
        # Sort by country/state/sub
        for s_cond, col in [[s_country, 'country'], [s_region, 'region'], [s_subregion, 'subregion']]:
            if s_cond!=None:
                if isinstance(s_cond, str):
                    df_temp = df_temp[df_temp[col]==s_cond]
                elif isinstance(s_cond, type([])):
                    df_temp = df_temp[df_temp[col].isin(s_cond)]
                else:
                    print('\tUnexpected %s type ' % col, s_cond)
    
        # Sort by date
        df_date = [fromisoformat(val) for val in list(df_temp['date'])]
        df_pass = [True for i in range(len(df_temp))]
        df_locs = [i                               for i in range(len(df_pass)) if df_pass[i] and int((df_date[i]-ref_date).days)>0]
        df_time = [int((df_date[i]-ref_date).days) for i in range(len(df_pass)) if df_pass[i] and int((df_date[i]-ref_date).days)>0]
        df_temp = df_temp.iloc[df_locs]
        print2('\tNumber of sequences: %s' % len(df_temp))
    
        # Compile sequences and tags

        temp_msa = [list(ref_seq)]
        temp_tag = [REF_TAG]
        for i in range(len(df_temp)):
            tag = '.'.join([str(df_temp.iloc[i].accession), str(df_time[i])])
            if tag not in temp_tag:
                temp_tag.append(tag)
                temp_msa.append(list(df_temp.iloc[i].sequence))
            else:
                print('\tUnexpected duplicate sequence %s' % tag)
        temp_msa = np.array(temp_msa)         
    
        # Data processing, time ordering, and quality checks
        ## - put sequences in time order (first entry is reference sequence)
        print2('\ttime-ordering sequences...')
        temp_msa, temp_tag = order_sequences(temp_msa, temp_tag)
        region_timeorder_time = timer()
        print2('\ttime-ordering took %d seconds' % (region_timeorder_time - region_start_time))
    
        ## - impute ambiguous nucleotides and gaps at start/end of alignment
        print2('\timputing ambiguous nucleotides...')
        temp_msa = impute_ambiguous(temp_msa, temp_tag, start_index=1, verbose=False, impute_edge_gaps=True)

        region_impute_time = timer()
        print2('\timputing ambiguous nucleotides took %d seconds' % (region_impute_time - region_timeorder_time))
    
        # Identify and record polymorphic sites
        print2("\tidentifying and recording polymorphic sites...")
        out_str = os.path.join(out_dir, '%s-index.csv' % sub_label[-1])
        ref_sites, poly_sites, temp_msa, temp_tag, ref_poly = create_index(temp_msa, temp_tag, temp_msa[list(temp_tag).index(REF_TAG)], 
                                                                           1, MIN_SEQS, MAX_DT, out_str, return_polymorphic=True, 
                                                                           return_truncated=True)
        
        region_polymorphic_time = timer()
        print2("\tidentifying and recording polymorphic sites took %d seconds" % (region_polymorphic_time - region_impute_time))
    
        out_str = os.path.join(out_dir, '%s-poly.dat' % sub_label[-1])
        poly_states, poly_times = save_MPL_alignment(temp_msa, temp_tag, out_str, polymorphic_sites=poly_sites, 
                                                     return_states=True)
        #full_states.append(poly_states)
        
        print2("\tconverting sequences to binary and calculating number of each genome...")
        genome_str = os.path.join(genome_dir, f"{sub_label[-1]}")
        freqs_from_seqs_new(out_str, genome_str, ref_poly, poly_sites, ref_sites)

        region_conversion_time = timer()
        print2('\tconverting sequences to binary took %d seconds' % (region_conversion_time - region_polymorphic_time))
        
        new_time = timer()
        print2('\tregion %s took %d seconds to process' % (temp_label, new_time - old_time))
        old_time = timer()
        
        dp.save_MSA([ref_seq], [REF_TAG], REF_TAG + '-' + sub_label[-1])
    
    # save reference sequence
    print(f'current working directory is {os.getcwd()}')
    print(f'saving reference sequence to {REF_TAG}.fasta')
    dp.save_MSA([ref_seq], [REF_TAG], REF_TAG)

    syn_time = timer()
    
    if not arg_list.no_trim:
        trim_dir = os.path.join(out_dir, 'genome-trimmed')
        dp.restrict_sampling_intervals(genome_dir, trim_dir, window=5, max_dt=5, min_seqs=20, min_range=20)
    
        freq_dir1 = os.path.join(out_dir, 'freq_0.05')
        dp.eliminate_low_freq_multisite(trim_dir, freq_dir1, ref_seq=list(ref_seq), freq_tol=0.05)
    
        freq_dir2 = os.path.join(out_dir, 'freq_0.01')
        dp.eliminate_low_freq_multisite(trim_dir, freq_dir2, ref_seq=list(ref_seq), freq_tol=0.01)
    
        freq_dir3 = os.path.join(out_dir, 'freq_0.1')
        dp.eliminate_low_freq_multisite(trim_dir, freq_dir3, ref_seq=list(ref_seq), freq_tol=0.1)
    
    trim_time = timer()
    print2('finding intervals with good sampling in each region and eliminating low frequency mutations took %d seconds' % (trim_time - syn_time))
    
    


if __name__ == '__main__':
    main()