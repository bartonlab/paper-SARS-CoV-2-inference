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
        ref_sites.append(ref_idxs[i])

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


def freqs_from_seqs_new(seq_file, out_file, ref_seq_poly, poly_sites, ref_sites, new_filetype=False):
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
    
    if new_filetype:
        f = open(out_file + '.csv', mode='w')
        f.write('dates,count,sequence')
        for i in range(len(nVec)):
            for j in range(len(nVec[i])):
                f.write('%d\t%d\t%s\n' % (dates_full[i], nVec[i][j], ' '.join([str(k) for k in sVec[i][j]])))
        f.close()
        g = open(out_file + '.npz', mode='w+b')
        np.savez_compressed(g, mutant_sites=poly_sites, ref_sites=ref_sites)
        g.close()
    else:
        f = open(out_file + ".npz", mode='w+b')
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


def filter_excess_gaps(msa, tag, sequence_max_gaps, site_max_gaps, verbose=True):
    """ Remove sequences and sites from the alignment which have excess gaps. """
    
    #init_index  = 316   # the site number on the alignment corresponding to the 150th nucleotide in the reference sequence
    #final_index = 34898 # the site number on the alignment corresponding to the 29590th nucleotide in the reference sequence
    msa = list(msa)
    tag = list(tag)
    
    ref_idx = tag.index(REF_TAG)
    ref_seq = list(msa[ref_idx])
    del msa[ref_idx]
    del tag[ref_idx]
    
    # get unique site index from site labels for the reference sequence
    ref_df      = pd.read_csv('ref-index.csv')
    ref_indices = list(ref_df['ref_index'])
    
    """
    # Remove sequences with too many gaps
    temp_msa = []
    temp_tag = []
    ref_gaps = np.sum(np.array(ref_seq)=='-')
    print(f'There are {ref_gaps} gaps in the reference sequence')
    #print('reference sequence', ref_seq)
    for i in range(len(msa)):
        if np.sum(msa[i]=='-')-ref_gaps<sequence_max_gaps:
            temp_msa.append(list(msa[i]))
            temp_tag.append(tag[i])
    temp_msa = np.array(temp_msa)
    if verbose:
        print('\tselected %d of %d sequences with <%d gaps in excess of reference' %
              (len(temp_msa), len(msa), sequence_max_gaps))
    """
    
    # Drop sites that have too many gaps
    kept_indices = []
    for i in range(len(ref_seq)):
        if np.sum(temp_msa[:,i]=='-')/len(temp_msa)<site_max_gaps:
            # determine index number on reference sequence
            """
            if ref_indices[i][-2] in list(ALPHABET):
                ref_index = int(ref_indices[i][:-2])
            elif ref_indices[i][-1] in list(ALPHABET):
                ref_index = int(ref_indices[i][:-1])
            else:
                ref_index = int(ref_indices[i])
            """
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
        """
        temp  = file.split('---')
        locs  = temp[0]
        times = temp[1][:-4].split('-')
        init_time = '-'.join(times[:3])
        end_time  = '-'.join(times[3:])
        if locs.find('2')==-1:
            new_loc = locs
        else:
            new_loc = locs[:locs.find('2')-1]
        new_loc = new_loc.split('-')
        while len(new_loc) < 4:
            new_loc.append(None)
        if len(new_loc)>4:
            new_loc = new_loc[:4]
        if new_loc[-1] == 'south' and new_loc[-2] == 'california':
            new_loc = new_loc[:3]
            new_loc.append('None')
        elif new_loc[-1]=='england_wales_scotland':
            new_loc = new_loc[:-3]
            new_loc.append(['england', 'wales', 'scotland'])
        new_loc.append(init_time)
        new_loc.append(end_time)
        selected.append(new_loc)
        """
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
