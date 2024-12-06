# LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
from timeit import default_timer as timer   # timer for performance
from copy import deepcopy
from multiprocessing import Pool
from scipy import linalg
import shutil

# GLOBAL VARIABLES

NUC = ['-', 'A', 'C', 'G', 'T']
PRO = ['-', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
       'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
REF = NUC[0]
REF_TAG  = 'EPI_ISL_402125'
ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+++++++++++++++++++++++++++'
NUMBERS  = list('0123456789')
TIME_INDEX = 1
PROTEIN_LENGTHS = {'ORF3a' : 828, 'E' : 228, 'ORF6' : 186, 'ORF7a' : 365, 'ORF7b' : 132, 'S' : 3822,
                   'N' : 1260, 'M' : 669, 'ORF8' : 336, 'ORF10' : 117, 'NSP1' : 539, 'NSP2' : 1914,
                   'NSP3' : 5834, 'NSP4' : 1500, 'NSP5' : 917, 'NSP6' : 870, 'NSP7' : 249, 'NSP8' : 594,
                   'NSP9' : 339, 'NSP10' : 417, 'NSP12' :2795, 'NSP13' : 1803, 'NSP14' : 1582, 
                   'NSP15' : 1038, 'NSP16' : 894}

START_IDX = 0 
END_IDX   = 29800
"""
ALPHABET_NEW = []
for i in ALPHABET:
    ALPHABET_NEW.append(i)
for i in ALPHABET:
    for j in ALPHABET:
        ALPHABET_NEW.append(i + j)
"""

def load(file):
    return np.load(file, allow_pickle=True)

# FUNCTIONS

def get_MSA(ref, noArrow=True):
    """Take an input FASTA file and return the multiple sequence alignment, along with corresponding tags. """
    
    temp_msa = [i.split('\n') for i in open(ref).readlines()]
    temp_msa = [i for i in temp_msa if len(i)>0]
    
    msa = []
    tag = []
    
    for i in temp_msa:
        if i[0][0]=='>':
            msa.append('')
            if noArrow: tag.append(i[0][1:])
            else: tag.append(i[0])
        else: msa[-1]+=i[0]
    
    msa = np.array(msa)
    
    return msa, tag


def get_label(i):
    """ For a SARS-CoV-2 reference sequence index i, return the label in the form 
    'coding region - protein number-nucleotide in codon number'. 
    For example, 'ORF1b-204-1'. 
    Should check to make sure NSP12 labels are correct due to the frame shift."""
    i = int(i)
    frame_shift = str(i - get_codon_start_index(i))
    if   (25392<=i<26220):
        return "ORF3a-" + str(int((i - 25392) / 3) + 1)  + '-' + frame_shift
    elif (26244<=i<26472):
        return "E-"     + str(int((i - 26244) / 3) + 1)  + '-' + frame_shift
    elif (27201<=i<27387):
        return "ORF6-"  + str(int((i - 27201) / 3) + 1)  + '-' + frame_shift
    # ORF7a and ORF7b overlap by 4 nucleotides
    elif (27393<=i<27759):
        return "ORF7a-" + str(int((i - 27393) / 3) + 1)  + '-' + frame_shift
    elif (27755<=i<27887):
        return "ORF7b-" + str(int((i - 27755) / 3) + 1)  + '-' + frame_shift
    elif (  265<=i<805):
        return "NSP1-"  + str(int((i - 265  ) / 3) + 1)  + '-' + frame_shift
    elif (  805<=i<2719):
        return "NSP2-"  + str(int((i - 805  ) / 3) + 1)  + '-' + frame_shift
    elif ( 2719<=i<8554):
        return "NSP3-"  + str(int((i - 2719 ) / 3) + 1)  + '-' + frame_shift
            # Compound protein containing a proteinase, a phosphoesterase transmembrane domain 1, etc.
    elif ( 8554<=i<10054):
        return "NSP4-"  + str(int((i - 8554 ) / 3) + 1)  + '-' + frame_shift
            # Transmembrane domain 2
    elif (10054<=i<10972):
        return "NSP5-"  + str(int((i - 10054) / 3) + 1)  + '-' + frame_shift
            # Main proteinase
    elif (10972<=i<11842):
        return "NSP6-"  + str(int((i - 10972) / 3) + 1)  + '-' + frame_shift
            # Putative transmembrane domain
    elif (11842<=i<12091):
        return "NSP7-"  + str(int((i - 11842) / 3) + 1)  + '-' + frame_shift
    elif (12091<=i<12685):
        return "NSP8-"  + str(int((i - 12091) / 3) + 1)  + '-' + frame_shift
    elif (12685<=i<13024):
        return "NSP9-"  + str(int((i - 12685) / 3) + 1)  + '-' + frame_shift
            # ssRNA-binding protein
    elif (13024<=i<13441):
        return "NSP10-" + str(int((i - 13024) / 3) + 1)  + '-' + frame_shift
            # CysHis, formerly growth-factor-like protein
    # Check that aa indexing is correct for NSP12, becuase there is a -1 ribosomal frameshift at 13468. It is not, because the amino acids in the first frame
    # need to be added to the counter in the second frame.
    elif (13441<=i<13467):
        return "NSP12-" + str(int((i - 13441) / 3) + 1)  + '-' + frame_shift
    elif (13467<=i<16236):
        return "NSP12-" + str(int((i - 13467) / 3) + 10) + '-' + frame_shift
            # RNA-dependent RNA polymerase
    elif (16236<=i<18039):
        return "NSP13-" + str(int((i - 16236) / 3) + 1)  + '-' + frame_shift
            # Helicase
    elif (18039<=i<19620):
        return "NSP14-" + str(int((i - 18039) / 3) + 1)  + '-' + frame_shift
            # 3' - 5' exonuclease
    elif (19620<=i<20658):
        return "NSP15-" + str(int((i - 19620) / 3) + 1)  + '-' + frame_shift
            # endoRNAse
    elif (20658<=i<21552):
        return "NSP16-" + str(int((i - 20658) / 3) + 1)  + '-' + frame_shift
            # 2'-O-ribose methyltransferase
    elif (21562<=i<25384):
        return "S-"     + str(int((i - 21562) / 3) + 1)  + '-' + frame_shift
    elif (28273<=i<29533):
        return "N-"     + str(int((i - 28273) / 3) + 1)  + '-' + frame_shift
    elif (29557<=i<29674):
        return "ORF10-" + str(int((i - 29557) / 3) + 1)  + '-' + frame_shift
    elif (26522<=i<27191):
        return "M-"     + str(int((i - 26522) / 3) + 1)  + '-' + frame_shift
    elif (27893<=i<28259):
        return "ORF8-"  + str(int((i - 27893) / 3) + 1)  + '-' + frame_shift
    else:
        return "NC-"    + str(int(i))
    
        
def get_codon_start_index(i):
    """ Given a sequence index i, determine the index of the first nucleotide in the codon. """
    if   (13467<=i<=21554):
        return i - (i - 13467)%3
    elif (25392<=i<=26219):
        return i - (i - 25392)%3
    elif (26244<=i<=26471):
        return i - (i - 26244)%3
    elif (27201<=i<=27386):
        return i - (i - 27201)%3
    # new to account for overlap of orf7a and orf7b (UNCOMMENT)
    elif (27393<=i<=27754):
        return i - (i - 27393)%3
    elif (27755<=i<=27886):
        return i - (i - 27755)%3
    ### considered orf7a and orf7b as one reading frame (INCORRECT)
    ### REMOVE BELOW
    #elif (27393<=i<=27886):
    #    return i - (i - 27393)%3
    ### REMOVE ABOVE
    elif (  265<=i<=13467):
        return i - (i - 265  )%3
    elif (21562<=i<=25383):
        return i - (i - 21562)%3
    elif (28273<=i<=29532):
        return i - (i - 28273)%3
    elif (29557<=i<=29673):
        return i - (i - 29557)%3
    elif (26522<=i<=27190):
        return i - (i - 26522)%3
    elif (27893<=i<=28258):
        return i - (i - 27893)%3
    else:
        return 0


def get_label_orf(i, split_orf1=False):
    """ Converts the number in the sequence to the protein codon number, ORF1ab is not split up into non-structural proteins."""
    
    i = int(i)
    frame_shift = str(i - get_codon_start_index(i))
    if   (25392<=i<=26219):
        return "ORF3a-"  + str(int((i - 25392) / 3) + 1) + '-' + frame_shift
    elif (26244<=i<=26471):
        return "E-"      + str(int((i - 26244) / 3) + 1) + '-' + frame_shift
    elif (27201<=i<=27386):
        return "ORF6-"   + str(int((i - 27201) / 3) + 1) + '-' + frame_shift
    elif (27393<=i<=27758):
        return "ORF7a-"  + str(int((i - 27393) / 3) + 1) + '-' + frame_shift
    elif (27755<=i<=27886):
        return "ORF7b-"  + str(int((i - 27755) / 3) + 1) + '-' + frame_shift
    elif (  265<=i<=21551):
        if split_orf1:
            if (  265<=i<13467):
                return "ORF1a-" + str(int((i - 265  ) / 3) + 1)     + '-' + frame_shift
            else:
                return "ORF1b-" + str(int((i - 13467) / 3) + 1)     + '-' + frame_shift
        else:
            if (  265<=i<13467):
                return "ORF1ab-" + str(int((i - 265  ) / 3) + 1)    + '-' + frame_shift
            else:
                return "ORF1ab-" + str(int((i - 13467) / 3) + 4402) + '-' + frame_shift
    elif (21562<=i<=25383):
        return "S-"      + str(int((i - 21562) / 3) + 1) + '-' + frame_shift
    elif (28273<=i<=29532):
        return "N-"      + str(int((i - 28273) / 3) + 1) + '-' + frame_shift
    elif (29557<=i<=29673):
        return "ORF10-"  + str(int((i - 29557) / 3) + 1) + '-' + frame_shift
    elif (26522<=i<=27190):
        return "M-"      + str(int((i - 26522) / 3) + 1) + '-' + frame_shift
    elif (27893<=i<=28258):
        return "ORF8-"   + str(int((i - 27893) / 3) + 1) + '-' + frame_shift
    else:
        return "NC-"     + str(int(i))

def get_label_orf_new(i, split_orf1=False):
    nuc   = i[-1]
    index = i.split('-')[0]
    if index[-1] in NUMBERS:
        return get_label2(i)
    else:
        if index[-1] in list(ALPHABET) and index[-2] in list(ALPHABET):
            temp = get_label_orf(index[:-2])
            gap  = index[-2:]
        elif index[-1] in list(ALPHABET):
            temp = get_label_orf(index[:-1])
            gap  = index[-1]
        else:
            temp = get_label_orf(index)
            gap  = None
        temp = temp.split('-')
        if gap is not None:
            temp[1] += gap
            #print(temp, gap)
        temp.append(nuc)
        label = '-'.join(temp)
        return label
    
def orf_to_nsp(i, split_orf1=False):
    """Given a nucleotide label in the form orf1-#, return label in the form NSP#-#-#"""
    n = 0
    i_end = int(i[i.find('-')+1:])
    if i[:i.find('-')]=='orf1a':
        n += 264
    elif i[:i.find('-')]=='orf1b':
        n += 13466
    else:
        print('input invalid')
    n += i_end * 3
    return get_label(n)
    
    
def nsp_to_orf(i):
    """ Given a nucleotide label in the form NSP#-#-# where the first number is the nonstructural protein number, 
    the second is codon number in the NSP and the third is the site number in the codon, returns a label in the 
    form ORF#-#-#.
    CHECK THAT THIS WORKS"""
    
    n = 0
    if i[:3] != 'NSP':
        return i
    else:
        # determine which NSP it came from and add the appropriate number of nucleotides
        if   i[4:6]=='1-':
            n += 265
        elif i[4:6]=='2-':
            n += 805
        elif i[4:6]=='3-':
            n += 2719
        elif i[4:6]=='4-':
            n += 8554
        elif i[4:6]=='5-':
            n += 10054
        elif i[4:6]=='6-':
            n += 10972
        elif i[4:6]=='7-':
            n += 11842
        elif i[4:6]=='8-':
            n += 12091
        elif i[4:6]=='9-':
            n += 12687
        elif i[4:6]=='10':
            n += 13024
        elif i[4:6]=='12':
            n += 13441
        elif i[4:6]=='13':
            n += 16236
        elif i[4:6]=='14':
            n += 18039
        elif i[4:6]=='15':
            n += 19620
        elif i[4:6]=='16':
            n += 20658
        # add 3 times the codon number
        idx1 = i.find('-')+1
        idx2 = i.find('-', idx1+1)
        n += (int(i[idx1:idx2])-1) * 3
        # add the nucleotide number within the codon
        n += int(i[-1])
        return(get_label_orf(n))
    

def get_label2(i):
    return get_label(i[:-2]) + '-' + i[-1]


def get_label_new(i):
    nuc   = i[-1]
    index = i.split('-')[0]
    if index[-1] in NUMBERS:
        return get_label(i[:-2]) + '-' + i[-1]
    else:
        if index[-1] in list(ALPHABET) and index[-2] in list(ALPHABET):
            temp = get_label(index[:-2])
            gap  = index[-2:]
        elif index[-1] in list(ALPHABET):
            temp = get_label(index[:-1])
            gap  = index[-1]
        else:
            temp = get_label(index)
            gap  = None
        temp = temp.split('-')
        if gap is not None:
            temp[1] += gap
            #print(temp, gap)
        temp.append(nuc)
        label = '-'.join(temp)
        return label
    
def separate_label_idx(i):
    """return the part of the site label that is an integer and the part that is letters for a nucleotide site such as 21000aE"""
    index = str(i)
    if index[-1] in NUMBERS:
        return index, ''
    else:
        if index[-1] in list(ALPHABET) and index[-2] in list(ALPHABET):
            return index[:-2], index[-2:]
        elif index[-1] in list(ALPHABET):
            return index[:-1], index[-1:]
        else:
            return index, ''
        

def index2frame(i):
    """ Return the open reading frames corresponding to a given SARS-CoV-2 reference sequence index. """

    frames = []
    #   ORF1b                ORF3a                E                    ORF6                 ORF7a
    if (13468<=i<=21555) or (25393<=i<=26220) or (26245<=i<=26472) or (27202<=i<=27387) or (27394<=i<=27759):
        frames.append(1)
    #   ORF1a                S                    N                    ORF10
    if (  266<=i<=13483) or (21563<=i<=25384) or (28274<=i<=29533) or (29558<=i<=29674):
        frames.append(2)
    #   M                    ORF8
    if (26523<=i<=27191) or (27894<=i<=28259):
        frames.append(3)

    return frames
    

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


def freq_change_correlation(traj_file, group1, group2, region=False, df_var_label='variant_names'):
    """ Takes two groups of mutations and the csv file that contains the changes in frequency for the mutations, 
    and calculates the correlation between them.
    group1 and group2 can be names of variants. 
    FUTURE: let group1 and group2 also be lists of mutations"""
    
    # load data
    df        = pd.read_csv(traj_file)
    
    var_names = list(df[df_var_label])
    #sites     = list(df['sites'])
    if isinstance(group1, str) and isinstance(group2, str):
        
        # find rows with the appropriate groups
        df1 = df[df[df_var_label]==group1]
        df2 = df[df[df_var_label]==group2]
        df1 = df1[[df_var_label, 'frequencies', 'location', 'times']].sort_values(by='location')
        df2 = df2[[df_var_label, 'frequencies', 'location', 'times']].sort_values(by='location')
        
        # find rows that have the same location for the two variants
        locs1     = list(df1['location'])
        locs2     = list(df2['location'])
        locs_both = list(set(locs1).intersection(set(locs2)))    # locations in which both variants appear
        if region is not False:
            if isinstance(region, str):
                locs_both = list(np.array(locs_both)[np.array(locs_both)==region])
            else:
                locs_temp = []
                for reg in region:
                    locs_temp.append(list(np.array(locs_both)[np.array(locs_both)==reg]))
                locs_both = locs_temp
        new_df1   = df1[np.isin(locs1, locs_both)]
        new_df2   = df2[np.isin(locs2, locs_both)]
        assert list(new_df1['location'])==list(new_df2['location'])
        
        # find frequency changes
        freqs1    = [np.array(i.split(' '), dtype=np.float32) for i in new_df1['frequencies'].to_numpy()]
        freqs2    = [np.array(i.split(' '), dtype=np.float32) for i in new_df2['frequencies'].to_numpy()]
        #print(freqs1[100])
        delta_x1  = [np.diff(i) for i in freqs1]
        delta_x2  = [np.diff(i) for i in freqs2]

        # combine into a single list and find the correlation
        total1    = [delta_x1[i][j] for i in range(len(delta_x1)) for j in range(len(delta_x1[i]))]
        total2    = [delta_x2[i][j] for i in range(len(delta_x2)) for j in range(len(delta_x2[i]))]
        return np.corrcoef(total1, total2)[0, 1]
    
    
def freq_change_correlation2(traj_file, group1, group2):
    """ Takes two groups of mutations and the csv file that contains the changes in frequency for the mutations, 
    and calculates the correlation between them.
    group1 and group2 can be names of variants. 
    FUTURE: let group1 and group2 also be lists of mutations"""
    
    # load data
    data = np.load(traj_file, allow_pickle=True)
    muts = data['mutant_sites']
    traj = data['traj']
    freqs1 = []
    freqs2 = []
    for i in range(len(muts)):
        muts_temp = [get_label2(j) for j in muts[i]]
        if group1 in muts_temp and group2 in muts_temp:
            temp1 = np.diff(traj[i][:, muts_temp.index(group1)])
            temp2 = np.diff(traj[i][:, muts_temp.index(group2)])
            for j in range(len(temp1)):
                freqs1.append(temp1[j])
                freqs2.append(temp2[j])
    return np.corrcoef(freqs1, freqs2)[0,1]


def find_selection_all_time(inf_dir=None, link_file=None, sCutoff=None):
    """Find the selection coefficients for all sites and all groups of linked sites over all time"""
    alleles = []
    locs    = []
    paths   = []
    linked  = np.load(link_file, allow_pickle=True)
    for file in os.listdir(inf_dir):
        if os.path.isdir(os.path.join(inf_dir, file)):
            continue
        data = np.load(os.path.join(inf_dir, file), allow_pickle=True)
        alleles.append([get_label2(i) for i in data['allele_number']])
        locs.append(file)
        paths.append(os.path.join(inf_dir, file))
    allele_number = np.unique([alleles[i][j] for i in range(len(alleles)) for j in range(len(alleles[i]))])
    linked_all    = np.unique([linked[i][j] for i in range(len(linked)) for j in range(len(linked[i]))])
    labels        = [i for i in allele_number if i not in linked_all]
    labels_sorted = np.argsort(labels)
    
    s_nolink = []
    s_link   = []
    for file in paths:
        data  = np.load(file, allow_pickle=True)
        s     = data['selection']
        muts  = [get_label2(i) for i in data['allele_number']]
        positions  = np.searchsorted(np.array(labels)[labels_sorted], muts)
        positions  = labels_sorted[positions]
        s_new      = np.zeros((len(s), len(labels)))
        s_link_new = np.zeros((len(s), len(linked)))
        for i in range(len(muts)):
            if muts[i] in linked_all:
                for j in range(len(linked)):
                    if muts[i] in linked[j]:
                        s_link_new[:, j] += s[:, i]
            elif muts[i] in labels:
                s_new[:, positions[i]] += s[:, i]
            else:
                print(f'error: mutation {muts[i]} not found in overall list of mutations')
        s_nolink.append(s_new)
        s_link.append(s_link_new)
    if sCutoff is None:
        # returns the locations, and the linked and nonlinked coefficients over all time in each location
        return locs, s_link, s_nolink
    else:
        largest_locs  = []
        largest_sites = []
        largest_sel   = []
        for loc in range(len(locs)):
            s_link_temp1 = np.amax(s_link[loc], axis=0)
            s_link_temp2 = np.amin(s_link[loc], axis=0)
            for i in range(len(s_link_temp1)):
                if s_link_temp1[i] > sCutoff:
                    largest_locs.append(locs[loc])
                    largest_sites.append('/'.join(linked[i]))
                    largest_sel.append(s_link_temp1[i])
                if s_link_temp2[i] < -sCutoff:
                    largest_locs.append(locs[loc])
                    largest_sites.append('/'.join(linked[i]))
                    largest_sel.append(s_link_temp2[i])
            s_temp1 = np.amax(s_nolink[loc], axis=0)
            s_temp2 = np.amin(s_nolink[loc], axis=0)
            for i in range(len(s_temp1)):
                if s_temp1[i] > sCutoff:
                    largest_locs.append(locs[loc])
                    largest_sites.append(allele_number[i])
                    largest_sel.append(s_temp1[i])
                if s_temp2[i] < -sCutoff:
                    largest_locs.append(locs[loc])
                    largest_sites.append(allele_number[i])
                    largest_sel.append(s_temp2[i])
        
        df_data = {
            'locations' : largest_locs,
            'sites'     : largest_sites,
            'selection' : largest_sel
        }
        df = pd.DataFrame(data=df_data)
        return df


def find_null_distribution(file, link_file, neutral_tol=0.01):
    """ Given the inferred coefficients over all time, find those that are ultimately inferred to be nearly neutral,
    and then collect the inferred coefficients for these sites at every time."""
    
    # loading and processing the data
    data            = np.load(file,      allow_pickle=True)
    linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in every region
    traj_data       = np.load(traj_file, allow_pickle=True)
    traj            = data['traj']
    mutant_sites    = data['mutant_sites']
    times           = data['times']
    allele_number   = data['allele_number']
    labels          = [get_label2(i) for i in allele_number]
    linked_all      = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    inferred        = data['selection']
    digits          = len(str(len(inferred)*len(inferred[0])))
    
    # finding first time a mutation was observed
    t_observed     = np.zeros(len(allele_number))
    alleles_sorted = np.sort(alelle_number)
    pos_all        = [np.searchsorted(alleles_sorted, mutant_sites[i]) for i in range(len(mutant_sites))]
    for i in range(len(traj)):
        positions   = pos_all[i]
        first_times = [times[i][np.nonzero(traj[i][:, j])[0][0]] for j in range(len(traj[i][0]))]
        for j in range(len(mutant_sites[i])):
            t_observed[positions[j]] = min(t_observed[positions[j]], first_times[j])
            
    t_init_link   = np.zeros(len(linked_sites))
    for i in range(len(labels)):
        if labels[i] in linked_all:
            for j in range(len(linked_sites)):
                if labels[i] in list(linked_sites[j]):
                    t_init_link[j] = min(t_init_link[j], t_observed[i])
        
    # finding total coefficients for linked sites and adding them together.
    inferred_link   = np.zeros((len(inferred), len(linked_sites)))
    inferred_nolink = []
    labels_nolink   = []
    t_init_nolink   = []
    for i in range(len(linked_sites)):
        for j in range(len(linked_sites[i])):
            if np.any(linked_sites[i][j]==np.array(labels)):
                inferred_link[:,i] += inferred[:, np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
    counter = 0
    for i in range(len(labels)):
        if labels[i] not in linked_all:
            inferred_nolink.append(inferred[:,i])
            labels_nolink.append(labels[i])
            t_init_nolink.append(t_observed[i])
            counter+=1
    nolink_new = np.zeros((len(inferred_nolink[0]), len(inferred_nolink)))
    for i in range(len(inferred_nolink)):
        nolink_new[:,i] = inferred_nolink[i]
    inferred_nolink = nolink_new
    
    # determing sites that are ultimately inferred to be nearly neutral
    inf_new       = np.concatenate((inferred_link, np.array(inferred_nolink)), axis=1)
    t_init        = np.concatenate((t_init_link, t_init_nolink))
    neutral_mask  = np.absolute(inf_new[-1])<neutral_tol
    L             = len([i for i in neutral_mask if i])
    inferred_neut = np.zeros((len(inf_new), L))
    t_init_new    = np.array(t_init)[neutral_mask]
    for i in range(len(inferred)):
        inferred_neut[i] = inf_new[i][neutral_mask]
    
    times_inf = data['times_inf']
    assert(len(inferred_neut)==len(times_inf))
    s_neutral = []
    for i in range(len(inferred_neut[0])):
        for j in range(len(inferred_neut)):
            if t_init_new[i] <= times_inf[j]:
                s_neutral.append(inferred_neut[j][i])
    inferred_neut = s_neutral
    
    inferred_neut = [inferred_neut[i][j] for i in range(len(inferred_neut)) for j in range(len(inferred_neut[i]))]
    
    # If a site is inferred to have a selection coefficient of exactly zero, this is because it is the reference nucleotide.
    inferred_neut = np.array(inferred_neut)[np.array(inferred_neut)!=0]
    
    return inferred_neut


def infer_fast(numerator, covariance, alleles, g=40):
    """Infer selection coefficients given the numerator, the covariance and the alleles"""
    alleles_unique   = np.unique([int(i[:-2]) for i in alleles])
    ref_seq, ref_tag = get_MSA(REF_TAG + '.fasta')
    ref_seq          = list(ref_seq[0])
    ref_poly         = np.array(ref_seq)[alleles_unique]
    g1               = g1 * (N * k) / (1 + (k / R))
    for i in range(len(covariance)):
        covariance[i, i] += g1
    
    # infer selection coefficients
    s      = linalg.solve(covariance, numerator, assume_a='sym')
    s_ind  = numerator / np.diag(covariance)
    errors = 1 / np.sqrt(np.absolute(np.diag(covariance)))

    # normalize reference nucleotide to zero
    L = int(len(s) / 5)
    selection  = np.reshape(s, (L, q))
    selection_nocovar = np.reshape(s_ind, (L, q))
    
    # normalize reference mutation to have selection coefficient of zero    
    s_new = []
    s_SL  = []
    for i in range(L):
        idx = NUC.index(ref_poly[i])
        temp_s    = selection[i]
        temp_s    = temp_s - temp_s[idx]
        temp_s_SL = selection_nocovar[i]
        temp_s_SL = temp_s_SL - temp_s_SL[idx]
        s_new.append(temp_s)
        s_SL.append(temp_s_SL)
    selection         = s_new
    selection_nocovar = s_SL

    selection         = np.array(selection).flatten()
    selection_nocovar = np.array(selection_nocovar).flatten()
    error_bars        = errors
    
    return selection, error_bars, selection_nocovar
    
    
def find_linked_coefficients(link_file, infer_file):
    """ Finds the sum of the coefficients for linked sites given a .npz inference file"""
    linked_sites  = np.load(link_file,  allow_pickle=True)
    data          = np.load(infer_file, allow_pickle=True)
    inferred      = data['selection']
    error         = data['error_bars']
    alleles       = data['allele_number']
    labels        = [get_label_new(i) for i in alleles]

    inferred_link = np.zeros(len(linked_sites))
    error_link    = np.zeros(len(linked_sites))
    for i in range(len(linked_sites)):
        for j in range(len(linked_sites[i])):
            if linked_sites[i][j] in list(labels):
                loc = list(labels).index(linked_sites[i][j])
                inferred_link[i] += inferred[loc]
                error_link[i]    += error[loc] ** 2
    error_link = np.sqrt(error_link)
    return inferred_link, error_link  


def linked_coefficients(linked_sites, mutant_sites, selection, error):
    """ Calculate the selection coefficients for the linked groups and the errors"""
    inferred_link = np.zeros(len(linked_sites))
    error_link    = np.zeros(len(linked_sites))
    for i in range(len(linked_sites)):
        for j in range(len(linked_sites[i])):
            if linked_sites[i][j] in list(mutant_sites):
                loc = list(mutant_sites).index(linked_sites[i][j])
                inferred_link[i] += selection[loc]
                error_link[i]    += error[loc] ** 2
    error_link = np.sqrt(error_link)
    return inferred_link, error_link


def bootstrap_variant_stats(directory, linked_sites, variant_names=None, out_file='bs-variant-stats'):
    """Finds the inferred coefficients for variants across a series of bootstrap inferences and then 
    calculates the mean/standard deviation"""
    selection = []
    errors    = []
    for file in os.listdir(directory):
        data = np.load(os.path.join(directory, file), allow_pickle=True)
        sel  = data['selection']
        muts = [get_label_new(i) for i in data['allele_number']]
        err  = data['error_bars']
        inf, err = linked_coefficients(linked_sites, muts, sel, err)
        selection.append(inf)
        errors.append(err)
    mean_s  = np.mean(selection, axis=0)
    std_dev = np.std(selection,  axis=0)
    avg_dev = np.sum([np.abs(i - mean_s) for i in selection], axis=0) / len(selection)
    if variant_names==None:
        variant_names = [f'Group {i}' for i in range(len(linked_sites))]
    f = open(out_file + '.csv', 'w')
    f.write('variant,mean_selection,standard_deviation,average_deviation,sites\n')
    for i in range(len(linked_sites)):
        sites = ' '.join(linked_sites[i])
        f.write(f'{variant_names[i]},{mean_s[i]},{std_dev[i]},{avg_dev[i]},{sites}\n')
    f.close()


def find_bootstrap_stats(directory, out_file, mutant_sites=None):
    """Find the mean and standard deviation of the inferred coefficients across a series of bootstrap inferences"""
    selection = []
    errors    = []
    alleles   = []
    mut_sites = np.load(os.path.join(directory, os.listdir(directory)[0]), allow_pickle=True)['allele_number']
    num_bootstraps = len(os.listdir(directory))
    for file in os.listdir(directory):
        data = np.load(os.path.join(directory, file), allow_pickle=True)
        selection.append(data['selection'])
        errors.append(data['error_bars'])
        alleles.append(data['allele_number'])
    if mutant_sites is not None:
        alleles_sorted = np.argsort(mutant_sites)
        #positions      = [np.searchsorted(np.array(mutant_sites)[alleles_sorted], i) for i in alleles]
        #positions      = [alleles_sorted[i] for i in positions]
        #print(positions[0])
        #print(mutant_sites)
        #print(alleles[0][positions[0]])
        new_s   = np.zeros((num_bootstraps, len(mutant_sites)))
        new_err = np.zeros((num_bootstraps, len(mutant_sites)))
        for i in range(num_bootstraps):
            mask = np.isin(alleles[i], mutant_sites)
            temp_muts  = alleles[i][mask]
            positions  = np.searchsorted(np.array(mutant_sites)[alleles_sorted], temp_muts)
            positions  = alleles_sorted[positions]
            
            temp_s   = selection[i][mask]
            temp_err = errors[i][mask]
            for j in range(len(temp_muts)):
                new_s[i, positions[j]]   = temp_s[j]
                new_err[i, positions[j]] = temp_err[j]
            #for j in range(len(positions[i])):
            #    new_s[i, positions[i][j]]   = selection[i][j]
            #    new_err[i, positions[i][j]] = errors[i][j]
        selection = new_s
        errors    = new_err
        mut_sites = mutant_sites
    mean_s  = np.mean(selection, axis=0)
    std_dev = np.std(selection,  axis=0)
    mean_error = np.mean(errors, axis=0)
    f = open(out_file + '.csv', 'w')
    f.write('mutant_site,mean_selection,standard_deviation,mean_error\n')
    for i in range(len(mut_sites)):
        f.write(f'{mut_sites[i]},{mean_s[i]},{std_dev[i]},{mean_error[i]}\n')
    f.close()


# _ _ _ FUNCTIONS FOR CONVERTING BETWEEN BINARY SEQUENCE AND LABELED ONE _ _ _ #

def label_to_binary(seq, allele_number):
    """ Transforms a labeled sequence into a binary one. """
    
    new_seq = np.zeros(len(allele_number))
    for i in range(len(seq)):
        loc = np.where(allele_number==seq[i])
        new_seq[loc] = 1
    return new_seq


def binary_to_labeled(seq, mutant_sites):
    """ Transforms a binary sequence into a labeled one. """
    
    return np.array([mutant_sites[i] for i in range(len(seq)) if seq[i]==1])


def construct_sVec(sVec, allele_number):
    """ Constructs a binary sVec from a labeled one. """
    
    new_sVec = []
    for i in range(len(sVec)):
        sVec_temp = []
        for j in range(len(sVec[i])):
            sVec_temp.append(label_to_binary(sVec[i][j], allele_number))
        new_sVec.append(sVec_temp)
    return new_sVec

# ^ ^ ^ FUNCTIONS FOR CONVERTING BETWEEN BINARY SEQUENCE AND LABELED ONE ^ ^ ^  #


def clip_sequences(file, outfile, start=0, end=0):
    """ Clip the sequences start and end times and resave the file"""
    
    data = np.load(file, allow_pickle=True)
    nVec = data['nVec']
    sVec = data['sVec']
    mutant_sites = data['mutant_sites']
    times = data['times']
    if end > 0:
        nVec  = nVec[:-end]
        sVec  = sVec[:-end]
        times = times[:-end]
    if start > 0:
        nVec  = nVec[start:]
        sVec  = sVec[start:]
        times = times[start:]
    f = open(outfile, mode='w+b')
    np.savez_compressed(f, nVec=nVec, sVec=sVec, mutant_sites=mutant_sites, times=times)
    f.close()
    
    
def combine_nonsyn_files(nonsyn_file, alternate_orf_file, out_file):
    """ Combine the information in the regular nonsynonymous file with the information in the alternative
    reading frames."""
    
    data1     = np.load(nonsyn_file, allow_pickle=True)
    types     = data1['types']
    locs      = data1['locations']
    data2     = np.load(alternate_orf_file, allow_pickle=True)
    types_alt = data2['types']
    types_new = []
    for i in range(len(locs)):
        if types[i] == 'S' and types_alt[i] == 'NS':
            print('yes')
            types_new.append('NNS')
        else:
            types_new.append(types[i])
    f = open(out_file + '.npz', mode='w+b')
    np.savez_compressed(f, locations=locs, types=types_new)
    f.close()

    
def trajectory_calc_20e_eu1(nVec, sVec, mutant_sites_samp, d=5):
    """ Calculates the frequency trajectories"""
    variant_muts  = ['NSP16-199-2-C', 'NSP1-60-2-C', 'NSP3-1189-2-T', 'M-93-2-G', 'N-220-1-T', 'ORF10-30-0-T', 'S-222-1-T']
    variant_nucs  = [i[-1] for i in variant_muts]
    variant_sites = [list(mutant_sites_samp).index(i[:-2]) for i in variant_muts]
    Q = np.ones(len(nVec))
    for t in range(len(nVec)):
        if len(nVec[t]) > 0:
            Q[t] = np.sum(nVec[t])
    single_freq_s = np.zeros(len(nVec))
    for t in range(len(nVec)):
        for j in range(len(sVec[t])):
            seq_is_var = True
            for i in range(len(variant_muts)):
                if NUC[sVec[t][j][variant_sites[i]]] != variant_nucs[i]:
                    seq_is_var = False
                    break
            if seq_is_var:
                single_freq_s[t] += nVec[t][j] / Q[t]
    return single_freq_s


def get_noncanonical_orfs(i):
    # For a sequence index i, find the noncanonical reading frames, protein number, and nucleotide in codon number
    
    i            = int(i)
    proteins     = []
    start_inds   = [21744, 25814, 25457, 25524, 25596, 28284, 28734]
    end_inds     = [21860, 25879, 25579, 25694, 25694, 28574, 28952]
    labels       = ['ORF2b', 'ORF3b', 'ORF3c', 'ORF3d', 'ORF3d-2', 'ORF9b', 'ORF9c']
    #start_codons = ['CUG', 'AUG', 'ACG', 'AUC', 'AUG', 'AUG', 'AUG', 'AUG', 'AUU', 'AUU', 'UUG', 'AUC', 'AUG', 'AUG', 'AUG']
    for j in range(len(start_inds)):
        if (start_inds[j]<=i<=end_inds[j]):
            proteins.append(labels[j] + '-' + str(int((i - start_inds[j] + 1) / 3 ) + 1) + '-' + str((i - start_inds[j] + 1) % 3))
    return proteins


def get_noncanonical_codon_start_index(i):
    # Given a sequence index i, determine the index of the first nucleotide in the codon in each of the noncanonical reading frames.
    
    i            = int(i)
    start_inds   = [21744, 25814, 25457, 25524, 25596, 28284, 28734]
    end_inds     = [21860, 25879, 25579, 25694, 25694, 28574, 28952]
    labels       = ['ORF2b', 'ORF3b', 'ORF3c', 'ORF3d', 'ORF3d-2', 'ORF9b', 'ORF9c']
    codon_inds   = []
    for j in range(len(start_inds)):
        if (start_inds[j]<=i<=end_inds[j]):
            codon_inds.append(i - (i - start_inds[j] + 1)%3)
    return codon_inds    

    
def classify_mutations_noncanonical(ref_seq, poly_sites, likely_alleles, likely_mutations):
    """ Given the sites that are polymorphic in a specific region, determine if the mutations are synonymous or nonsynonymous. """
    
    sites, states, ns_counter = [], [], []
    for site in poly_sites:
        start_inds = get_noncanonical_codon_start_index(site)
        sites.append(get_noncanonical_orfs(site))
        if len(start_inds)>0:
            counter = 0
            mut_type = "S"
            for i in start_inds:
                ref_codon  = ref_seq[i:i+3]
                mut_codon = [j for j in ref_codon]
                indices = np.arange(i, i+3)
                mut_codon[list(indices).index(site)] = likely_mutations[poly_sites.index(site)]
                for j in range(len(indices)):
                    if indices[j] in np.array(poly_sites) and indices[j]!=site:
                        mut_codon[j] = likely_alleles[poly_sites.index(indices[j])]
                if codon2aa(ref_codon)!=codon2aa(mut_codon):
                    mut_type = "NS"
                    counter += 1
            ns_counter.append(counter)
            states.append(mut_type)
        else:
            ns_counter.append(0)
            states.append('None')
    return sites, states, ns_counter    
        
    
def separate_by_protein(inf_file, out_file=None):
    """ Takes the inferred coefficients and separates them according to what protein they belong to"""
    data     = np.load(inf_file, allow_pickle=True)
    alleles  = data['allele_number']
    labels   = [get_label2(i) for i in alleles]
    inferred = data['selection']
    proteins = ['ORF3a', 'E', 'ORF6', 'ORF7a', 'ORF7b', 'S', 'N', 'M', 'ORF8', 'ORF10', 'NSP1-', 'NSP2',
                'NSP3', 'NSP4', 'NSP5', 'NSP6', 'NSP7', 'NSP8', 'NSP9', 'NSP10', 'NSP12', 'NSP13', 'NSP14', 
                'NSP15', 'NSP16']
    inf_new   = []
    label_new = []
    for p in proteins:
        temp_inf = []
        temp_lab = []
        for i in range(len(inferred)):
            if labels[i][:len(p)]==p:
                temp_inf.append(inferred[i])
                temp_lab.append(labels[i])
        inf_new.append(temp_inf)
        label_new.append(temp_lab)
    proteins_new = []
    for p in proteins:
        if p.find('-')!=-1:
            proteins_new.append(p[:p.find('-')])
        else:
            proteins_new.append(p)
    proteins = proteins_new
    if out_file:
        f = open(out_file + '.csv', mode='w')
        f.write('protein,site,selection')
        for i in range(len(inf_new)):
            for j in range(len(inf_new[i])):
                f.write('%s,%s,%df' % (proteins[i], label_new[i][j], inf_new[i][j]))
        f.close()
    return proteins, label_new, inf_new


# ˅ ˅ ˅ ˅ ˅ NEW VERSIONS ˅ ˅ ˅ ˅ ˅ #

def find_site_index_file(filepath):
    """ Given a sequence file find the correct corresponding file that has the site names"""
    directory, file = os.path.split(filepath)
    if file.find('---')==-1:
        return filepath[:-4] + '-sites.csv'
    else:
        return filepath[:filepath.find('---')] + '-sites.csv' 
    
      
def calculate_selection(tv_inf_dir, link_file, cutoff=0.1318, start_date=None):
    """Finds all individual sites and groups of linked sites that would be inferred to be larger than
    the cutoff in any region at any time"""
    linked_sites = np.load(link_file, allow_pickle=True)
    linked_all   = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    locations = []
    s_large   = []
    sites     = []
    for file in os.listdir(tv_inf_dir):
        data  = np.load(os.path.join(tv_inf_dir, file), allow_pickle=True)
        s     = data['selection']
        t_inf = data['time_infer']
        muts  = [get_label_new(i) for i in data['allele_number']]
        if start_date!=None:
            if start_date > t_inf[-1]:
                continue
            elif start_date in t_inf:
                s = s[list(t_inf).index(start_date):]
        
        muts_nolink = [i for i in muts if i not in linked_all]
        s_link      = np.zeros((len(linked_sites), len(s)))
        s_nolink    = np.zeros((len(muts_nolink),  len(s)))
        for i in range(len(muts_nolink)):
            s_nolink[i] += s[:, muts.index(muts_nolink[i])]
        for i in range(len(linked_sites)):
            for j in range(len(linked_sites[i])):
                if linked_sites[i][j] in muts:
                    s_link[i] += s[:, muts.index(linked_sites[i][j])]
        s_link   = np.amax(s_link, axis=1)
        s_nolink = np.amax(s_nolink, axis=1)
        
        # masking out mutations and groups with selection coefficient smaller than cutoff
        link_mask = s_link >= cutoff
        muts_link = np.array(linked_sites)[link_mask]
        s_link_large = s_link[link_mask]
        for i in range(len(muts_link)):
            locations.append(file[:-4])
            sites.append(muts_link[i])
            s_large.append(s_link_large[i])
        nolink_mask = s_nolink >= cutoff
        muts_nolink = np.array(muts_nolink)[nolink_mask]
        s_nolink_large = s_nolink[nolink_mask]
        for i in range(len(muts_nolink)):
            locations.append(file[:-4])
            sites.append(muts_nolink[i])
            s_large.append(s_nolink_large[i])
    return locations, sites, s_large

    
def calculate_selection2(tv_inf_dir, link_file, cutoff=0.1318, start_date=None, elim_prior=False):
    """Finds all individual sites and groups of linked sites that would be inferred to be larger than
    the cutoff in any region at any time"""
    linked_sites = np.load(link_file, allow_pickle=True)
    linked_all   = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    locations = []
    s_large   = []
    sites     = []
    for file in os.listdir(tv_inf_dir):
        data  = np.load(os.path.join(tv_inf_dir, file), allow_pickle=True)
        s     = data['selection']
        t_inf = data['time_infer']
        muts  = [get_label_new(i) for i in data['allele_number']]
        if start_date!=None:
            if start_date > t_inf[-1]:
                continue
        #    elif start_date in t_inf:
        #        s = s[list(t_inf).index(start_date):]
        
        muts_nolink = [i for i in muts if i not in linked_all]
        s_link      = np.zeros((len(linked_sites), len(s)))
        s_nolink    = np.zeros((len(muts_nolink),  len(s)))
        for i in range(len(muts_nolink)):
            s_nolink[i] += s[:, muts.index(muts_nolink[i])]
        for i in range(len(linked_sites)):
            for j in range(len(linked_sites[i])):
                if linked_sites[i][j] in muts:
                    s_link[i] += s[:, muts.index(linked_sites[i][j])]
        #s_link   = np.amax(s_link, axis=1)
        #s_nolink = np.amax(s_nolink, axis=1)
        
        # masking out mutations and groups with selection coefficient smaller than cutoff
        #link_mask = s_link >= cutoff
        new_mask  = []
        if start_date!=None:
            for i in range(len(linked_sites)):
                if np.amax(s_link[i]) < cutoff:
                    continue
                if t_inf[np.where(s_link[i]>=cutoff)[0][0]] < start_date:
                    continue
                else:
                    new_mask.append(i)
            if len(new_mask) == 0:
                continue
            link_mask = np.array(new_mask)
        else:
            link_mask = np.amax(s_link, axis=1) >= cutoff
                
        muts_link = np.array(linked_sites)[link_mask]
        s_link_large = s_link[link_mask]
        for i in range(len(muts_link)):
            locations.append(file[:-4])
            sites.append(muts_link[i])
            s_large.append(s_link_large[i])
            
        #nolink_mask = s_nolink >= cutoff
        new_mask    = []
        if start_date!=None:
            for i in range(len(muts_nolink)):
                if np.amax(s_nolink[i]) < cutoff:
                    continue
                if t_inf[np.where(s_nolink[i]>=cutoff)[0][0]] < start_date:
                    continue
                else:
                    new_mask.append(i)
            if len(new_mask) == 0:
                continue
            nolink_mask = np.array(new_mask)
        else:
            nolink_mask = np.amax(s_nolink, axis=1) >= cutoff
        
        muts_nolink = np.array(muts_nolink)[nolink_mask]
        s_nolink_large = s_nolink[nolink_mask]
        for i in range(len(muts_nolink)):
            locations.append(file[:-4])
            sites.append(muts_nolink[i])
            s_large.append(s_nolink_large[i])
    return locations, sites, s_large    
    

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


def slice_time_series(file, times_new, out_file):
    """ Cuts off the sequences at the beginning and end and resaves the data. """
    df = pd.read_csv(file)
    df_new = df[(df['date'] >= times_new[0]) & (df['date'] <= times_new[-1])]
    df_new.to_csv(out_file + '.csv', index=False)


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
    
# ^ ^ ^ NEW VERSIONS ^ ^ ^ #

    
def find_sampling_intervals(in_file, window=5, min_seqs=20, max_dt=5, min_range=20, end_cutoff=3):
    """ Given the output file of count_sequences, determine the time intervals that have good enough sampling
    to be used for inference. Expects an input .npz file.
    CHECK: last_time_used is supposed to set it up so that the intervals are not allowed to overlap."""
    
    data  = np.load(in_file, allow_pickle=True)
    nVec  = data['nVec']
    sVec  = data['sVec']
    times = data['times']
    a = [np.sum(nVec[t]) for t in range(len(nVec))]
    df = pd.DataFrame.from_dict({'counts' : a, 'times' : times})
    counts = [np.sum(nVec[t]) for t in range(len(nVec))]
    n_intervals = 0
    new_nVec  = []
    new_sVec  = []
    new_times = []
    cumulative_counts = 0
    last_time_used    = 0
    recording = False
    for t in range(len(times)):
        cumulative_counts += counts[t]
        if t > window - 1:
            cumulative_counts -= counts[t - window]
        if cumulative_counts >= min_seqs:
            if not recording:
                if counts[t - window + 1]!=0 and (t - window + 1)>=last_time_used:
                    recording  = True
                    zero_dt    = 0
                    nVec_temp  =  list(nVec[t - window + 1 : t + 1])
                    sVec_temp  =  list(sVec[t - window + 1 : t + 1])
                    times_temp =  list(times[t - window + 1 : t + 1])
                else:
                    continue
            else:
                nVec_temp.append(nVec[t])
                sVec_temp.append(sVec[t])
                times_temp.append(times[t])
            if counts[t]==0:
                zero_dt += 1
                if zero_dt >= max_dt:
                    recording = False
                    cumulative_counts = np.sum(counts[t - window + 1:t + 1])
                    zero_dt = 0
                    if len(nVec_temp) - max_dt - end_cutoff >= min_range:
                        new_nVec.append(nVec_temp[:-max_dt-end_cutoff])
                        new_sVec.append(sVec_temp[:-max_dt-end_cutoff])
                        new_times.append(times_temp[:-max_dt-end_cutoff])
                        last_time_used = list(times).index(new_times[-1][-1])
                        n_intervals += 1
            else:
                zero_dt = 0
        else:
            if recording:
                recording = False
                cumulative_counts = np.sum(counts[t - window + 1:t + 1])
                if len(nVec_temp) - end_cutoff - zero_dt >= min_range:
                    if zero_dt > 0:
                        new_nVec.append(nVec_temp[:-zero_dt-end_cutoff])
                        new_sVec.append(sVec_temp[:-zero_dt-end_cutoff])
                        new_times.append(times_temp[:-zero_dt-end_cutoff])
                        last_time_used = list(times).index(new_times[-1][-1])
                    elif end_cutoff!=0:
                        new_nVec.append(nVec_temp[:-end_cutoff])
                        new_sVec.append(sVec_temp[:-end_cutoff])
                        new_times.append(times_temp[:-end_cutoff])
                        last_time_used = new_times[-1][-1]
                    else: 
                        new_nVec.append(nVec_temp)
                        new_sVec.append(sVec_temp)
                        new_times.append(times_temp)
                        last_time_used = list(times).index(new_times[-1][-1])
                    zero_dt      = 0
                    n_intervals += 1
        if times[t] == times[-1] and recording and len(nVec_temp) >= min_range:
            new_nVec.append(nVec_temp)
            new_sVec.append(sVec_temp)
            new_times.append(times_temp)
            n_intervals +=1
    
    ### THERE IS NO NEED TO RETURN new_nVec or new_sVec
    return n_intervals, new_nVec, new_sVec, new_times


def shorten_times(file, start_cut, end_cut, out_file):
    " Cuts off the sequences at the beginning and end and resaves the data. "
    
    data = np.load(file, allow_pickle=True)
    mutant_sites = data['mutant_sites']
    if isinstance(mutant_sites[0], list) or isinstance(mutant_sites[0], np.ndarray):
        mutant_sites = mutant_sites[0]
    if 'ref_sites' in data:
        ref_sites = data['ref_sites']
    else:
        ref_sites = mutant_sites
    
    times_all = data['times']
    start_cut = list(times_all).index(times_all[0] + start_cut)
    end_cut   = list(times_all).index(times_all[-1] - end_cut)
    #### ADDED NEXT LINE IN ORDER TO REMOVE INCONSISTENCY BETWEEN THE TWO LABELING INDICES AND BECUASE mutant_sites IS NO LONGER USEFUL
    mutant_sites  = ref_sites 
    if end_cut!=0:
        times = data['times'][start_cut:end_cut]
        nVec  = data['nVec'][start_cut:end_cut]
        sVec  = data['sVec'][start_cut:end_cut]
    else:
        times = data['times'][start_cut:]
        nVec  = data['nVec'][start_cut:]
        sVec  = data['sVec'][start_cut:]
    times_all = data['times']
    nVec_all  = data['nVec']
    print(f'length of new time series = {len(times)}')
    print(f'length of original times series = {len(times_all)}')
    print(f'length of full nVec = {len(nVec_all)}')
    f = open(out_file+'.npz', mode='w+b')
    np.savez_compressed(f, times=times, mutant_sites=mutant_sites, nVec=nVec, sVec=sVec, ref_sites=ref_sites)
    f.close()
            
            
def restrict_time_series(input_file, out_folder, window=5, max_dt=5, min_seqs=20, min_range=20, end_cutoff=0, ignore_short=True):
    """finds the intervals that have good sampling in a region, cuts the data array to these 
    intervals, and resaves the file."""
    ref_date      = datetime.date(2020, 1, 1)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    filename = os.path.split(input_file)[1]
    print(f'working on file {filename}')
    full_times = np.load(input_file, allow_pickle=True)['times']
    if filename.find('None') != -1:
        location = filename[:filename.find('None') - 1]
    else:
        location = filename[:np.amin([filename.find(i) for i in [str(j) for j in np.arange(10)]]) - 1]
    if location.find('.') != -1:
        location = location[:location.find('.')]
    regions = filename.split('-')
    if location.split('-')[1] in ['united kingdom', 'united_kingdom'] and ignore_short:
        n_intervals = 1
        data        = np.load(input_file, allow_pickle=True)
        nVec        = [data['nVec']]
        sVec        = [data['sVec']]
        times       = [data['times']]
    else:
        n_intervals, nVec, sVec, times = find_sampling_intervals(input_file, max_dt=max_dt, window=window,
                                                                 min_seqs=min_seqs, min_range=min_range, 
                                                                 end_cutoff=end_cutoff)
        
    for i in range(n_intervals):
        times_temp = times[i]
        start_year, end_year   = (ref_date + datetime.timedelta(int(times_temp[0]))).year,   (ref_date + datetime.timedelta(int(times_temp[-1]))).year
        start_month, start_day = (ref_date + datetime.timedelta(int(times_temp[0]))).month,  (ref_date + datetime.timedelta(int(times_temp[0]))).day
        end_month, end_day     = (ref_date + datetime.timedelta(int(times_temp[-1]))).month, (ref_date + datetime.timedelta(int(times_temp[-1]))).day
        start_cut = times_temp[0]  - full_times[0]
        end_cut   = full_times[-1] - times_temp[-1]
        out_file = location + '---' + str(start_year) + '-' + str(start_month) + '-' + str(start_day) + '-' + str(end_year) + '-' + str(end_month) + '-' + str(end_day)
        shorten_times(input_file, start_cut, end_cut, os.path.join(out_folder, out_file))
        
        
def variant_selection(sites, inf_file, out_file=None, sort=True):
    """make a csv file containing the sites and selection coefficients for each mutation in a variant"""
    if inf_file[-4:]=='.npz':
        data = np.load(inf_file, allow_pickle=True)
        muts = [get_label2(i) for i in data['allele_number']]
        s    = data['selection']
    
    var_s = []
    labs  = []
    for i in sites:
        if i in muts:
            labs.append(i)
            var_s.append(s[muts.index(i)])
    if sort:        
        sorter = np.argsort(np.abs(var_s))[::-1]
        var_s  = np.array(var_s)[sorter]
        labs   = np.array(labs)[sorter]
        
    df = pd.DataFrame.from_dict({'site' : labs, 'selection' : var_s})
    df.to_csv(out_file + '.csv')
    
    
def save_traj_selection(tv_inf, group=None, traj_site=None, out_file=None):
    """Given a file containing selection coefficients and trajectories over time, save them in a csv."""
    data       = np.load(tv_inf, allow_pickle=True)
    times      = data['times']
    inf_times  = data['times_inf']
    selection  = data['selection']
    alleles    = data['allele_number']
    mutants    = data['mutant_sites']
    traj       = data['traj']
    labels     = np.array([get_label_new(i) for i in alleles])
    #print(inf_times, len(inf_times))
    #print(np.shape(selection))
    print(tv_inf)
    
    # Eliminate the last time point in the trajectories because the covariance matrix will be 1 shorter (since integrated)
    traj       = [i[:-1] for i in traj]
    times      = [i[:-1] for i in times]
    
    idxs_group = np.array([list(labels).index(i) for i in group])
    s_group    = np.sum(selection[:, idxs_group], axis=1)
    if traj_site is None:
        traj_site = group[0]
        
    new_times  = []
    new_traj   = []
    for i in range(len(times)):
        mask = np.isin(times[i], inf_times)
        new_times.append(np.array(times[i])[mask])
        new_traj.append(np.array(traj[i])[mask])
    times      = new_times
    traj       = new_traj
    
    t_present  = list(np.unique([times[i][j] for i in range(len(times)) for j in range(len(times[i]))]))
    #t_present.pop(t_present.index(363))
    t_present  = np.array(t_present)
    #print(t_present)
    t_mask     = np.isin(inf_times, t_present)
    s_group    = s_group[t_mask]
    
    traj_full  = np.zeros(len(s_group))
    pos_full   = [np.searchsorted(t_present, i) for i in times]
    for i in range(len(traj)):
        t_temp    = times[i]
        traj_temp = traj[i]
        muts_temp = [get_label_new(j) for j in mutants[i]]
        pos_temp  = pos_full[i]
        if traj_site in muts_temp:
            idx       = muts_temp.index(traj_site)
            site_traj = traj_temp[:, idx]
            for j in range(len(t_temp)):
                traj_full[pos_temp[j]] = site_traj[j]
    
    f = open(out_file + '.csv', mode='w')
    f.write('time,frequency,selection\n')
    for i in range(len(t_present)):
        f.write('%d,%.8f,%.8f\n' % (t_present[i], traj_full[i], s_group[i]))
    f.close()


### FUNCTIONS FOR THE SMOOTHING DECONVOLUTION ALGORITHM    


def msa_to_unique(file, out_folder, output=False):
    """ Given a data file (the output of save_MPL_alignment), find the unique sequences and save them to different csv files. 
    In order to use smoothing algorithm."""
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    times, seqs = get_sequences(file)
    n_unique    = []
    seqs_unique = []
    t_grouped   = []
    for i in range(len(seqs)):
        if list(seqs[i]) in seqs_unique:
            n_unique[np.where(np.array(seqs_unique)==np.array(seqs[i]))[0][0]] += 1
            t_grouped[np.where(np.array(seqs_unique)==np.array(seqs[i]))[0][0]].append(times[i])
        else:
            seqs_unique.append(' '.join([str(seqs[i][j]) for j in range(len(seqs[i]))]))
            n_unique.append(1)
            t_grouped.append([times[i]])
    t_unique = [[np.unique(i) for i in t_grouped[j]] for j in range(len(t_grouped))]
    counts   = [np.sum([np.isin(t_grouped[i], t_unique[i][j]) for j in range(len(t_unique[i]))]) for i in range(len(t_unique))]
    t_str    = [' '.join([str(i) for i in t_unique[j]]) for j in range(len(t_unique))]
    df_full  = pd.DataFrame.from_dict({'sequence' : seqs_unique, 'times' : t_str, 'counts' : counts})
    df_full.to_csv(os.path.join(out_folder, 'full.csv'), index=False)
    
    for i in range(len(n_unique)):
        df = pd.DataFrame.from_dict({'times' : t_unique[i], 'counts' : counts[i]})
        df.to_csv(os.path.join(out_folder, f'{i}.csv'), index=False)
        
    if output:
        return t_grouped, counts, n_unique
    
    
def msa_to_identical(file, out_folder, output=False, timed=1):
    """ Given a data file (the output of save_MPL_alignment), find the groups of different sequences and save their time series csv files. 
    In order to use smoothing algorithm. file can also be a list of files."""
    
    if timed>0:
        t_start = timer()
    
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    multiple_dir = os.path.join(out_folder, 'nonunique')
    if not os.path.exists(multiple_dir):
        os.mkdir(multiple_dir)
    
    
    ### UPDATE SO THAT SEQUENCES FROM DIFFERENT TIME SERIES IN THE SAME REGION ARE MADE THE SAME LENGTH ###
    if not isinstance(file, list) and not isinstance(file, np.ndarray): files = [file]
    else: files = file
    seqs_unique = []
    t_grouped   = []
    counts      = []
    for file in files:
        data = np.load(file, allow_pickle=True)
        sVec      = data['sVec']
        nVec      = data['nVec']
        times     = data['times']
        muts      = data['mutant_sites']
        np.save(os.path.join(out_folder, 'mutant_sites.npy'), muts)
        for t in range(len(sVec)):
            for i in range(len(sVec[t])):
                seq = ''.join(list(np.array(sVec[t][i], dtype=str)))
                if seq in seqs_unique:
                    idx = seqs_unique.index(seq)
                    t_grouped[idx].append(times[t])
                    counts[idx].append(nVec[t][i])
                else:
                    seqs_unique.append(seq)
                    t_grouped.append([times[t]])
                    counts.append([nVec[t][i]])
        
    # separate sequences that only appear once from the rest
    times_ind   = []
    seqs_ind    = []
    times_mult  = []
    seqs_mult   = []
    counts_mult = []
    for i in range(len(counts)):
        if np.sum(counts[i])==1:
            times_ind.append(t_grouped[i][0])
            seqs_ind.append(seqs_unique[i])
        else:
            times_mult.append(t_grouped[i])
            seqs_mult.append(seqs_unique[i])
            counts_mult.append(counts[i])
            
    # save mutant sites file
    
    # save sequences that only appear once in a region in a separate csv file
    df_ind = pd.DataFrame.from_dict({'times' : times_ind, 'sequences' : seqs_ind})
    df_ind.to_csv(os.path.join(out_folder, 'unique-sequences.csv'), index=False)
    
    # save sequences that appear multiple times, along with their index, so they can be reattached with the counts after smoothing
    df_idx = pd.DataFrame.from_dict({'index' : np.arange(len(counts_mult)), 'sequences' : seqs_mult})
    df_idx.to_csv(os.path.join(out_folder, 'sequence-index.csv'), index=False)
    
    # save sequences that appear multiple times in separate csv files to be smoothed
    for i in range(len(seqs_mult)):
        df = pd.DataFrame.from_dict({'times' : times_mult[i], 'counts' : counts_mult[i]})
        df.to_csv(os.path.join(multiple_dir, f'{i}.csv'), index=False)
        
    if output:
        return t_grouped, counts

    if timed>0:
        t_end = timer()
        t_tot = t_end - t_start
        print(f'total time = {t_tot} seconds')
        print(f'total time = {float(t_tot)/60} minutes')
        
        
def smoothing_to_MPL_alt(smooth_dir, prob_dist, nosmooth_file, out_file, timed=1):
    """ Takes the output directory containing files that have times and distributions resulting from the deconvolution method,
    and outputs a file that can be read by the inference script.
    prob_dist is a list containing the probability distribution for a single sequence.
    nosmooth file is the .npz file containing the data before smoothing."""
    
    if timed>0:
        t_start = timer()
    
    # get the mutant sites from the non-smoothed data file
    muts        = np.load(nosmooth_file, allow_pickle=True)[0]
    
    # load the file that identifies sequences with an index
    df_idx      = pd.read_csv(os.path.join(smooth_dir, 'sequence-index.csv'))
    idxs        = list(df_idx['index'])
    seqs_mult   = list(df_idx['sequences'])
    
    # load the file containing sequences that only appear once
    df_ind      = pd.read_csv(os.path.join(smooth_dir, 'unique-sequences.csv'))
    if 'sequences' in df_ind.columns.values:
        seqs_ind = list(df_ind['sequences'])
    else:
        seqs_ind = list(df_ind['seqeunces'])
    times_ind   = list(df_ind['times'])
    times_ind   = [np.arange(times_ind[i] - len(prob_dist) + 1, times_ind[i] + 1) for i in range(len(times_ind))]
    
    # load the data for the sequences that appear multiple times
    times_mult  = np.empty(len(idxs), dtype=object)
    counts_mult = np.empty(len(idxs), dtype=object)
    seqs_new    = np.empty(len(idxs), dtype=object)
    nonunique_dir = os.path.join(smooth_dir, 'nonuniquewrite')
    for file in os.listdir(nonunique_dir):
        identifier  = int(file[:file.find('.csv')])
        df_temp     = pd.read_csv(os.path.join(nonunique_dir, file))
        times_temp  = list(df_temp['bp.epoch'])
        counts_temp = list(df_temp['bp.upperbound'])
        seq_idx     = idxs.index(identifier)
        times_mult[seq_idx]  = times_temp
        counts_mult[seq_idx] = counts_temp
    
    print(times_ind)
    print(times_mult)
    # find all times and set up arrays to be filled with sequences and counts
    times_all = np.sort(np.unique([times_ind[i][j]  for i in range(len(times_ind))  for j in range(len(times_ind[i]))] + 
                                  [times_mult[i][j] for i in range(len(times_mult)) for j in range(len(times_mult[i]))]))
    times_all = list(times_all)

    # create the sequence and count arrays to be used for MPL
    #nVec      = [[] for i in range(len(times_all))]
    #sVec      = [[] for i in range(len(times_all))]
    
    # add the sequences that only appear once
    #for i in range(len(times_ind)):
    #    time_idxs = [times_all.index(j) for j in times_ind[i]]
    #    for j in range(len(times_ind[i])):
    #        sVec[times_idxs[j]].append(list(np.array(list(seqs_ind[i]), dtype=int)))
    #        nVec[times_idxs[j]].append(prob_dist[j])

    # add the sequences that appear multiple times
    #for i in range(len(times_mult)):
    #    time_idxs = [times_all.index(j) for j in times_mult[i]]
    #    for j in range(len(times_mult[i])):
    #        sVec[times_idxs[j]].append(list(np.array(list(seqs_mult[i]), dtype=int)))
    #        nVec[times_idxs[j]].append(counts_mult[i][j])
            
    # save the file
    #f = open(out_file + '.npz', mode='wb')
    #np.savez_compressed(f, mutant_sites=mutant_sites, nVec=nVec, sVec=sVec)
    
    if timed>0:
        t_end = timer()
        print('total time = ', t_end - t_start)

    # make .csv file instead of nVec and sVec
    headers = 'sequence,' + ','.join(np.array(times_all, dtype=str)) + '\n'
    g = open(out_file + '.csv', mode='w')
    g.write(headers)
    for i in range(len(seqs_ind)):
        time_idxs   = [times_all.index(j) for j in times_ind[i]]
        counts_temp = np.zeros(len(times_all))
        for j in range(len(time_idxs)):
            counts_temp[time_idxs[j]] += prob_dist[j]
        counts_temp = np.array(counts_temp, dtype=str)
        counts_str = ','.join(counts_temp)
        g.write(f'{seqs_ind[i]},{counts_str}\n')
    for i in range(len(seqs_mult)):
        time_idxs   = [times_all.index(j) for j in times_mult[i]]
        counts_temp = np.zeros(len(times_all))
        for j in range(len(time_idxs)):
            counts_temp[time_idxs[j]] += counts_mult[i][j]
        counts_temp = np.array(counts_temp, dtype=str)
        counts_str = ','.join(counts_temp)
        g.write(f'{seqs_mult[i]},{counts_str}\n')
    g.close()
    
    
def smoothing_to_MPL2(mut_file, seq_file, out_file=None, max_seqs=50000):
    """ Converts a .csv file of sequences (produced from the function above "smoothing_to_MPL_alt") into nVec/sVec files"""
    
    status_name = 'deconv-status.csv'
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
    
    def convert_format(df):
        """ Given a dataframe with a sequence column and times columns, where the counts for sequence i at time j are in cell ij,
        convert this to a vector of counts at each time for each sequence and a vector of sequences at each time."""
        times = list(df.columns.values[1:])
        T     = len(df.columns.values) - 1
        seqs  = list(df['sequence'])
        nVec  = [[] for i in range(T)]
        sVec  = [[] for i in range(T)]
        for t, counts in df.iteritems():
            if t=='sequence':
                continue
            indices = np.array(list(counts)).nonzero()[0]
            t_idx   = times.index(t)
            for i in range(len(indices)):
                sVec[t_idx].append(np.array(list(seqs[indices[i]]), dtype=int))
                nVec[t_idx].append(counts[indices[i]])
        return nVec, sVec
    
    if out_file==None:
        out_file = seq_file[:-4]
        
    muts   = np.load(mut_file, allow_pickle=True)
    
    df     = pd.read_csv(seq_file)
    times  = df.columns.values[1:]
    counts = list(df.drop('sequence', axis=1).sum())
    if np.sum(counts) < max_seqs:
        print2('only making a single time series')
        nVec, sVec = convert_format(df)
        f = open(out_file + '.npz', mode='wb')
        np.savez_compressed(f, mutant_sites=muts, nVec=nVec, sVec=sVec, times=np.array(times, dtype=int))
        f.close()
    else:
        # cut the time-series into shorter time series
        cum_sum     = np.cumsum(counts)
        t_idxs      = [list(times)[0]]    # the times at which to split the data
        t_numerical = np.array(list(times), dtype=int)
        for i in range(len(cum_sum)):
            denominator = max_seqs * (len(t_idxs))
            if int(cum_sum[i] / denominator) > 1:
                t_idxs.append(list(times)[i])
        t_idxs.append(str(int(list(times)[-1]) + 1))
        #print2(times)
        print2(t_idxs)
        for i in range(1, len(t_idxs)):
            kept_times   = [list(times)[j] for j in range(len(times)) if int(t_idxs[i-1])<=t_numerical[j]<=int(t_idxs[i])]
            kept_columns = ['sequence'] + kept_times
            nVec, sVec   = convert_format(df[kept_columns])
            start_time   = datetime.date(2020,1,1) + datetime.timedelta(days=int(kept_times[0]))
            start_year   = str(start_time.year)
            start_month  = str(start_time.month)
            start_day    = str(start_time.day)
            start_string = start_year + '-' + start_month + '-' + start_day
            end_time     = datetime.date(2020,1,1) + datetime.timedelta(days=int(kept_times[-1]))
            end_year     = str(end_time.year)
            end_month    = str(end_time.month)
            end_day      = str(end_time.day)
            end_string   = end_year + '-' + end_month + '-' + end_day
            temp_out = out_file + '---' + start_string + '-' + end_string + '.npz'
            f = open(temp_out, mode='wb')
            np.savez_compressed(f, nVec=nVec, sVec=sVec, mutant_sites=muts, times=np.array(kept_times, dtype=int))
            f.close()
            
            
def smoothing_to_MPL_full(smooth_dir, prob_dist, mut_file, out_file, timed=1):
    """ Combines the above 2 functions to fully decode the deconvolved data into the MPL readable format"""
    smoothing_to_MPL_alt(smooth_dir, prob_dist, mut_file, out_file)
    smoothing_to_MPL2(mut_file, out_file + '.csv', out_file=out_file)
    
    
#def smoothing_to_MPL(smooth_dir, prob_dist, nosmooth_file, out_file, timed=1):
    """ Takes the output directory containing files that have times and distributions resulting from the deconvolution method,
    and outputs a file that can be read by the inference script.
    prob_dist is a list containing the probability distribution for a single sequence.
    nosmooth file is the .npz file containing the data before smoothing."""
    
    """
    status_name = f'deconvolution-status.csv'
    status_file = open(status_name, 'w')
    status_file.close()
    
    def print2(*args):
        stat_file = open(status_name, 'a+')
        line      = [str(i) for i in args]
        string    = '\t'.join(line)
        stat_file.write(string+'\n')
        stat_file.close()
        print(string)
    
    if timed>0:
        t_start = timer()
        print2(t_start)
    
    # get the mutant sites from the non-smoothed data file
    muts        = np.load(nosmooth_file, allow_pickle=True)[0]
    print2(muts)
    
    # load the file that identifies sequences with an index
    df_idx      = pd.read_csv(os.path.join(smooth_dir, 'sequence-index.csv'))
    idxs        = list(df_idx['index'])
    seqs_mult   = list(df_idx['sequences'])
    print2('sequence index file loaded')
    
    # load the file containing sequences that only appear once
    df_ind      = pd.read_csv(os.path.join(smooth_dir, 'unique-sequences.csv'))
    if 'sequences' in df_ind.columns.values:
        seqs_ind = list(df_ind['sequences'])
    else:
        seqs_ind = list(df_ind['seqeunces'])
    times_ind   = list(df_ind['times'])
    times_ind   = [np.arange(times_ind[i] - len(prob_dist) + 1, times_ind[i] + 1) for i in range(len(times_ind))]
    
    print2('unique sequence data loaded')
    
    # load the data for the sequences that appear multiple times
    times_mult    = np.empty(len(idxs), dtype=object)
    counts_mult   = np.empty(len(idxs), dtype=object)
    nonunique_dir = os.path.join(smooth_dir, 'nonuniquewrite')
    for file in os.listdir(nonunique_dir):
        identifier  = int(file[:file.find('.csv')])
        df_temp     = pd.read_csv(os.path.join(nonunique_dir, file))
        times_temp  = list(df_temp['bp.epoch'])
        counts_temp = list(df_temp['bp.upperbound'])
        seq_idx     = idxs.index(identifier)
        times_mult[seq_idx]  = times_temp
        counts_mult[seq_idx] = counts_temp
        
    print2('nonunique sequence data loaded')
    
    # find all times and set up arrays to be filled with sequences and counts
    times_all  = np.sort(np.unique([times_ind[i][j]  for i in range(len(times_ind))  for j in range(len(times_ind[i]))] + 
                                  [times_mult[i][j] for i in range(len(times_mult)) for j in range(len(times_mult[i]))]))
    times_all  = list(times_all)
    times_full = list(np.arange(times_all[0], times_all[-1] + 1))
    
    # create the sequence and count arrays to be used for MPL
    nVec       = [[] for i in range(len(times_full))]
    sVec       = [[] for i in range(len(times_full))]
    
    # add the sequences that only appear once
    for i in range(len(times_ind)):
        time_idxs = [times_full.index(j) for j in times_ind[i]]
        for j in range(len(times_ind[i])):
            seqs = [int(k) for k in list(seqs_ind[i])]
            sVec[time_idxs[j]].append(seqs)
            nVec[time_idxs[j]].append(prob_dist[j])
            
    # add the sequences that appear multiple times
    for i in range(len(times_mult)):
        time_idxs = [times_full.index(j) for j in times_mult[i]]
        for j in range(len(times_mult[i])):
            seqs = [int(k) for k in list(seqs_mult[i])]
            sVec[time_idxs[j]].append(seqs)
            nVec[time_idxs[j]].append(counts_mult[i][j])
            
    print('data combined')
            
    # save the file
    f = open(out_file + '.npz', mode='wb')
    np.savez_compressed(f, mutant_sites=muts, nVec=nVec, sVec=sVec, times=times_full)
    
    if timed>0:
        t_end = timer()
        print('total time = ', t_end - t_start)
    """
        
### END OF DECONVOLUTION RELATED FUNCTIONS


### FUNCTIONS FOR MAKING FILES FOR WEBSITE ###
        
    
def inference_to_covar(infer_file, out_file):
    """ Takes the integrated covariance file and saves it as a csv (for website)"""
    
    data  = np.load(infer_file, allow_pickle=True)
    nucs  = data['allele_number']
    muts  = [get_label2(i) for i in data['allele_number']]
    covar = data['covar_int']
    f     = open(out_file + '.csv', mode='w')
    f.write('nucleotide_number,site_1,site_2,integrated_covariance\n')
    for i in range(len(muts)):
        for j in range(i, len(muts)):
            f.write('{},{},{},{},{}\n'.format(str(nucs[i]), muts[i], str(nucs[j]), muts[j], str(covar[i][j])))
    f.close()
    
    
def trajectory_csv(in_file, out_file):
    """ Takes an inference file and saves the trajectories for each of the mutant sites into a csv file. (for website) """
    
    data      = np.load(in_file, allow_pickle=True)
    locations = data['locations']
    traj      = data['traj']
    muts_temp = data['mutant_sites']
    times     = data['times']
    mutants   = [[get_label2(i) for i in muts_temp[j]] for j in range(len(muts_temp))]
    muts_all  = [get_label2(i)  for i in data['allele_number']]
    traj      = [np.array(traj[i],  dtype=str) for i in range(len(traj))]
    times     = [np.array(times[i], dtype=str) for i in range(len(times))]
    f         = open(out_file + '.csv', mode='w')
    f.write('variant_name,location,frequency_trajectory,times\n')
    for i in range(len(muts_all)):
        for j in range(len(mutants)):
            if muts_all[i] in list(mutants[j]):
                idx = mutants[j].index(muts_all[i])
                f.write('{},{},{},{}\n'.format(muts_all[i], locations[j], ' '.join(list(traj[j][:, idx])), ' '.join(list(times[j]))))
    f.close()
    

def make_sampling_dir(out_dir, in_dir):
    """ Make a directory containing the number of samples for different regions"""
    
    for file in os.listdir(in_dir):
        data  = np.load(os.path.join(in_dir, file), allow_pickle=True)
        nVec  = data['nVec']
        times = data['times']
        f = open(os.path.join(out_dir, file), mode='wb')
        np.savez_compressed(f, nVec=nVec, times=times)
        f.close()
    
    
def website_file2(infer_file, syn_file, link_file, out_file, bootstrap_file=None, ref_file='ref-index.csv', for_paper=False):
    """ Makes a csv file with the mutations, selection coefficients, errors, summed coefficients for linked sites,
    which protein, amino acid changes, synonymous or not, linked mutations, errors for summed coefficients,"""
    
    ref_index = pd.read_csv(ref_file)
    index     = list(ref_index['ref_index'])
    index     = [str(i) for i in index]
    ref_seq   = list(ref_index['nucleotide'])
    new_index = list(ref_index['index'])
    #new_index = np.arange(len(index))
    
    infer_data = np.load(infer_file, allow_pickle=True)
    syn_data   = np.load(syn_file,   allow_pickle=True)
    link_sites = np.load(link_file,  allow_pickle=True)
    aa_sites, aa_data = [], []
    
    syn_sites  = [i for i in syn_data['locations']]
    syn_types  = syn_data['types']
    aa_changes = syn_data['aa_changes']
    nuc_sites  = infer_data['allele_number']
    inferred   = infer_data['selection']
    errors     = infer_data['error_bars']
    infer_ind  = infer_data['selection_independent']
    mut_sites  = [get_label_new(i) for i in nuc_sites]     # labels by protein
    orf_labels = [get_label_orf_new(i) for i in nuc_sites]    # labels by reading frame
    
    # filtering out sites that aren't present in the inference from the amino acid change data and the synonymous sites data
    mask       = np.isin(syn_sites, mut_sites)
    types      = list(syn_types[mask])
    aa_changes = list(aa_changes[mask])
    syn_sites  = list(np.array(syn_sites)[mask])
    new_muts       = []
    new_types      = []
    new_aa_changes = []
    for i in range(len(mut_sites)):
        if mut_sites[i] not in syn_sites:
            if mut_sites[i][-1]=='-':
                new_types.append('S')
            else:
                new_types.append('NS')
            temp_allele = mut_sites[i].split('-')
            if '-'.join(temp_allele[1:]) in ['214m-2-G', '214n-2-A', '214o-2-G', '214s-2-G', '214t-2-A', '214u-2-A']:
                new_aa_changes.append('->E')
            elif '-'.join(temp_allele[1:]) in ['214p-2-C', '214q-2-A', '214r-2-A']:
                new_aa_changes.append('->P')
            elif '-'.join(temp_allele[1:3]) in ['214m-2', '214q-2', '214n-2', '214o-2', '214s-2', '214t-2', '214u-2', '214p-2', '214r-2']:
                new_aa_changes.append('->-')
            else:
                print(temp_allele)
        else:
            new_types.append(types[list(syn_sites).index(mut_sites[i])])
            new_aa_changes.append(aa_changes[list(syn_sites).index(mut_sites[i])])
        new_muts.append(mut_sites[i])
    types      = new_types
    syn_sites  = new_muts
    aa_changes = new_aa_changes
    
    mask2      = np.isin(mut_sites, syn_sites)
    nuc_sites  = nuc_sites[mask2]
    inferred   = inferred[mask2]
    errors     = errors[mask2]
    infer_ind  = infer_ind[mask2]
    mut_sites  = list(np.array(mut_sites)[mask2])
    orf_labels = list(np.array(orf_labels)[mask2])
    
    # finding the sum of the coefficients for the linked sites
    infer_link  = np.zeros(len(link_sites))
    error_link  = np.zeros(len(link_sites))
    link_labels = []
    counter     = np.zeros(len(link_sites))
    link_types  = []
    for i in range(len(link_sites)):
        labels_temp = []
        types_temp  = []
        for j in range(len(link_sites[i])):
            if link_sites[i][j] in mut_sites:
                infer_link[i] += inferred[mut_sites.index(link_sites[i][j])]
                error_link[i] += errors[mut_sites.index(link_sites[i][j])] ** 2
                labels_temp.append(mut_sites[mut_sites.index(link_sites[i][j])])
                types_temp.append(types[mut_sites.index(link_sites[i][j])])
                counter[i] += 1
        link_labels.append(labels_temp)
        link_types.append(types_temp)
    counter[counter==0] = 1
    error_link = np.sqrt(error_link / counter)
    
    # Finding all other sites that a given site is linked to 
    link_full  = []   # the i'th entry of this will contain all sites that mut_sites[i] is linked to
    types_full = []
    for i in range(len(mut_sites)):
        if np.any([mut_sites[i] in list(link_sites[j]) for j in range(len(link_sites))]):
            site_found = False
            for j in range(len(link_sites)):
                if mut_sites[i] in list(link_sites[j]):
                    if not site_found:
                        link_full.append([k for k in link_sites[j] if k!=mut_sites[i]])
                        if 'NS' in link_types[j]:
                            types_full.append('NS')
                        else:
                            types_full.append('S')
                    else:
                        print(f'site {mut_sites[i]} already found in a different group')
                    site_found = True
        else:
            link_full.append('None')
            types_full.append('None')

    selection_link_full = np.zeros(len(mut_sites))    # the i'th entry of this will contain the sum of the inferred coefficients for sites linked to mut_sites[i]
    error_link_full     = np.zeros(len(mut_sites))
    for i in range(len(mut_sites)):
        if np.any([mut_sites[i] in list(link_labels[j]) for j in range(len(link_labels))]):
            for j in range(len(link_labels)):
                if mut_sites[i] in list(link_labels[j]):
                    selection_link_full[i] = infer_link[j]
                    error_link_full[i]     = error_link[j]
        else:
            selection_link_full[i] = inferred[i]
            error_link_full[i]     = errors[i]
            
    proteins      = [mut_sites[i][:mut_sites[i].find('-')]       for i in range(len(mut_sites))]
    aa_number_pro = [mut_sites[i][mut_sites[i].find('-')+1:-4]   for i in range(len(mut_sites))]    # amino acid number in protein
    aa_number_rf  = [orf_labels[i][orf_labels[i].find('-')+1:-4] for i in range(len(mut_sites))]    # amino acid number in reading frame
    nucleotide    = [mut_sites[i][-1]                            for i in range(len(mut_sites))]
    
    # Finding the errors from the bootstrap stats data
    if bootstrap_file is not None:
        df = pd.read_csv(bootstrap_file)
        bs_muts = [get_label_new(i) for i in df['mutant_site']]
        bs_stddev = df['standard_deviation']
        stddev_new = []
        for i in mut_sites:
            if i in bs_muts:
                stddev_new.append(bs_stddev[bs_muts.index(i)])
            else:
                print(f'site {i} not in bootstrap data')
                stddev_new.append(0)
    else:
        stddev_new = np.zeros(len(nuc_sites))
    
    nuc_sites = np.array([str(i[:-2]) for i in nuc_sites])
    #ref_nucs  = np.array(list(ref_seq))[np.array(nuc_sites)]
    index     = list(index)
    ref_nucs  = [ref_seq[index.index(i)]   for i in nuc_sites if i in index]
    new_index = [new_index[index.index(i)] for i in nuc_sites if i in index]
    for i in nuc_sites:
        if i not in index:
            print(f'site {i} not in reference index file')
    #for i in nuc_sites:
    #    if i in list(index):
    #        ref_nucs.append(ref_seq[list(index).index(i)])
    
    # changing the amino acid change mutation in non-coding regions to be 'NA'
    for i in range(len(proteins)):
        #print(proteins[i])
        if proteins[i]=='NC':
            aa_changes[i] = 'NA'
    
    if for_paper:
        new_nucs = []
        for i in range(len(nuc_sites)):
            index, letters = separate_label_idx(nuc_sites[i])
            new_nucs.append(str(int(index) + 1) + letters)
        nuc_sites = new_nucs
        
    print('all fields calculated, putting data into dataframe...')

    # making dataframe and saving csv file        
    dic = {
        'index' : new_index,
        'nucleotide number' : nuc_sites, 
        'protein' : proteins, 
        'amino acid number in protein' : aa_number_pro,
        'amino acid number in reading frame' : aa_number_rf, 
        'synonymous' : types, 
        'amino acid mutation' : aa_changes, 
        'nucleotide' : nucleotide, 
        'reference nucleotide' : ref_nucs,
        'selection coefficient' : inferred, 
        'standard deviation' : stddev_new,
        'error' : errors, 
        'independent model coefficient' : infer_ind, 
        'linked sites' : link_full, 
        'type of linked group' : types_full,
        'total coefficient for linked group' : selection_link_full, 
        'error for linked group' : error_link_full
    }
    if bootstrap_file is None:
        del dic['standard deviation']
    for key in dic:
        print(key, len(dic[key]))

    df = pd.DataFrame.from_dict(dic)
    df.to_csv(out_file, index=False)
    
    
    
def fix_aa_changes(aa_file, link_file, out_file):
    """ For groups of linked sites, checks for codons that have multiple linked mutations in them
    and then adjusts the amino acid changes to correct for this."""
    
    ref_seq, ref_tag = get_MSA(REF_TAG + '.fasta')
    ref_seq = list(ref_seq[0])
    
    linked  = np.load(link_file, allow_pickle=True)
    aa_data = np.load(aa_file, allow_pickle=True)
    nucs    = aa_data['nuc_index']
    types   = aa_data['types']
    locs    = aa_data['locations']
    aas     = aa_data['aa_changes']
    for group in linked:
        new_group  = np.sort(group)
        new_linked = [i[:-2].split('-') for i in new_group]
        protein    = [i[0] for i in new_linked]
        codon      = [i[1] for i in new_linked]
        codon_idx  = []
        for site in new_linked:
            if len(site)<3:
                codon_idx.append('NA')
            else:
                codon_idx.append(site[2])
        nuc        = [i[-1] for i in new_group]
        L          = len(group)
        idxs_used  = []
        for i in range(L):
            if i in idxs_used or i == L-1 or protein[i]=='NC':
                continue
            if protein[i]!=protein[i+1] or codon[i]!=codon[i+1]:
                continue
            ref_idx   = int(nucs[list(locs).index(new_group[i])][:-2])
            if codon_idx[i]=='1':
                ref_idx -= 1
            ref_codon = ''.join(ref_seq[ref_idx:ref_idx+3])
            ref_aa    = codon2aa(ref_codon)
            two_sites = False
            if i == L-2: 
                two_sites = True
            elif protein[i]!=protein[i+2] or codon[i]!=codon[i+2]:
                two_sites = True
            if two_sites:
                idx1 = list(locs).index(new_group[i])
                idx2 = list(locs).index(new_group[i+1])
                print(new_group[i], new_group[i+1])
                if codon_idx[i]=='0' and codon_idx[i+1]=='1':
                    cod = nuc[i] + nuc[i+1] + ref_seq[int(nucs[idx1][:-2]) + 2]
                elif codon_idx[i]=='1' and codon_idx[i+1]=='2':
                    cod = ref_seq[int(nucs[idx1][:-2]) - 1] + nuc[i] + nuc[i+1]
                elif codon_idx[i]=='0' and codon_idx[i+1]=='2':
                    cod = nuc[i] + ref_seq[int(nucs[idx1][:-2]) + 1] + nuc[i+1]
                else:
                    print('error with codon indices')
                aa = codon2aa(cod, noq=True)
                aas[idx1] = aas[idx1][:-1] + aa
                aas[idx2] = aas[idx2][:-1] + aa
                if ref_aa != aa:
                    types[idx1] = 'NS'
                    types[idx2] = 'NS'
                else:
                    types[idx1] = 'S'
                    types[idx2] = 'S'
                idxs_used.append(i+1)
            else:
                mut_codon = nuc[i] + nuc[i+1] + nuc[i+2]
                aa   = codon2aa(mut_codon, noq=True)
                idx1 = list(locs).index(new_group[i])
                idx2 = list(locs).index(new_group[i+1])
                idx3 = list(locs).index(new_group[i+2])
                aas[idx1] = aas[idx1][:-1] + aa
                aas[idx2] = aas[idx2][:-1] + aa
                aas[idx3] = aas[idx3][:-1] + aa
                for idx in [idx1, idx2, idx3]:
                    if ref_aa!=aa:
                        types[idx] = 'NS'
                    else:
                        types[idx] = 'S'
                idxs_used.append(i+1)
                idxs_used.append(i+2)
                        
    np.savez_compressed(out_file + '.npz', types=types, aa_changes=aas, locations=locs, nuc_index=nucs)
    
    
### END OF FUNCTION FOR MAKING FILES FOR WEBSITE ###

    
def linked_analysis_alt(infer_file, link_file, syn_prot_file, variant_file, out_file, bootstrap_dir=None, ref_file='ref-index.csv', separate_labels=False, no_repeats=True, for_paper=False):
    """ Makes a csv file with the mutations, selection coefficients, errors, summed coefficients for linked sites,
    which protein, amino acid changes, synonymous or not, linked mutations, errors for summed coefficients.
    If no_repeats is false then groups of linked mutations that show up in the major variants will be included separately as well."""
    syn_data   = np.load(syn_prot_file, allow_pickle=True)
    link_sites = np.load(link_file,    allow_pickle=True)

    #ref_seq, ref_tag = get_MSA(ref_file)
    #ref_seq = ref_seq[0]
    ref_index = pd.read_csv(ref_file)
    index     = list(ref_index['ref_index'])
    index     = [str(i) for i in index]
    ref_seq   = list(ref_index['nucleotide'])
    
    #syn_sites  = [get_label(i) for i in syn_data['locations']]
    syn_types   = syn_data['types']
    syn_locs    = syn_data['nuc_index']
    aa_data     = syn_data['aa_changes']
    syn_mut_nuc = [syn_locs[i][-1] for i in range(len(syn_locs))]
    syn_number  = [syn_locs[i][:-2] for i in range(len(syn_locs))]
    syn_sites   = list(syn_data['locations'])
    syn_ref_nuc = []
    for i in range(len(syn_number)):
        if str(syn_number[i]) in index:
            syn_ref_nuc.append(ref_seq[index.index(str(syn_number[i]))])
        else:
            print(f'{syn_number[i]} not in reference sequence index file')
    #syn_ref_nuc = [ref_seq[int(i)] for i in syn_number]
    
    if infer_file[-4:]=='.npz':
        infer_data = np.load(infer_file,    allow_pickle=True)
        nuc_locs    = infer_data['allele_number']
        inferred    = infer_data['selection']
        errors      = infer_data['error_bars']
        infer_ind   = infer_data['selection_independent']
    #elif infer_file[-4:]=='.csv':
    #    inf_data = pd.read_csv(infer_file)
    mut_number  = [i[:-2] for i in nuc_locs]     # labels by protein
    mut_nucs    = [i[-1] for i in nuc_locs]
    site_labels = [get_label_new(i)[:-2] for i in nuc_locs]
    mut_sites   = [get_label_new(i)      for i in nuc_locs]
    orf_labels  = [get_label_orf_new(i)[:-2] for i in nuc_locs]    # labels by reading frame
    
    variants   = np.load(variant_file, allow_pickle=True) 
    var_nums   = np.cumsum([len(i) for i in variants])
    
    # removing groups of linked sites that contain sites that are in the defined variants
    variant_muts = [variants[i][j] for i in range(len(variants)) for j in range(len(variants[i]))]
    idxs_remove  = []
    for i in range(len(variant_muts)):
        for j in range(len(link_sites)):
            if variant_muts[i] in link_sites[j] and j not in idxs_remove:
                idxs_remove.append(j)
    if no_repeats:
        idxs_keep = [i for i in range(len(link_sites)) if i not in idxs_remove]
        link_sites = list(link_sites[np.array(idxs_keep)])  
        
    # adding the groups of mutations belonging to the variants to the linked sites
    for i in range(len(variants)):
        link_sites.append(variants[i])
    
    # filtering out sites that aren't present in the inference from the amino acid change data and the synonymous sites data
    aa_changes, types, mut_nucs, ref_nucs = [], [], [], []
    #print(mut_sites[:50], syn_sites[:50])
    for i in range(len(mut_sites)):
        if mut_sites[i] in syn_sites:
            site_idx = syn_sites.index(mut_sites[i])
            aa_changes.append(aa_data[site_idx])
            types.append(syn_types[site_idx])
            mut_nucs.append(np.array(syn_mut_nuc)[site_idx])
            ref_nucs.append(np.array(syn_ref_nuc)[site_idx])
        else:
            aa_changes.append('unknown')
            types.append('unknown')
            mut_nucs.append('unknown')
            ref_nucs.append('unknown')
            
    # Finding the variant names for the linked sites
    variant_identifiers = {'S-13-1-T' : 'epsilon', 'S-570-1-A' : 'alpha', 'S-222-1-T' : '20E_EU1', 'S-1176-0-T' : 'gamma', 
                           'S-19-1-G' : 'delta', 'S-80-1-C' : 'beta', 'NSP5-15-0-A' : 'lambda', 'S-5-0-T' : 'iota',
                           'S-371-0-C' : 'BA.1', 'S-405-0-A' : 'BA.2', 'S-1071-2-T' : 'kappa', 'S-677-2-C' : 'eta',
                           'S-346-2-C' : 'mu'}
    variant_keys  = [i for i in variant_identifiers]
    variant_names = []
    counter       = 1
    for i in range(len(link_sites)):
        if np.any(np.isin(variant_keys, link_sites[i])):
            for j in range(len(variant_keys)):
                if variant_keys[j] in list(link_sites[i]):
                    variant_names.append(variant_identifiers[variant_keys[j]])
                    continue
        elif 'S-614-1-G' in link_sites[i] and len(link_sites[i])==4:
            variant_names.append(f'B.1')
        else:
            variant_names.append(f'Group {counter}')
            counter += 1
            
    omicron_subvariants = {'S-704-1-T' : 'BA.2.12.1', 'ORF7b-11-2-T' : 'BA.4', 'M-3-0-A' : 'BA.5'}
    for i in range(len(link_sites)):
        if variant_names[i] in ['BA.1', 'BA.2']:
            for key in omicron_subvariants:
                if key in list(link_sites[i]):
                    variant_names[i] = omicron_subvariants[key]
    
    # finding the sum of the coefficients for the linked sites
    infer_link  = np.zeros(len(link_sites))
    ind_link    = np.zeros(len(link_sites))
    error_link  = np.zeros(len(link_sites))
    link_labels = []
    counter     = np.zeros(len(link_sites))
    for i in range(len(link_sites)):
        labels_temp = []
        for j in range(len(link_sites[i])):
            if link_sites[i][j] in mut_sites:
                infer_link[i] += inferred[mut_sites.index(link_sites[i][j])]
                ind_link[i]   += infer_ind[mut_sites.index(link_sites[i][j])]
                error_link[i] += errors[mut_sites.index(link_sites[i][j])] ** 2
                labels_temp.append(mut_sites[mut_sites.index(link_sites[i][j])])
                counter[i] += 1
            else:
                print(link_sites[i][j])
        link_labels.append(labels_temp)
    counter[counter==0] = 1
    error_link = np.sqrt(error_link / counter)
    
    # finding the inferred coefficients for the bootstraped sequence data, to get error bars
    if bootstrap_dir:
        inf_bs  = np.zeros((len(os.listdir(bootstrap_dir)), len(link_sites)))
        counter = 0
        for bsfile in os.listdir(bootstrap_dir):
            bs_data = np.load(os.path.join(bootstrap_dir, bsfile), allow_pickle=True)
            bs_muts = [get_label_new(i) for i in bs_data['allele_number']]
            bs_sel  = bs_data['selection']
            for j in range(len(link_sites)):
                for k in range(len(link_sites[j])):
                    if link_sites[j][k] in bs_muts:
                        inf_bs[counter, j] += bs_sel[bs_muts.index(link_sites[j][k])]
            counter += 1
        bs_stddev = np.std(inf_bs, axis=0)
    else:
        bs_stddev = np.zeros(len(infer_link))
            
    # finding amino acid changes for the linked sites
    aa_new      = []
    types_new   = []
    nucs_new    = []
    mut_nuc_new = []
    ref_nuc_new = []
    for i in range(len(link_sites)):
        aa_temp      = []
        types_temp   = []
        nucs_temp    = []
        mut_nuc_temp = []
        ref_nuc_temp = []
        for j in range(len(link_sites[i])):
            if link_sites[i][j] in mut_sites:
                site_idx = mut_sites.index(link_sites[i][j])
                nuc      = np.array(mut_number)[site_idx]
                if for_paper: nuc += 1
                nuc      = str(nuc)
                aa_temp.append(aa_changes[site_idx])
                types_temp.append(types[site_idx])
                nucs_temp.append(nuc)
                mut_nuc_temp.append(np.array(mut_nucs)[site_idx])
                ref_nuc_temp.append(np.array(ref_nucs)[site_idx])
            else:
                aa_temp.append('not_present')
                types_temp.append('not_present')
                nucs_temp.append('not_present')
                mut_nuc_temp.append('not_present')
                ref_nuc_temp.append('not_present')
        aa_new.append('/'.join(aa_temp))
        types_new.append('/'.join(types_temp))
        nucs_new.append('/'.join(nucs_temp))
        mut_nuc_new.append('/'.join(mut_nuc_temp))
        ref_nuc_new.append('/'.join(ref_nuc_temp))
    link_temp = [[link_sites[i][j][:-2] for j in range(len(link_sites[i]))] for i in range(len(link_sites))]
    link_new  = ['/'.join(i) for i in link_temp]
            
    # making dataframe and saving csv file    
    if not separate_labels:
        dic = {'nucleotides' : nucs_new, 
               'mutant_nuc' : mut_nuc_new, 
               'reference_nuc' : ref_nuc_new,
               'sites' : link_new, 
               'synonymous' : types_new, 
               'aa_mutations' : aa_new, 
               'selection_coefficient' : infer_link, 
               'selection_independent' : ind_link, 
               'standard_deviation' : bs_stddev,
               'theoretical_error' : error_link, 
               'variant_names' : variant_names}
        if not bootstrap_dir:
            del dic['standard_deviation']
    else:
        sites     = [link_new[i].split('/') for i in range(len(link_new))]
        sites     = [[sites[i][j].split('-') for j in range(len(sites[i]))] for i in range(len(sites))]
        proteins  = [[sites[i][j][0] for j in range(len(sites[i]))] for i in range(len(sites))]
        proteins  = ['/'.join(i) for i in proteins]
        codons    = [[sites[i][j][1] for j in range(len(sites[i]))] for i in range(len(sites))]
        codons    = ['/'.join(i) for i in codons]
        codon_idx = []
        for group in sites:
            idxs_temp = []
            for i in group:
                if len(i)==2:
                    idxs_temp.append('NA')
                else:
                    idxs_temp.append(i[2])
            codon_idx.append(idxs_temp)
        dic = {'nucleotides' : nucs_new, 
               'proteins' : proteins, 
               'codons' : codons, 
               'indices_in_codon' : codon_idx, 
               'mutant_nuc' : mut_nuc_new, 
               'reference_nuc' : ref_nuc_new, 
               'synonymous' : types_new, 
               'aa_mutations' : aa_new, 
               'selection_coefficient' : infer_link, 
               'selection_independent' : ind_link, 
               'theoretical_error' : error_link, 
               'variant_names' : variant_names}
    #for key in dic:
    #    print(key, len(dic[key]))
    df = pd.DataFrame.from_dict(dic)
    df = df.sort_values(by='selection_coefficient')
    df.to_csv(out_file, index=False)
    

def linked_traj_csv(link_analysis, traj_dir, out_file, delta_file=None, ba2_file=None, lambda_file=None):
    """ Takes the output of the linked_analysis function above and an inference file and finds the trajectories for the linked groups"""
    
    variant_identifiers = {'S-13-1-T' : 'epsilon', 'S-570-1-A' : 'alpha', 'S-222-1-T' : '20E_EU1', 'S-1176-0-T' : 'gamma', 
                           'S-950-0-A' : 'delta', 'S-80-1-C' : 'beta', 'NSP5-15-0-A' : 'lambda', 'S-5-0-T' : 'iota', 
                           'S-371-0-C' : 'BA.1', 'S-405-0-A' : 'BA.2', 'S-704-1-T' : 'BA.2.12.1', 
                           'ORF7b-11-2-T' : 'BA.4', 'M-3-0-A' : 'BA.5'}
    
    variant_keys  = [i for i in variant_identifiers]
    variant_names = [variant_identifiers[i] for i in variant_keys]
    var_dic       = dict(zip(variant_names, variant_keys))
    
    #inf_data  = np.load(infer_file, allow_pickle=True)
    #traj      = inf_data['traj']
    #locs      = inf_data['locations']
    #times     = inf_data['times']
    #nuc_sites = inf_data['mutant_sites']
    #mut_sites = [[get_label_new(i) for i in nuc_sites[j]] for j in range(len(nuc_sites))]
    
    link_data  = pd.read_csv(link_analysis)
    link_muts  = list(link_data['sites'])
    link_nucs  = list(link_data['mutant_nuc'])
    link_nucs  = [i.split('/') for i in link_nucs]
    link_sites = [i.split('/') for i in link_muts]
    link_sites = [[link_sites[i][j] + '-' + link_nucs[i][j] for j in range(len(link_sites[i]))] for i in range(len(link_sites))]
    var_names  = list(link_data['variant_names'])
    
    #traj_link  = []
    #traj_locs  = []
    #traj_times = []
    traj_muts  = []
    traj_files = []
    for file in os.listdir(traj_dir):
        df = pd.read_csv(os.path.join(traj_dir, file), nrows=2)
        nuc_muts = [str(i) for i in list(df.columns)[1:]]
        muts     = [get_label_new(str(i)) for i in nuc_muts]
        traj_muts.append(muts)
        traj_files.append(os.path.join(traj_dir, file))
        
    f = open(out_file + '.csv', 'w')
    f.write('sites,location,times,frequencies,variant_names\n')
    for i in range(len(var_names)):
        if var_names[i] in variant_names:
            variant_mut = var_dic[var_names[i]]
        else:
            variant_mut = link_sites[i][0]
        if var_names[i]=='delta' and delta_file is not None:
            delta_muts = link_muts[i]
            continue
        elif var_names[i]=='BA.2' and ba2_file is not None:
            ba2_muts = link_muts[i]
            continue
        elif var_names[i]=='lambda' and lambda_file is not None:
            lambda_muts = link_muts[i]
            continue
        for j in np.arange(len(traj_files)):
            muts     = traj_muts[j]
            if variant_mut not in muts:
                continue
            df = pd.read_csv(traj_files[j])
            nuc_muts = [str(k) for k in list(df.columns)[1:]]
            times = list(df['time'])
            traj  = list(df[nuc_muts[muts.index(variant_mut)]])
            loc   = file[:file.find('---')]
            times = ' '.join([str(j) for j in times])
            traj  = ' '.join([str(j) for j in traj])
            f.write(f'{link_muts[i]},{loc},{times},{traj},{var_names[i]}\n')
    if delta_file:
        delta_data = pd.read_csv(delta_file) 
        for row_idx, row in delta_data.iterrows():
            f.write(f'{delta_muts},{row.location},{row.times},{row.frequencies},delta\n')
    if ba2_file:
        ba2_data = pd.read_csv(ba2_file)
        for row_idx, row in ba2_data.iterrows():
            f.write(f'{ba2_muts},{row.location},{row.times},{row.frequencies},BA.2\n')
    if lambda_file:
        lambda_data = pd.read_csv(lambda_file)
        for row_idx, row in lambda_data.iterrows():
            f.write(f'{lambda_muts},{row.location},{row.times},{row.frequencies},lambda\n')
    f.close()


def inference_to_simulation(in_file, out_file, window=10, uk=False):
    """ Writes the output of the inference using real data to a file that can be used to run a simulation with the following characteristics:
    1. The selection coefficients for the various sites are the same as those inferred using the real data.
    2. The frequencies of each of the genomes are the same as those at the last date (or last few dates) used to perform the inference."""
    
    data = np.load(in_file, allow_pickle=True)
    allele_number = data['allele_number']
    selection     = data['selection']
    locations     = data['locations']
    mutant_sites  = data['mutant_sites']
    counts, sequences = [], []
    for sim in range(len(data['nVec'])):
        nVec      = data['nVec'][sim]
        sVec      = data['sVec'][sim]
        mutants   = data['mutant_sites'][sim]
        nVec_temp, sVec_temp = list(nVec[-1]), list(sVec[-1])
        for i in range(1, window):
            transfer_in(sVec_temp, nVec_temp, sVec[-1-i], nVec[-1-i])
        sVec  = [binary_to_labeled(seq, mutants) for seq in sVec_temp]         # Transforming the sequences to a labeled format
        nVec  = [int(10000 * i/np.sum(nVec_temp)) for i in range(len(nVec_temp))]   # Making the initial frequencies the same as those sampled but with a larger population size
        counts.append(nVec)
        sequences.append(sVec)
    f = open(out_file + '.npz', mode='wb')
    np.savez_compressed(f, counts=counts, sequences=sequences, selection=selection, locations=locations, mutant_sites=mutant_sites)
    f.close()
    if uk:
        index = 0
        for i in range(len(locations)):
            if locations[i][:6]=='united':
                break
            else:
                index += 1
        selection_uk = [selection[i] for i in range(len(selection)) if np.isin(allele_number[i], mutant_sites[index])]
        g = open(out_file + '-uk.npz', mode='wb')
        np.savez_compressed(g, counts=counts[index], sequences=sequences[index], selection=selection_uk, mutant_sites=mutant_sites[index])
        g.close()

        
def compare_trajectory_plots(sim_file, data_file, start_date, window=10, filter_tol=0.05):
    """ Calculates the trajectories from the simulation and from the real data and compares them.
    Assumes both files contain only a single location. start_date is the time when the simulation began."""
    
    def filter_frequencies(traj, muts, filter_tol=0.05):
        """ Filters out the sites whose maximum frequencies are below filter_tol"""
        new_traj, new_muts = [], []
        for i in range(len(traj[0])):
            if np.amax(traj[:,i])>=filter_tol:
                new_traj.append(traj[:, i])
                new_muts.append(muts[i])
        return np.swapaxes(new_traj, 0, 1), new_muts
    
    # load the data
    sim_data  = np.load(sim_file, allow_pickle=True)
    sim_traj  = moving_average(sim_data['traj_record'][0], window=window)
    sim_muts  = sim_data['mutant_sites'][0]
    sim_time  = [i*5 - int(window/2) + int((datetime.date.fromisoformat(start_date) - datetime.date(2020,1,1)).days) for i in np.arange(len(sim_traj))]
    real_data = np.load(data_file, allow_pickle=True)
    real_n    = real_data['nVec']
    real_s    = real_data['sVec']
    real_muts = real_data['mutant_sites']
    real_time = real_data['times']
    real_traj = trajectory_calc(real_n, real_s, real_muts)
    traj_new  = np.zeros((len(real_traj)-window+1, len(real_traj[0])))
    for i in range(len(real_traj[0])):
        traj_new[:, i] = moving_average(real_traj[:, i], window=window)
    real_traj = traj_new
    max_time  = max(sim_time[-1], real_time[-1])
    min_time  = min(sim_time[0], real_time[0])
    
    # filter out low sites at very low frequencies in both regions
    real_traj, real_muts = filter_frequencies(real_traj, real_muts, filter_tol=filter_tol)
    sim_traj,  sim_muts  = filter_frequencies(sim_traj,  sim_muts,  filter_tol=filter_tol)
    muts_both = [i for i in list(real_muts) if i in list(sim_muts)]
    labels = [get_label(i) for i in muts_both]
    
    # plot the trajectories against one another
    for s in range(int(len(muts_both)/500) + 1):
        muts_temp = muts_both[s*500:(s+1)*500]
        plt.figure(figsize=[10, 4*len(muts_temp)])
        grid = plt.GridSpec(len(muts_temp), 1)
        for i in range(len(muts_temp)):
            plt.subplot(grid[i, 0])
            plt.yticks([])
            plt.ylim(0,1)
            plt.xlim(min_time, max_time)
            plt.ylabel(labels[i+500*s])
            plt.plot(sim_time, sim_traj[:, list(sim_muts).index(muts_temp[i])], color='k', label='simulation')
            plt.plot(real_time[:-window+1], real_traj[:, list(real_muts).index(muts_temp[i])], color='r', label='real')    

