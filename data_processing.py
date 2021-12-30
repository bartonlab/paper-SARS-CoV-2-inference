# LIBRARIES

import numpy as np
import pandas as pd
import datetime
import os
from timeit import default_timer as timer   # timer for performance
from copy import deepcopy
from scipy import linalg

# GLOBAL VARIABLES

NUC = ['-', 'A', 'C', 'G', 'T']
PRO = ['-', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
       'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
REF        = NUC[0]
REF_TAG    = 'EPI_ISL_402125'
ALPHABET   = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+++++++++++++++++++++++++++'
PROTEIN_LENGTHS = {'ORF3a' : 828, 'E' : 228, 'ORF6' : 186, 'ORF7a' : 365, 'ORF7b' : 132, 'S' : 3822,
                   'N' : 1260, 'M' : 669, 'ORF8' : 336, 'ORF10' : 117, 'NSP1' : 539, 'NSP2' : 1914,
                   'NSP3' : 5834, 'NSP4' : 1500, 'NSP5' : 917, 'NSP6' : 870, 'NSP7' : 249, 'NSP8' : 594,
                   'NSP9' : 339, 'NSP10' : 417, 'NSP12' :2795, 'NSP13' : 1803, 'NSP14' : 1582, 
                   'NSP15' : 1038, 'NSP16' : 894}

# FUNCTIONS
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
    # new to account for overlap of orf7a and orf7b
    #elif (27393<=i<=27754):
    #    return i - (i - 27393)%3
    #elif (27755<=i<=27886):
    #    return i - (i - 27755)%3
    ### considered orf7a and orf7b as one reading frame (INCORRECT)
    elif (27393<=i<=27886):
        return i - (i - 27393)%3
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

    
def get_label2(i):
    return get_label(i[:-2]) + '-' + i[-1]


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


def save_MSA(msa, tag, out, fasta_width=100):
    """ Write a multiple sequence alignment with corresponding tags as a FASTA file. """
    
    f = open(out+'.fasta','w')
    
    for i in range(len(msa)):
        f.write('>'+tag[i]+'\n')
        count=0
        while count<len(msa[i]):
            for j in range(fasta_width):
                f.write(msa[i][count])
                count += 1
                if count==len(msa[i]):
                    break
            f.write('\n')
    f.close()


def freq_change_correlation(traj_file, group1, group2, region=False):
    """ Takes two groups of mutations and the csv file that contains the changes in frequency for the mutations, 
    and calculates the correlation between them.
    group1 and group2 can be names of variants. 
    FUTURE: let group1 and group2 also be lists of mutations"""
    
    # load data
    df        = pd.read_csv(traj_file)
    var_names = list(df['variant_names'])
    sites     = list(df['sites'])
    if isinstance(group1, str) and isinstance(group2, str):
        
        # find rows with the appropriate groups
        df1 = df[df['variant_names']==group1]
        df2 = df[df['variant_names']==group2]
        df1 = df1[['variant_names', 'frequencies', 'location', 'times']].sort_values(by='location')
        df2 = df2[['variant_names', 'frequencies', 'location', 'times']].sort_values(by='location')
        
        # find rows that have the same location for the two variants
        locs1     = list(df1['location'])
        locs2     = list(df2['location'])
        locs_both = list(set(locs1).intersection(set(locs2)))    # locations in which both variants appear
        if region:
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
        delta_x1  = [np.diff(i) for i in freqs1]
        delta_x2  = [np.diff(i) for i in freqs2]

        # combine into a single list and find the correlation
        total1    = [delta_x1[i][j] for i in range(len(delta_x1)) for j in range(len(delta_x1[i]))]
        total2    = [delta_x2[i][j] for i in range(len(delta_x2)) for j in range(len(delta_x2[i]))]
        return np.corrcoef(total1, total2)[0, 1]
    
    
def make_sampling_dir(out_dir, in_dir):
    """ Make a directory containing the number of samples for different regions"""
    
    for file in os.listdir(in_dir):
        data  = np.load(os.path.join(in_dir, file), allow_pickle=True)
        nVec  = data['nVec']
        times = data['times']
        f = open(os.path.join(out_dir, file), mode='wb')
        np.savez_compressed(f, nVec=nVec, times=times)
        f.close()


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
    
    
def find_nonsynonymous_all(ref_tag, data_folder, out_file, prot_out_file, status_name='nonsynonymous_status.csv', simple=True, multisite=False):
    """ Finds the sites that are nonsynonymous and synonymous given the reference sequence and the folder containing the sequences
    and the polymorphic indices.
    If simple=True, then the mutated site is placed in the reference genome and it is checked whether or not the resulting codon 
    codes for the same amino acid."""
    ### FUTURE: when there are multiple mutations in a codon, consider whether reverting the mutation to the background 
    ### that it arose on makes it nonsynonymous, instead of just comparing with the reference sequence.
    ### Could do this by looking at the sequences that have the mutation and seeing the most frequent nucleotides to
    ### appear at the other sites in the codon as well.
    
    f = open(status_name, 'w')
    f.close()
    
    def print2(*args):
        stat_file = open(status_name, 'a+')
        line   = [str(i) for i in args]
        string = '\t'.join(line)
        stat_file.write(string+'\n')
        stat_file.close()
        print(string)
    
    start_time = timer()
    ref_seq, ref_tag = get_MSA(os.path.join(REF_TAG+'.fasta'))
    ref_seq = list(ref_seq[0])
        
    # Determining whether mutations are nonsynonymous or synonymous
    types      = []    # S or NS depending on if a mutation is synonymous or nonsynonymous
    aa_changes = []
    new_sites  = []
    nuc_nums   = []
    for i in range(len(ref_seq)):
        for j in NUC:
            start_idx = get_codon_start_index(i)
            if start_idx == 0:
                types.append('S')
                aa_changes.append(f'NC-{i}')
            else:
                codon_idxs     = np.arange(start_idx,start_idx+3)
                pos            = list(codon_idxs).index(i)    # the position of the mutation in the codon
                ref_codon      = ref_seq[start_idx:start_idx+3]
                ref_aa         = codon2aa(ref_codon)
                if j == '-' and ref_seq[i] != '-':
                    types.append('NS')
                    aa_changes.append(f'{ref_aa}>-')
                else:
                    mut_codon      = deepcopy(ref_seq[start_idx:start_idx+3])
                    mut_codon[pos] = j
                    mut_aa         = codon2aa(mut_codon)
                    if ref_aa == mut_aa:
                        types.append('S')
                    else:
                        types.append('NS')
                    aa_changes.append(ref_aa + '>' + mut_aa)
            new_sites.append(get_label(i) + f'-{j}')
            nuc_nums.append(str(i) + f'-{j}')
                    
    
    f = open(out_file + '.npz', 'wb')
    np.savez_compressed(f, types=types, locations=new_sites)
    f.close()
    
    g = open(prot_out_file, 'w')
    for i in range(len(new_sites)):
        g.write('%s,%s\n' % (new_sites[i], aa_changes[i]))
    g.close()
    
    h = open(out_file + '-prot.npz', 'wb')
    np.savez_compressed(h, types=types, locations=new_sites, aa_changes=aa_changes, nuc_index=nuc_nums)
    h.close()    


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


def count_total_sequences(region_file):
    """ Finds the total number of sequences in a region given the .npz files. """
    nVec = np.load(region_file, allow_pickle=True)['nVec']
    return np.sum(np.array([np.sum(i) for i in nVec]))
    
    
def save_traj_selection(tv_inf, group=None, traj_site=None, out_file=None):
    """Given a file containing selection coefficients and trajectories over time, save them in a csv.
    Assumes that the trajectories do not overlap. Fix this."""
    data       = np.load(tv_inf, allow_pickle=True)
    times      = data['times']
    inf_times  = data['times_inf']
    selection  = data['selection']
    alleles    = data['allele_number']
    mutants    = data['mutant_sites']
    traj       = data['traj']
    labels     = np.array([get_label2(i) for i in alleles])
    
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
        new_times.append(times[i][mask])
        new_traj.append(traj[i][mask])
    times      = new_times
    traj       = new_traj
    
    t_present  = list(np.unique([times[i][j] for i in range(len(times)) for j in range(len(times[i]))]))
    t_present.pop(t_present.index(363))
    t_present  = np.array(t_present)
    #print(t_present)
    t_mask     = np.isin(inf_times, t_present)
    s_group    = s_group[t_mask]
    
    traj_full  = np.zeros(len(s_group))
    pos_full   = [np.searchsorted(t_present, i) for i in times]
    for i in range(len(traj)):
        t_temp    = times[i]
        traj_temp = traj[i]
        muts_temp = [get_label2(j) for j in mutants[i]]
        pos_temp  = pos_full[i]
        idx       = muts_temp.index(traj_site)
        site_traj = traj_temp[:, idx]
        for j in range(len(t_temp)):
            traj_full[pos_temp[j]] = site_traj[j]
    
    f = open(out_file + '.csv', mode='w')
    f.write('time,frequency,selection\n')
    for i in range(len(t_present)):
        f.write('%d,%.8f,%.8f\n' % (t_present[i], traj_full[i], s_group[i]))
    f.close()
    
    
def website_file2(infer_file, syn_file, link_file, out_file, for_paper=False):
    """ Makes a csv file with the mutations, selection coefficients, errors, summed coefficients for linked sites,
    which protein, amino acid changes, synonymous or not, linked mutations, errors for summed coefficients,"""
    
    ref_seq, ref_tag = get_MSA(REF_TAG + '.fasta')
    ref_seq = ref_seq[0]
    
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
    mut_sites  = [get_label(i[:-2]) + '-' + i[-1] for i in nuc_sites]     # labels by protein
    orf_labels = [get_label_orf(i[:-2]) + '-' + i[-1] for i in nuc_sites]    # labels by reading frame
    
    # filtering out sites that aren't present in the inference from the amino acid change data and the synonymous sites data
    mask       = np.isin(syn_sites, mut_sites)
    types      = list(syn_types[mask])
    aa_changes = list(aa_changes[mask])
    syn_sites  = list(np.array(syn_sites)[mask])
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
    
    nuc_sites     = np.array([int(i[:-2]) for i in nuc_sites])
    ref_nucs      = np.array(list(ref_seq))[np.array(nuc_sites)]
    
    # changing the amino acid change mutation in non-coding regions to be 'NA'
    for i in range(len(proteins)):
        #print(proteins[i])
        if proteins[i]=='NC':
            aa_changes[i] = 'NA'
    
    if for_paper:
        nuc_sites += 1

    # making dataframe and saving csv file        
    dic = {'nucleotide number' : nuc_sites, 'protein' : proteins, 'amino acid number in protein' : aa_number_pro,
           'amino acid number in reading frame' : aa_number_rf, 'synonymous' : types, 
           'amino acid mutation' : aa_changes, 'nucleotide' : nucleotide, 'reference nucleotide' : ref_nucs,
           'selection coefficient' : inferred, 
           'error' : errors, 'independent model coefficient' : infer_ind, 'linked sites' : link_full, 
           'type of linked group' : types_full,
           'total coefficient for linked group' : selection_link_full, 'error for linked group' : error_link_full}

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
    
    
def linked_analysis_alt(infer_file, link_file, syn_prot_file, variant_file, out_file, ref_file=f'{REF_TAG}.fasta', separate_labels=False, no_repeats=True, for_paper=False):
    """ Makes a csv file with the mutations, selection coefficients, errors, summed coefficients for linked sites,
    which protein, amino acid changes, synonymous or not, linked mutations, errors for summed coefficients.
    If no_repeats is false then groups of linked mutations that show up in the major variants will be included separately as well."""
    
    infer_data = np.load(infer_file,    allow_pickle=True)
    syn_data   = np.load(syn_prot_file, allow_pickle=True)
    link_sites = np.load(link_file,    allow_pickle=True)

    ref_seq, ref_tag = get_MSA(ref_file)
    ref_seq = ref_seq[0]
    
    #syn_sites  = [get_label(i) for i in syn_data['locations']]
    syn_types   = syn_data['types']
    syn_locs    = syn_data['nuc_index']
    aa_data     = syn_data['aa_changes']
    syn_mut_nuc = [syn_locs[i][-1] for i in range(len(syn_locs))]
    syn_number  = [syn_locs[i][:-2] for i in range(len(syn_locs))]
    syn_sites   = list(syn_data['locations'])
    syn_ref_nuc = [ref_seq[int(i)] for i in syn_number]
    
    nuc_locs    = infer_data['allele_number']
    inferred    = infer_data['selection']
    errors      = infer_data['error_bars']
    infer_ind   = infer_data['selection_independent']
    mut_number  = [int(i[:-2]) for i in nuc_locs]     # labels by protein
    mut_nucs    = [i[-1] for i in nuc_locs]
    site_labels = [get_label(i) for i in mut_number]
    mut_sites   = [site_labels[i] + '-' + mut_nucs[i] for i in range(len(mut_nucs))]
    orf_labels  = [get_label_orf(i) for i in mut_number]    # labels by reading frame
    
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
                           'S-19-1-G' : 'delta', 'S-80-1-C' : 'beta', 'S-76-1-T' : 'lambda', 'S-5-0-T' : 'iota'}
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
        link_labels.append(labels_temp)
    counter[counter==0] = 1
    error_link  = np.sqrt(error_link / counter)
            
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
        dic = {'nucleotides' : nucs_new, 'mutant_nuc' : mut_nuc_new, 'reference_nuc' : ref_nuc_new,
               'sites' : link_new, 'synonymous' : types_new, 'aa_mutations' : aa_new, 
               'selection_coefficient' : infer_link, 'selection_independent' : ind_link, 
               'error' : error_link, 'variant_names' : variant_names}
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
        dic = {'nucleotides' : nucs_new, 'proteins' : proteins, 'codons' : codons, 'indices_in_codon' : codon_idx, 
               'mutant_nuc' : mut_nuc_new, 'reference_nuc' : ref_nuc_new, 'synonymous' : types_new, 
               'aa_mutations' : aa_new, 'selection_coefficient' : infer_link, 'selection_independent' : ind_link, 
               'error' : error_link, 'variant_names' : variant_names}
    for key in dic:
        print(key, len(dic[key]))
    df = pd.DataFrame.from_dict(dic)
    df = df.sort_values(by='selection_coefficient')
    df.to_csv(out_file, index=False)
    

def linked_traj_csv(link_analysis, infer_file, out_file):
    """ Takes the output of the linked_analysis function above and an inference file and finds the trajectories for the linked groups"""
    
    variant_identifiers = {'S-13-1-T' : 'epsilon', 'S-570-1-A' : 'alpha', 'S-222-1-T' : '20E_EU1', 'S-1176-0-T' : 'gamma', 
                           'S-19-1-G' : 'delta', 'S-80-1-C' : 'beta', 'S-76-1-T' : 'lambda', 'S-5-0-T' : 'iota'}
    variant_keys  = [i for i in variant_identifiers]
    variant_names = [variant_identifiers[i] for i in variant_keys]
    var_dic       = dict(zip(variant_names, variant_keys))
    
    inf_data  = np.load(infer_file, allow_pickle=True)
    traj      = inf_data['traj']
    locs      = inf_data['locations']
    times     = inf_data['times']
    nuc_sites = inf_data['mutant_sites']
    mut_sites = [[get_label(int(i[:-2])) + '-' + i[-1] for i in nuc_sites[j]] for j in range(len(nuc_sites))]
    
    link_data  = pd.read_csv(link_analysis)
    link_muts  = list(link_data['sites'])
    link_nucs  = list(link_data['mutant_nuc'])
    link_nucs  = [i.split('/') for i in link_nucs]
    link_sites = [i.split('/') for i in link_muts]
    link_sites = [[link_sites[i][j] + '-' + link_nucs[i][j] for j in range(len(link_sites[i]))] for i in range(len(link_sites))]
    var_names  = list(link_data['variant_names'])
    
    traj_link  = []
    traj_locs  = []
    traj_times = []
    f = open(out_file + '.csv', 'w')
    f.write('sites,location,times,frequencies,variant_names\n')
    for i in range(len(var_names)):
        if var_names[i] in variant_names:
            variant_mut = var_dic[var_names[i]]
        else:
            variant_mut = link_sites[i][0]
        for j in range(len(mut_sites)):
            if variant_mut in list(mut_sites[j]):
                traj_temp    = np.array(traj[j][:, list(mut_sites[j]).index(variant_mut)], dtype=str)
                traj_temp    = ' '.join(traj_temp)
                times_temp   = ' '.join(np.array(times[j], dtype=str))
                f.write(f'{link_muts[i]},{locs[j]},{times_temp},{traj_temp},{var_names[i]}\n')
    f.close()


