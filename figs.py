"""
A preamble here.
"""

#############  PACKAGES  #############

import sys, os
from copy import deepcopy

import numpy as np

import scipy as sp
import scipy.stats as st

import pandas as pd

import datetime as dt

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

import seaborn as sns

from colorsys import hls_to_rgb

import mplot as mp


############# PARAMETERS #############

# GLOBAL VARIABLES

NUC = ['A', 'C', 'G', 'T', '-']
REF = NUC[-1]
EXT = '.pdf'

# SARS-CoV-2 information

PROTEINS = ['NSP1', 'NSP2', 'NSP3', 'NSP4', 'NSP5', 'NSP6', 'NSP7', 'NSP8','NSP9', 'NSP10', 'NSP12', 'NSP13', 'NSP14',
            'NSP15', 'NSP16', 'S', 'ORF3a', 'E', 'M', 'ORF6', 'ORF7a', 'ORF7b', 'ORF8', 'N', 'ORF10']

PROTEIN_LENGTHS = {'ORF3a' : 828, 'E' : 228, 'ORF6' : 186, 'ORF7a' : 365, 'ORF7b' : 132, 'S' : 3822,
                   'N' : 1260, 'M' : 669, 'ORF8' : 336, 'ORF10' : 117, 'NSP1' : 539, 'NSP2' : 1914,
                   'NSP3' : 5834, 'NSP4' : 1500, 'NSP5' : 917, 'NSP6' : 870, 'NSP7' : 249, 'NSP8' : 594,
                   'NSP9' : 339, 'NSP10' : 417, 'NSP12' :2795, 'NSP13' : 1803, 'NSP14' : 1582,
                   'NSP15' : 1038, 'NSP16' : 894}

## Code Ocean directories
#MLB_DIR = '../data/HIV'
#WFS_DIR = '../data/wfsim'
#HIV_MPL_DIR = '../data/HIV/MPL'
#SIM_MPL_DIR = '../data/simulation/MPL'
#HIV_DIR = '../data/HIV'
#SIM_DIR = '../data/simulation'
#FIG_DIR = '../results'

# GitHub directories
SIM_DIR = 'sims'
FIG_DIR = 'figures'

# Standard color scheme

BKCOLOR    = '#252525'
LCOLOR     = '#969696'
C_BEN      = '#EB4025' #'#F16913'
C_BEN_LT   = '#F08F78' #'#fdd0a2'
C_NEU      =  LCOLOR   #'#E8E8E8' # LCOLOR
C_NEU_LT   = '#E8E8E8' #'#F0F0F0' #'#d9d9d9'
C_DEL      = '#3E8DCF' #'#604A7B'
C_DEL_LT   = '#78B4E7' #'#dadaeb'
C_MPL      = '#FFB511'
C_SL       = C_NEU
PALETTE    = sns.hls_palette(2)  # palette1
MAIN_COLOR = PALETTE[0]
COMP_COLOR = PALETTE[1]
COLOR_1    = '#1295D8'
COLOR_2    = '#FFB511'
VAR_NAMES  = ['Alpha', 'Delta', 'Omicron (BA.1)', 'BA.2', 'BA.4', 'BA.5', 'JN.1', 'BA.2.86', 'XBB', 'XBB.1.5', 'EG.5', 'Gamma', 'Beta', 'Lambda', 'Epsilon', 'Mu']
VAR_COLORS = sns.husl_palette(len(VAR_NAMES)+2, h=0.15)
# VAR_COLORS = sns.color_palette('rainbow', len(VAR_NAMES)+2)
VAR_N2C    = {'Epsilon': VAR_COLORS[0], 'Alpha': VAR_COLORS[1], 'Beta': VAR_COLORS[2], 'Lambda': VAR_COLORS[3],
              'Gamma': VAR_COLORS[4], 'Mu': VAR_COLORS[5], 'Delta': VAR_COLORS[6], 'Omicron (BA.1)': VAR_COLORS[7], 
              'BA.1': VAR_COLORS[7], 'BA.2': VAR_COLORS[8], 'BA.4': VAR_COLORS[9], 'BA.5': VAR_COLORS[10], 'XBB': VAR_COLORS[11],
              'XBB.1.5': VAR_COLORS[12], 'EG.5': VAR_COLORS[13], 'EG.5.1': VAR_COLORS[13], 'BA.2.86': VAR_COLORS[14], 'JN.1': VAR_COLORS[15]}
C_BA4      = VAR_N2C['BA.4']
C_BA5      = VAR_N2C['BA.5']
C_BA2      = VAR_N2C['BA.2']
C_OMICRON  = VAR_N2C['Omicron (BA.1)']
C_DELTA    = VAR_N2C['Delta']
C_ALPHA    = VAR_N2C['Alpha']
C_A222V    = VAR_COLORS[-1]

#VAR_COLORS = sns.husl_palette(12)
#C_DELTA    = VAR_COLORS[0]
#C_GAMMA    = VAR_COLORS[1]
#C_ALPHA    = VAR_COLORS[2]
#C_BETA     = VAR_COLORS[3]
#C_LAMBDA   = VAR_COLORS[4]
#C_EPSILON  = VAR_COLORS[5]
#C_D614G    = VAR_COLORS[6]
#C_A222V    = VAR_COLORS[7]

def S2COLOR(s):
    """ Convert selection coefficient to color. """
    if s>0:    return C_BEN
    elif s==0: return C_NEU
    else:      return C_DEL

# Plot conventions

cm2inch = lambda x: x/2.54
SINGLE_COLUMN   = cm2inch(8.8)
ONE_FIVE_COLUMN = cm2inch(11.4)
DOUBLE_COLUMN   = cm2inch(18.0)
SLIDE_WIDTH     = 10.5

GOLDR      = (1.0 + np.sqrt(5)) / 2.0
TICKLENGTH = 3
TICKPAD    = 3

# # paper style
# FONTFAMILY    = 'Arial'
# SIZESUBLABEL  = 8
# SIZELABEL     = 6
# SIZETICK      = 6
# SMALLSIZEDOT  = 6.
# SIZELINE      = 0.6
# AXES_FONTSIZE = 6
# AXWIDTH       = 0.4

# # grant style
# FONTFAMILY   = 'Arial'
# SIZESUBLABEL = 12
# SIZELABEL    = 8
# SIZETICK     = 8
# SMALLSIZEDOT = 6. * 1.3
# SIZELINE     = 0.6
# AXWIDTH      = 0.4

# slides style
FONTFAMILY   = 'Avenir'
SIZESUBLABEL = 18 #24 #14
SIZELABEL    = 18 #14
SIZETICK     = 18 #14
SMALLSIZEDOT = 8. * 7
SIZELINE     = 1.5
AXWIDTH      = 1.0

# circle plot formatting
S_MULT   = 40*SMALLSIZEDOT #40*SMALLSIZEDOT
S_CUTOFF = 0.01
DS_NORM  = 0.008
#S_CUTOFF = 0.01
#DS_NORM  = 0.006


FIGPROPS = {
    'transparent' : True,
    #'bbox_inches' : 'tight'
}

DEF_BARPROPS = {
    'lw'          : SIZELINE/2,
    'width'       : 0.25,
    'edgecolor'   : BKCOLOR,
    'align'       : 'center', #other option: edge
    'orientation' : 'vertical'
}

DEF_HISTPROPS = {
    'histtype'    : 'bar',
    'lw'          : SIZELINE/2,
    'rwidth'      : 0.8,
    'ls'          : 'solid',
    'edgecolor'   : BKCOLOR,
    'alpha'       : 0.5
}

DEF_ERRORPROPS = {
    'mew'        : AXWIDTH,
    'markersize' : SMALLSIZEDOT/2,
    'fmt'        : 'o',
    'elinewidth' : SIZELINE/2,
    'capthick'   : 0,
    'capsize'    : 0
}

DEF_LINEPROPS = {
    'lw' : SIZELINE,
    'ls' : '-'
}

DEF_LABELPROPS = {
    'family' : FONTFAMILY,
    'size'   : SIZELABEL,
    'color'  : BKCOLOR,
    'clip_on': False
}

DEF_SUBLABELPROPS = {
    'family'  : FONTFAMILY,
    'size'    : SIZESUBLABEL+1,
    'weight'  : 'bold',
    'ha'      : 'center',
    'va'      : 'center',
    'color'   : 'k',
    'clip_on' : False
}

DEF_TICKLABELPROPS = {
    'family' : FONTFAMILY,
    'size'   : SIZETICK,
    'color'  : BKCOLOR
}

DEF_TICKPROPS = {
    'length'    : TICKLENGTH,
    'width'     : AXWIDTH/2,
    'pad'       : TICKPAD,
    'axis'      : 'both',
    'direction' : 'out',
    'colors'    : BKCOLOR,
    'bottom'    : True,
    'left'      : True,
    'top'       : False,
    'right'     : False
}

DEF_MINORTICKPROPS = {
    'length'    : TICKLENGTH-1.25,
    'width'     : AXWIDTH/2,
    'axis'      : 'both',
    'direction' : 'out',
    'which'     : 'minor',
    'color'     : BKCOLOR
}

DEF_AXPROPS = {
    'linewidth' : AXWIDTH,
    'linestyle' : '-',
    'color'     : BKCOLOR
}

PARAMS = {'text.usetex': False, 'mathtext.fontset': 'stixsans', 'mathtext.default' : 'regular'}
plt.rcParams.update(PARAMS)

############# HELPER    FUNCTIONS #############

def GET_CODON_START_INDEX(i):
    """ Given a sequence index i, determine the index of the first nucleotide in the codon. """
    if   (13467<=i<=21554):
        return i - (i - 13467)%3
    elif (25392<=i<=26219):
        return i - (i - 25392)%3
    elif (26244<=i<=26471):
        return i - (i - 26244)%3
    elif (27201<=i<=27386):
        return i - (i - 27201)%3
    elif (27393<=i<=27886):
        return i - (i - 27393)%3
    elif (  265<=i<=13482):
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

def GET_LABEL(i):
    """ For a SARS-CoV-2 reference sequence index i, return the label in the form
    'coding region - protein number-nucleotide in codon number'.
    For example, 'ORF1b-204-1'.
    Should check to make sure NSP12 labels are correct due to the frame shift."""
    i = int(i)
    frame_shift = str(i - GET_CODON_START_INDEX(i))
    if   (25392<=i<26220):
        return "ORF3a-" + str(int((i - 25392) / 3) + 1)  + '-' + frame_shift
    elif (26244<=i<26472):
        return "E-"     + str(int((i - 26244) / 3) + 1)  + '-' + frame_shift
    elif (27201<=i<27387):
        return "ORF6-"  + str(int((i - 27201) / 3) + 1)  + '-' + frame_shift
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
        
def NUMBER_NONSYNONYMOUS(linked, site_types, labels, mutant_types):
    """ Given a group of mutations, find the percentage that are nonsynonymous.
    Mutant_types are the types (synonymous or nonsynonymous) for the group of mutations.
    site_types are the types for the linked sites."""
    
    # Filter out linked groups if all mutations are synonymous or nonsynonymous.
    linked_new = []
    for i in range(len(site_types)):
        if 0 < len(np.array(site_types[i])[np.array(site_types[i])=='NS']) and len(np.array(site_types[i])[np.array(site_types[i])=='NS']) < len(site_types[i]):
            linked_new.append(linked[i])
    linked_all = [linked_new[i][j] for i in range(len(linked_new)) for j in range(len(linked_new[i]))]
    
    # find percentage nonsynonymous
    syn        = np.array(labels)[np.array(mutant_types)=='S']
    num_linked = len([site for site in syn if site in linked_all])
    if len(labels)==0:
        return -1
    else:
        return (len(labels) - len(syn) + num_linked) / len(labels)
        
def NUMBER_NONSYNONYMOUS_NONLINKED(inferred, num_large, linked, site_types, labels_all, mutant_types):
    """ Find out the percentage of the num_large largest selection coefficients that are nonsynonymous
    and are not linked to some nonsynonymous mutation. """
        
    # Filter out linked groups if all mutations are synonymous or nonsynonymous.
    linked_new = []
    for i in range(len(site_types)):
        if 0 < len(np.array(site_types[i])[np.array(site_types[i])=='NS']) and len(np.array(site_types[i])[np.array(site_types[i])=='NS']) < len(site_types[i]):
            linked_new.append(linked[i])
    linked_all = [linked_new[i][j] for i in range(len(linked_new)) for j in range(len(linked_new[i]))]
        
    # determine how many coefficients of the largest coefficients are nonsynonymous and not linked to a nonsynonymous mutation.
    #large_inferred = np.sort(inferred)[-num_large:]
    #indices = np.argsort(inferred)[-num_large:]
    indices = np.argsort(np.absolute(inferred))[-num_large:]
    large_inferred = inferred[indices]
    large_mut_types, large_labels = [], []
    for i in range(num_large):
        large_mut_types.append(mutant_types[indices[i]])
        large_labels.append(labels_all[indices[i]])
    large_mut_types = np.array(large_mut_types)
    #nonsyn = np.array(large_labels)[np.array(large_mut_types)=='NS']
    syn = np.array(large_labels)[np.array(large_mut_types)=='S']
    num_linked = len([site for site in syn if site in linked_all])
    return (num_large - len(syn) + num_linked) / num_large
    
def CALCULATE_LINKED_COEFFICIENTS(inf_file, link_file, tv=False):
    """ Finds the coefficients and the labels for the linked sites"""
    
    # loading and processing the data
    data            = np.load(inf_file, allow_pickle=True)
    linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in every region
    inferred        = data['selection']
    if len(np.shape(inferred))==1:
        error       = data['error_bars']
    else:
        error       = np.zeros(np.shape(inferred))
    allele_number   = data['allele_number']
    labels          = [GET_LABEL(i) for i in allele_number]
    linked_all      = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
        
    # finding total coefficients for linked sites and adding them together.
    #if len(np.shape(inferred))==1:
    if not tv:
        inferred_link   = np.zeros(len(linked_sites))
        error_link      = np.zeros(len(linked_sites))
        labels_link     = []
        for i in range(len(linked_sites)):
            labels_temp = []
            for j in range(len(linked_sites[i])):
                if np.any(linked_sites[i][j]==np.array(labels)):
                    inferred_link[i] += inferred[np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
                    error_link[i]    += error[np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
                    labels_temp.append(linked_sites[i][j])
            labels_link.append(labels_temp)
    else:
        inferred_link = np.zeros((len(inferred), len(linked_sites)))
        error_link    = np.zeros((len(inferred), len(linked_sites)))
        labels_link   = []
        for i in range(len(linked_sites)):
            labels_temp = []
            for j in range(len(linked_sites[i])):
                if np.any(linked_sites[i][j]==np.array(labels)):
                    inferred_link[:, i] += inferred[:, np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
                    error_link[:, i]    += error[:, np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
                    labels_temp.append(linked_sites[i][j])
            labels_link.append(labels_temp)
        
    return labels_link, inferred_link, error_link
    
def FIND_NULL_DISTRIBUTION(file, link_file, neutral_tol=0.01):
    """ Given the inferred coefficients over all time, find those that are ultimately inferred to be nearly neutral,
    and then collect the inferred coefficients for these sites at every time."""
    
    # loading and processing the data
    data            = np.load(file, allow_pickle=True)
    linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in every region
    allele_number   = data['allele_number']
    labels          = [GET_LABEL(i) for i in allele_number]
    linked_all      = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    inferred        = data['selection']
    traj            = data['traj']
    times           = data['times']
    mutant_sites    = data['mutant_sites']
    digits          = len(str(len(inferred)*len(inferred[0])))
        
    # finding total coefficients for linked sites and adding them together.
    inferred_link   = np.zeros((len(inferred), len(linked_sites)))
    inferred_nolink = []
    labels_nolink   = []
    for i in range(len(linked_sites)):
        for j in range(len(linked_sites[i])):
            if np.any(linked_sites[i][j]==np.array(labels)):
                inferred_link[:,i] += inferred[:, np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
    counter = 0
    for i in range(len(labels)):
        if labels[i] not in linked_all:
            inferred_nolink.append(inferred[:,i])
            labels_nolink.append(labels[i])
            counter+=1
    nolink_new = np.zeros((len(inferred_nolink[0]), len(inferred_nolink)))
    for i in range(len(inferred_nolink)):
        nolink_new[:,i] = inferred_nolink[i]
    inferred_nolink = nolink_new
    
    # determing sites that are ultimately inferred to be nearly neutral
    inf_new       = np.concatenate((inferred_link, np.array(inferred_nolink)), axis=1)
    neutral_mask  = np.absolute(inf_new[-1])<neutral_tol
    L             = len([i for i in neutral_mask if i])
    inferred_neut = np.zeros((len(inf_new), L))
    for i in range(len(inferred)):
        inferred_neut[i] = inf_new[i][neutral_mask]
        
    # finding the initial times at which each variant arose
    ### If the inferred coefficient is actually zero (not approximately zero) this is almsot definitely because the mutation has not showed up anywhere
    inferred_neut = [inferred_neut[i][j] for i in range(len(inferred_neut)) for j in range(len(inferred_neut[i]))]
    print(len(inferred_neut))
    print(len(np.array(inferred_neut)[np.array(inferred_neut)!=0]))
    inferred_neut = np.array(inferred_neut)[np.array(inferred_neut)!=0]
    return inferred_neut


############# PLOTTING  FUNCTIONS #############

def plot_performance(sim_file, inf_file, replicate_file, pop_size, cutoff=0.01):

    # Load simulation data
    sim_data         = np.load(sim_file, allow_pickle=True)
    simulations      = sim_data['simulations']
    actual           = sim_data['selection_all']
    pop_size         = sim_data['pop_size']
    traj             = sim_data['traj_record']
    mutant_sites     = sim_data['mutant_sites']
    mutant_sites_all = sim_data['mutant_sites_all']

    # Load inference data
    inf_data     = np.load(inf_file, allow_pickle=True)
    inferred     = inf_data['selection']
    error        = inf_data['error_bars']
    inferred_ind = inf_data['selection_ind']
    error_ind    = inf_data['errors_ind']

    # Other parameters
    record = np.ones(simulations)
    pop_size_assumed = pop_size
    
    selection = {}
    for i in range(len(mutant_sites_all)):
        selection[mutant_sites_all[i]] = actual[i]

    # Functions 1
    def trajectory_reshape(traj):
        # reshape trajectories
        traj_reshaped = []
        for i in range(len(traj)):
            T_temp = len(traj[i])
            L_temp = len(traj[i][0])
            traj_temp = np.zeros((T_temp, L_temp))
            for j in range(T_temp):
                traj_temp[j] = traj[i][j]
            traj_reshaped.append(np.array(traj_temp))
        return traj_reshaped
    
    def find_significant_sites(traj, mutant_sites, cutoff):
        # find sites whose frequency is (not?) too small
        allele_number = []
        for i in range(len(traj)):
            for j in range(len(traj[i][0])):
                if np.amax(traj[i][:, j]) > cutoff:
                    allele_number.append(mutant_sites[i][j])
        allele_number = np.sort(np.unique(np.array(allele_number)))
        return allele_number
        
    # reorganize arrays and filter low frequency trajectories
    traj = trajectory_reshape(traj)
    
    allele_number = find_significant_sites(traj, mutant_sites, cutoff)
    actual = np.array([selection[i] for i in allele_number])
    L_traj = len(allele_number)
    if len(actual)!=0:
        s_value = np.amax(np.array(actual))
    else:
        s_value = 0
    
    # Functions 2
    def filter_sites(traj, error, inferred, mutant_sites, mutant_sites_all, allele_number):
        # eliminate sites whose frequency is too small
        traj_new = [[] for i in range(simulations)]
        mutant_sites_new = [[] for i in range(simulations)]
        inferred_new = np.zeros(len(allele_number))
        error_bars = np.zeros(len(allele_number))
        for i in range(len(traj)):
            for j in range(len(traj[i][0])):
                if mutant_sites[i][j] in allele_number:
                    traj_new[i].append(np.array(traj[i][:,j]))
                    mutant_sites_new[i].append(mutant_sites[i][j])
        for i in range(len(allele_number)):
            loc = np.where(allele_number[i] == mutant_sites_all)
            inferred_new[i] = inferred[loc]
            error_bars[i] = error[loc]
        return traj_new, inferred_new, error_bars, mutant_sites_new

    def filter_sites_ind(inferred, error):
        # eliminates sites whose frequency is too small in the individually inferred coefficients
        inferred_new = np.zeros((simulations, L_traj))
        error_new    = np.zeros((simulations, L_traj))
        for sim in range(simulations):
            for i in range(L_traj):
                loc = np.where(mutant_sites_all == allele_number[i])
                inferred_new[sim, i] = inferred[sim][loc]
                error_new[sim, i]    = error[sim][loc]
        return inferred_new, error_new
        
    def filter_sites_infer(traj, inferred, mutant_sites, mutant_sites_all, allele_number):
        # eliminate sites whose frequency is too small
        inferred_new = np.zeros(len(allele_number))
        mutant_sites_new = [[] for i in range(len(traj))]
        for i in range(len(traj)):
            for j in range(len(traj[i][0])):
                if mutant_sites[i][j] in allele_number:
                    mutant_sites_new[i].append(mutant_sites[i][j])
        for i in range(len(allele_number)):
            loc = np.where(allele_number[i] == mutant_sites_all)
            inferred_new[i] = inferred[loc]
        return inferred_new
        
    def separate_sign(selection):
        # finds the indices at which mutations are beneficial, neutral, and deleterious
        splus, sminus, szero = {}, {}, {}
        for i in selection:
            if selection[i] > 0:
                splus[i] = selection[i]
            elif selection[i] < 0:
                sminus[i] = selection[i]
            else:
                szero[i] = selection[i]
        mutant_sites = [i for i in selection]
        ind_plus, ind_minus, ind_zero = [], [], []
        for i in splus:
            ind_plus.append(np.where(i == mutant_sites)[0][0])
        for i in sminus:
            ind_minus.append(np.where(i == mutant_sites)[0][0])
        for i in szero:
            ind_zero.append(np.where(i == mutant_sites)[0][0])
        return ind_minus, ind_zero, ind_plus
        
    def sort_by_sign(selection, inferred):
        # sorts the beneficial, deleterious, and neutral mutations
        ind_minus, ind_zero, ind_plus = separate_sign(selection)
        inf_plus  = [inferred[i] for i in ind_plus]
        inf_minus = [inferred[i] for i in ind_minus]
        inf_zero  = [inferred[i] for i in ind_zero]
        return inf_minus, inf_zero, inf_plus
    
    traj, inferred, error, mutant_sites = filter_sites(traj, error, inferred, mutant_sites, mutant_sites_all, allele_number)
    mutant_sites_all = np.sort(np.unique([mutant_sites[j][i] for j in range(len(mutant_sites)) for i in range(len(mutant_sites[j]))]))
    
    # plot combining the frequencies, the selection coefficients, and the histograms
    indices       = np.argsort(actual)[::-1]
    actual        = np.array(actual)[indices]
    inferred      = inferred[indices]
    colors_inf    = np.array([S2COLOR(i) for i in actual], dtype=str)
    allele_number = allele_number[indices]
    
    # Finding the different sets of actual coefficients that are all the same
    actual_unique  = np.unique(actual)
    unique_lengths = [len(np.isin(actual, i).nonzero()[0]) for i in actual_unique]
    colors_unique  = np.array([S2COLOR(i) for i in actual_unique], dtype=str)
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.1
    fig   = plt.figure(figsize=(w, goldh))

    box_traj = dict(left=0.13, right=0.90, bottom=0.60, top=0.95)
    box_hist = dict(left=0.13, right=0.70, bottom=0.12, top=0.45)
    
    ## a -- individual beneficial/neutral/deleterious trajectories and selection coefficients

    ### trajectories
    
    n_exs  = len(inferred)
    colors = [C_BEN, C_BEN, C_NEU, C_NEU, C_DEL, C_DEL]
    
    gs = gridspec.GridSpec(n_exs, 2, width_ratios=[1, 1], wspace=0.05, **box_traj)
    ax = [[plt.subplot(gs[i, 0]), plt.subplot(gs[i, 1])] for i in range(n_exs)]
    
    legendprops = { 'loc' : 4, 'frameon' : False, 'scatterpoints' : 1, 'handletextpad' : 0.1,
                    'prop' : {'size' : SIZELABEL}, 'ncol' : 1 }
    lineprops   = { 'lw' : SIZELINE*1.5, 'linestyle' : '-', 'alpha' : 1.0 }
    fillprops   = { 'lw' : 0, 'alpha' : 0.3, 'interpolate' : True }
    
    pprops = { 'xticks' : [],
               'yticks' : [],
               'hide'   : ['top', 'bottom', 'left', 'right'],
               'theme'  : 'open' }
    
    for i in range(n_exs):
        pprops['colors'] = [colors[i]]
        pprops['xlim']   = [    0,  50]
        pprops['ylim']   = [-0.08, 1.08]
        if (i==n_exs-1):
            pprops['xticks']   = [0, 10, 20, 30, 40, 50]
            pprops['xlabel']   = 'Time (serial intervals)'
            pprops['hide']     = ['top', 'left', 'right']
            pprops['axoffset'] = 0.3
        for k in range(simulations):
            if allele_number[i] in mutant_sites[k]:
                loc = np.where(np.array(mutant_sites[k])==allele_number[i])[0][0]
                ydat = traj[k][loc]
                xdat = np.arange(len(traj[k][loc]))*record[k]
            else:
                ydat = np.zeros(len(traj[0]))
                xdat = np.arange(len(traj[0]))*record[k]
        mp.line(             ax=ax[i][0], x=[xdat], y=[ydat], plotprops=lineprops, **pprops)
        mp.plot(type='fill', ax=ax[i][0], x=[xdat], y=[ydat], plotprops=fillprops, **pprops)
    
    ### selection coefficient estimates
    
    dashlineprops = { 'lw' : SIZELINE * 2.0, 'ls' : ':', 'alpha' : 0.5 }
    sprops = dict(lw=0, s=9., marker='o')
    
    pprops = { 'yticks': [],
               'xticks': [],
               'hide'  : ['top', 'bottom', 'left', 'right'],
               'theme' : 'open' }
    
    for i in range(n_exs):
        pprops['xlim'] = [-0.04, 0.04]
        pprops['ylim'] = [  0.5,  1.5]
        ydat           = [1]
        xdat           = [inferred[i]]
        xerr           = error[i]
        if (i==n_exs-1):
            pprops['xticks']      = [-0.04,   -0.02,      0,   0.02,   0.04]
            pprops['xticklabels'] = [  ' ', r'$-2$', r'$0$', r'$2$', r'$4$']
            pprops['xlabel']      = 'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)'
            pprops['hide']        = ['top', 'left', 'right']
            pprops['axoffset']    = 0.3
        mp.plot(type='error', ax=ax[i][1], x=[xdat], y=[ydat], xerr=[xerr], colors=[colors[i]], **pprops)
        ax[i][1].axvline(x=actual_unique[(-(i//2) + 2)%3], color=BKCOLOR, **dashlineprops)
        
    ### add legend
    
    colors    = [C_BEN, C_NEU, C_DEL]
    colors_lt = [C_BEN_LT, C_NEU_LT, C_DEL_LT]
    plotprops = DEF_ERRORPROPS.copy()
    plotprops['clip_on'] = False
    
    invt = ax[-1][-1].transData.inverted()
    xy1  = invt.transform((   0, 0))
    xy2  = invt.transform((7.50, 9))
    xy3  = invt.transform((3.00, 9))
    xy4  = invt.transform((5.25, 9))

    legend_dx1 = xy1[0]-xy2[0]
    legend_dx2 = xy1[0]-xy3[0]
    legend_dx  = xy1[0]-xy4[0]
    legend_dy  = xy1[1]-xy2[1]
    legend_x   =  0.015
    legend_y   = [-4, -4 + legend_dy, -4 + (2*legend_dy)]
    legend_t   = ['Beneficial', 'Neutral', 'Deleterious']
    for k in range(len(legend_y)):
        mp.error(ax=ax[-1][-1], x=[[legend_x+legend_dx]], y=[[legend_y[k]]], edgecolor=[colors[k]], facecolor=[colors_lt[k]], plotprops=plotprops, **pprops)
        ax[-1][-1].text(legend_x, legend_y[k], legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    yy = -4 + (3.5*legend_dy)
    mp.line(ax=ax[-1][-1], x=[[legend_x+legend_dx2, legend_x+legend_dx1]], y=[[yy, yy]], colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':', clip_on=False), **pprops)
    ax[-1][-1].text(legend_x, yy, 'True selection\ncoefficient', ha='left', va='center', **DEF_LABELPROPS)
    
    yy = -4 + (5.5*legend_dy)
    mp.line(ax=ax[-1][-1], x=[[legend_x+legend_dx2, legend_x+legend_dx1]], y=[[yy, yy]], colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, clip_on=False), **pprops)
    ax[-1][-1].text(legend_x, yy, 'Mean inferred\ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    ### bounding boxes

    ax[0][0].text(box_traj['left']-0.08, box_traj['top']+0.01, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax[0][0].text(box_hist['left']-0.08, box_hist['top']+0.03, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    lineprops = { 'lw' : AXWIDTH/2., 'ls' : '-', 'alpha' : 1.0 }
    pprops    = { 'xlim' : [0, 1], 'xticks' : [], 'ylim' : [0, 1], 'yticks' : [],
        'hide' : ['top','bottom','left','right'], 'plotprops' : lineprops }
    txtprops = {'ha' : 'right', 'va' : 'center', 'color' : BKCOLOR, 'family' : FONTFAMILY,
        'size' : SIZELABEL, 'rotation' : 90, 'transform' : fig.transFigure}
    ax[0][0].text(box_traj['left']-0.025, box_traj['top'] - (box_traj['top']-box_traj['bottom'])/2., 'Frequency', **txtprops)

#    boxprops = {'ec' : BKCOLOR, 'lw' : SIZELINE/2., 'fc' : 'none', 'clip_on' : False, 'zorder' : -100}
#
#    dx  = 0.005
#    dxl = 0.000
#    dxr = 0.001
#    dy  = 0.003
#    ll = box_traj['left'] + dxl                     # left box left
#    lb = box_traj['bottom'] - dy                      # left box bottom
#    rl = (box_traj['left'] + box_traj['right'])/2. + dx + dxl # right box left
#    wd = (box_traj['right'] - box_traj['left'])/2. - dxr      # box width
#    ht = (box_traj['top'] - box_traj['bottom']) + (2. * dy)   # box height
#
#    recL = matplotlib.patches.Rectangle(xy=(ll, lb), width=wd, height=ht, transform=fig.transFigure, **boxprops)
#    recL = ax[0][0].add_patch(recL)
#    recR = matplotlib.patches.Rectangle(xy=(rl, lb), width=wd, height=ht, transform=fig.transFigure, **boxprops)
#    recR = ax[0][0].add_patch(recR)

    # b -- distribution of inferred selection coefficients
    
    rep_data = np.load(replicate_file, allow_pickle=True)
    positive, negative, zero = [], [], []
    for sim in range(len(rep_data['inferred'])):
        traj     = trajectory_reshape(rep_data['traj'][sim])
        inferred = rep_data['inferred'][sim]
        actual   = rep_data['actual'][sim]
        mutants  = rep_data['mutant_sites'][sim]
        error    = rep_data['errors'][sim]
        mutant_sites_all = np.sort(np.unique(np.array([mutants[i][j] for i in range(len(mutants))
                                               for j in range(len(mutants[i]))])))
        selection = {}
        for i in range(len(mutant_sites_all)):
            selection[mutant_sites_all[i]] = actual[i]
        allele_number = find_significant_sites(traj, mutants, cutoff)
        L_traj = len(allele_number)
        selection_new = {}
        for i in range(L_traj):
            selection_new[allele_number[i]] = selection[allele_number[i]]
        actual           = [selection_new[i] for i in allele_number]
        mutant_sites_all = np.sort(np.unique(np.array([mutants[i][j] for i in range(len(mutants))
                                               for j in range(len(mutants[i]))])))
        inferred  = filter_sites_infer(traj, inferred, mutants, mutant_sites_all, allele_number)
        inf_minus, inf_zero, inf_plus = sort_by_sign(selection_new, inferred)
        positive += inf_plus
        negative += inf_minus
        zero     += inf_zero
    
    ### plot histogram
    
    gs_hist = gridspec.GridSpec(1, 1, **box_hist)
    ax_hist = plt.subplot(gs_hist[0, 0])

    colors     = [C_DEL, C_NEU, C_BEN]
    tags       = ['deleterious', 'neutral', 'beneficial']
    s_true_loc = [-s_value, 0, s_value]

    dashlineprops  = { 'lw' : SIZELINE * 2.0, 'ls' : ':', 'alpha' : 0.5 }
    solidlineprops = { 'lw' : SIZELINE * 2.0, 'alpha' : 0.5 }
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.7, edgecolor='none',
                     orientation='vertical')
    pprops = { 'xlim':        [-0.10, 0.10],
               'xticks':      [   -0.10,    -0.075,  -0.05,     -0.025,     0.,   0.025,   0.05,    0.075,     0.10],
               'xticklabels': [r'$-10$', r'$-7.5$', r'$-5$', r'$-2.5$', r'$0$', r'$2.5$', r'$5$', r'$7.5$', r'$10$'],
               'ylim':        [0., 0.075],
               'yticks':      [0., 0.025, 0.05, 0.075],
               'yticklabels': [r'$0$', r'$2.5$', r'$5$', r'$7.5$'],
               'ylabel':      'Frequency (%)',
               'xlabel':      'Inferred selection coefficient, ' + r'$\hat{s}$' + ' (%)',
               'bins':        np.arange(-0.10, 0.10+0.002, 0.002),
               'combine':     True,
               'plotprops':   histprops,
               'axoffset':    0.1,
               'theme':       'open' }

    s_vals = [negative, zero, positive]
    for i in range(len(tags)):
        x = [s_vals[i]]
        ax_hist.axvline(x=s_true_loc[i], color=colors[i], **dashlineprops)
        if i<len(tags)-1: mp.hist(             ax=ax_hist, x=x, colors=[colors[i]], **pprops)
        else:             mp.plot(type='hist', ax=ax_hist, x=x, colors=[colors[i]], **pprops)
        
    for i in range(len(tags)):
        x = np.mean(s_vals[i])
        ax_hist.axvline(x=x, color=colors[i], **solidlineprops)

    # SAVE FIGURE

    plt.savefig('%s/fig-1%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_selection_statistics_mutations(selection_file_all, selection_file_ns, mutation_file, label2ddr=[]):
    """ Makes a plot containing the distribution of inferred coefficients and the enrichment of nonsynonymous mutations.
        Combines selection_dist_plot and syn_plot functions.
        correlation_plots determines whether or not the 3 plots containing the correlations are made."""
    
    # loading and processing the data
    df_sel_all = pd.read_csv(selection_file_all, comment='#', memory_map=True)
        
    # PLOT FIGURE
    
    ## set up figure grid
    
    # # OLD VERSION FOR TOP 45
    # w       = DOUBLE_COLUMN
    # hshrink = 0.85/0.60
    # goldh   = SINGLE_COLUMN * hshrink
    # fig     = plt.figure(figsize=(w, goldh))

    # x_scale = SINGLE_COLUMN / DOUBLE_COLUMN
    # l_hist  = 0.15
    # x_hist  = 0.95 - 0.15
    # l_circ  = 0.07
    # x_circ  = 0.92 - 0.07
    # l_mut   = 0.88 #0.55 

    # box_hist = dict(left=l_hist*x_scale, right=(l_hist+x_hist)*x_scale, bottom=0.80, top=0.95)
    # box_circ = dict(left=l_circ*x_scale, right=(l_circ+x_circ)*x_scale, bottom=0.05, top=0.65)
    # # box_mut  = dict(left=l_mut, right=0.98, bottom=0.13, top=0.98)
    # box_mut  = dict(left=l_mut, right=0.98, bottom=0.175, top=0.94)

    # RESIZED TO FIT ALL TOP 50
    w       = DOUBLE_COLUMN
    hshrink = 0.85/0.60
    goldh   = SINGLE_COLUMN * hshrink
    v_exp   = 0.5
    v_scale = (SINGLE_COLUMN * hshrink) / (SINGLE_COLUMN * hshrink + v_exp)
    fig     = plt.figure(figsize=(w, goldh + v_exp))

    x_scale = SINGLE_COLUMN / DOUBLE_COLUMN
    l_hist  = 0.15
    x_hist  = 0.95 - 0.15
    l_circ  = 0.07
    x_circ  = 0.92 - 0.07
    l_mut   = 0.88 #0.55 
    ar      = w / (goldh + v_exp)
    circ_dy = 0.03

    box_hist = dict(left=l_hist*x_scale, right=(l_hist+x_hist)*x_scale, bottom=0.95 - 0.20*v_scale, top=0.95)
    box_circ = dict(left=l_circ*x_scale, right=(l_circ+x_circ)*x_scale + circ_dy, bottom=0.95 - 0.95*v_scale - ar*circ_dy, top=0.95 - 0.35*v_scale)
    box_mut  = dict(left=l_mut, right=0.98, bottom=0.175, top=0.94)
    
    ## a -- selection coefficient histogram
    
    ### plot histogram
    
    s_link    = np.array(df_sel_all['selection coefficient'])
    s_pos_log = np.log10(s_link[s_link>0])
    s_neg_log = np.log10(-s_link[s_link<0])
    
    n_hist = 15
    xmin   = -2.5
    xmax   = -0.5
    x_neg  = np.linspace(xmin, xmax, num=n_hist)
    x_pos  = np.linspace(xmin, xmax, num=n_hist)
    y_neg  = []
    y_pos  = []
    y_neu  = []
    for i in range(len(x_neg)-1):
        y_neg.append(np.sum((s_neg_log>x_neg[i]) * (s_neg_log<=x_neg[i+1])))
    y_neg.append(np.sum(s_neg_log>x_neg[-1]))
    y_neu.append(np.sum(s_neg_log<x_neg[0]))
    for i in range(len(x_pos)-1):
        y_pos.append(np.sum((s_pos_log>x_pos[i]) * (s_pos_log<=x_pos[i+1])))
    y_pos.append(np.sum(s_pos_log>x_pos[-1]))
    y_neu.append(np.sum(s_pos_log<x_pos[0]))
    
    norm  = 4
    c_neg = [hls_to_rgb(0.02, 1., 0.83)]
    c_pos = [hls_to_rgb(0.58, 1., 0.60)]
    for i in range(len(y_neg)):
        t = np.min([(x_neg[i] + norm)/norm, 1])
        c_neg.append(hls_to_rgb(0.58, 0.53 * t + 1. * (1 - t), 0.60))
        t = np.min([(x_pos[i] + norm)/norm, 1])
        c_pos.append(hls_to_rgb(0.02, 0.53 * t + 1. * (1 - t), 0.83))
        
    wspace  = 0.20
    gs_hist = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=wspace, **box_hist)
    
    hist_props = dict(lw=SIZELINE/2, width=1.5/n_hist, align='center', edgecolor=[BKCOLOR], orientation='vertical')
    
    ### negative coefficient subplot
    
    ax_neg = plt.subplot(gs_hist[0, 0])
    
    pprops = { 'colors':      c_neg,
               'xlim':        [xmin-4.5/n_hist, -0.5][::-1],
               'xticks':      [-2.5-3/n_hist, -2, -1, -0.5],
               'xticklabels': ['>-0.3', '-1', '-10', ''],
               'logy':        True,
               'ylim':        [1e0, 1e3],
               'yticks':      [1e0, 1e1, 1e2, 1e3],
               'ylabel':      'Counts',
               'hide':        ['top', 'right'] }
    
    x_neg = [[-2.5-2/n_hist]] + [[i] for i in x_neg]
    y_neg = [[y_neu[0]]] + [[i] for i in y_neg]
    
    mp.plot(type='bar', ax=ax_neg, x=x_neg, y=y_neg, plotprops=hist_props, **pprops)
    
    ### positive coefficient subplot
    
    ax_pos = plt.subplot(gs_hist[0, 1])
    
    pprops = { 'colors':      c_pos,
               'xlim':        [xmin-4.5/n_hist, -0.5],
               'xticks':      [-2.5-3/n_hist, -2, -1, -0.5],
               'xticklabels': ['<0.3', '1', '10', ''],
               'logy':        True,
               'ylim':        [1e0, 1e3],
               'yticks':      [],
               'hide':        ['top', 'right', 'left'] }
               
    x_pos = [[-2.5-2/n_hist]] + [[i] for i in x_pos]
    y_pos = [[y_neu[1]]] + [[i] for i in y_pos]
    
    mp.plot(type='bar', ax=ax_pos, x=x_pos, y=y_pos, plotprops=hist_props, **pprops)
    
    ## b -- circle plot

    gs_circ = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_circ)
    ax_circ = plt.subplot(gs_circ[0, 0])

    df_sel = pd.read_csv(selection_file_ns, comment='#', memory_map=True)
    
    sig_s, sig_site_real, sig_nuc_idx = plot_circle_formatter(ax_circ, df_sel, label2ddr)

    ## c -- mutation effects

    df_mut = pd.read_csv(mutation_file, comment='#', memory_map=True)
    n_mut  = len(df_mut)
    # n_mut = 45

    gs_mut = gridspec.GridSpec(n_mut, 1, **box_mut)
    ax     = [plt.subplot(gs_mut[i, 0]) for i in range(n_mut)]

    dashlineprops = dict(lw=SIZELINE, ls=':', alpha=0.25, clip_on=False)

    pprops = { 'yticks': [],
               'xticks': [],
               'hide'  : ['top', 'bottom', 'left', 'right'],
               'theme' : 'open' }
    
    xticks = [  0, 5, 10, 15]
    xmax   = 18

    t_prop_mut  = dict(ha='right', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
    t_mut_start = -52

    var_start     = -50
    # plot_vars     = ['Alpha', 'Delta', 'Gamma', 'BA.1', 'BA.2', 'BA.5', 'XBB.1.5', 'EG.5.1', 'JN.1']
    plot_vars     = ['Epsilon', 'Alpha', 'Beta', 'Lambda', 'Gamma', 'Mu', 'Delta', 'BA.1', 'BA.2', 'BA.4', 'BA.5', 'XBB', 'XBB.1.5', 'EG.5.1', 'BA.2.86', 'JN.1']
    x_var         = 1.8
    x_spc         = 1.2
    var_rec_props = dict(height=1, width=x_var, ec=BKCOLOR,  lw=AXWIDTH/2, clip_on=False)
    not_rec_props = dict(height=1, width=x_var, ec=C_NEU_LT, lw=AXWIDTH/2, clip_on=False)
    var_colors    = [VAR_N2C[v] for v in plot_vars]
    t_prop_var    = dict(ha='center', va='top', rotation=90, color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, clip_on=False)

    phen_start  = var_start + x_spc*x_var*(len(plot_vars) + 1)
    phen_fx     = ['Ab escape', 'Other imm. evasion', 'Replication/infectivity', 'Cell entry', 'Structure/cleavage']
    fx2col      = ['Neutralization Resistance (RBD/NTD)', 'Immune Evasion (IFN-mediated/T cell)', 'Viral Replication/Infectivity', 
                   'ACE2 Binding/Cell Entry', 'Proteolytic Activity/Structural Stability']
    # phen_colors = sns.color_palette('RdPu', 3*len(phen_fx))
    phen_colors = sns.color_palette('Blues', 3*len(phen_fx))
    phen_ci     = 3
    phen_props  = DEF_ERRORPROPS.copy()
    phen_props['clip_on'] = False
    phen_props['markersize'] = SMALLSIZEDOT/1.4
    p_spc = 1.3

    for i in range(n_mut):

        # Plot selection coefficients
        df_entry = df_mut[df_mut.Rank==i+1].iloc[0]
        pprops['xlim'] = [  0, xmax]
        pprops['ylim'] = [ 0.5, 1.5]
        ydat           = [1]
        xdat           = [df_entry.Selection]
        xerr           = [df_entry.Standard_deviation]
        if (i==n_mut-1):
            pprops['xticks']   = xticks
            pprops['xlabel']   = 'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)'
            pprops['hide']     = ['top', 'left', 'right']
            pprops['axoffset'] = 0.3

            for j in range(len(plot_vars)):
                ax[i].text(var_start + x_spc*x_var*j, -0.2, plot_vars[j], **t_prop_var)

            for j in range(len(phen_fx)):
                ax[i].text(phen_start + p_spc*x_var*j, -0.2, phen_fx[j], **t_prop_var)
        
        norm = 18
        ds = -3
        # t = np.min([(xdat[0] + norm)/norm, 1])
        t = (xdat[0]-ds)/(norm-ds)
        c = hls_to_rgb(0.02, 0.53 * t + 1. * (1 - t), 0.83)
        mp.plot(type='error', ax=ax[i], x=[xdat], y=[ydat], xerr=[xerr], colors=[c], **pprops)

        # Mutation name
        prot   = df_entry.Protein
        aa_mut = df_entry.Amino_Acid_Mutation
        if aa_mut[-1]=='-':
            aa_mut = r'$\Delta$' + aa_mut[1:-1]
        mut_str = '%s %s' % (prot, aa_mut)
        ax[i].text(t_mut_start, 1, mut_str, **t_prop_mut)

        # Presence in variants
        var_patches = []
        
        for j in range(len(plot_vars)):
            if df_entry[plot_vars[j]]==True:
                rec = matplotlib.patches.Rectangle(xy=(var_start + x_spc*x_var*j - x_var/2, 0.6), fc=var_colors[j], **var_rec_props)
                var_patches.append(rec)
            else:
                rec = matplotlib.patches.Rectangle(xy=(var_start + x_spc*x_var*j - x_var/2, 0.6), fc='none', **not_rec_props)
                var_patches.append(rec)

        for patch in var_patches:
            ax[i].add_artist(patch)

        # Phenotypic effects
        for j in range(len(phen_fx)):
            if df_entry[fx2col[j]]==True:
                mp.error(ax=ax[i], x=[[phen_start + p_spc*x_var*j]], y=[[1]], edgecolor=[BKCOLOR], facecolor=[phen_colors[j+phen_ci]], plotprops=phen_props, **pprops)
            else:
                mp.error(ax=ax[i], x=[[phen_start + p_spc*x_var*j]], y=[[1]], edgecolor=[C_NEU_LT], facecolor=['none'], plotprops=phen_props, **pprops)
        
    for i in range(len(xticks)):
        xdat = [xticks[i], xticks[i]]
        ydat = [-0.5, 1.23*n_mut]
        mp.line(ax=ax[-1], x=[xdat], y=[ydat], colors=[BKCOLOR], plotprops=dashlineprops)

    # MAKE LEGEND
    
    ax_neg.text(box_hist['left'] + (box_hist['right']-box_hist['left'])/2, box_hist['bottom']-0.06, 'Inferred selection coefficients, ' + '$\hat{s}$' + ' (%)',
                ha='center', va='top', transform=fig.transFigure, **DEF_LABELPROPS)

    invt = ax_circ.transData.inverted()
    xy1  = invt.transform((0,0))
    xy2  = invt.transform((0,9))
    
    coef_legend_x  = 0.60
    coef_legend_dx = 0.05
    coef_legend_y  = -0.89
    coef_legend_dy = 1.5*(xy1[1]-xy2[1])

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
    ex_s     = [-0.10, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10]
    show_s   = [    1,     0,     0,     0,     0, 1,    0,    0,    0,    0,    1]
    c_s      = [C_DEL, C_BEN]
    for i in range(len(ex_s)):
        plotprops = dict(lw=0, marker='o', s=(np.fabs(ex_s[i]) - DS_NORM)*S_MULT if ex_s[i]!=0 else 0, clip_on=False)
        mp.scatter(ax=ax_circ, x=[[coef_legend_x + i*coef_legend_dx]], y=[[coef_legend_y]], colors=[c_s[ex_s[i]>0]], plotprops=plotprops)
        if show_s[i]:
            ax_circ.text(coef_legend_x + i*coef_legend_dx, coef_legend_y + 0.75*coef_legend_dy, '%d' % (100*ex_s[i]), **txtprops)
    ax_circ.text(coef_legend_x + len(ex_s)*coef_legend_dx/2, coef_legend_y + (2.25*coef_legend_dy), 'Inferred selection\ncoefficient, $\hat{s}$ (%)', **txtprops)

    txt_y = 3.6
    var_x = (var_start + var_start + x_spc*x_var*(len(plot_vars) - 1))/2
    ax[0].text(var_x, txt_y, 'Presence in\nmajor variants', **txtprops)

    phen_x = (phen_start + phen_start + p_spc*x_var*(len(phen_fx) - 1))/2
    ax[0].text(phen_x, txt_y, 'Phenotypic\neffects', **txtprops)

    s_x = (0 + xmax)/2
    ax[0].text(s_x, txt_y, 'Effect on\ntransmission', **txtprops)
    
    ax_neg.text( 0.025, box_hist['top']+0.01, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_circ.text(0.025, box_circ['top']+0.01, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_circ.text(0.520, box_hist['top']+0.01, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    # SAVE FIGURE

    plt.savefig('%s/fig-2%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    
    
def plot_selection_statistics_extended(selection_file_all, selection_file_ns, label2ddr=[]):
    """ Makes a plot containing the distribution of inferred coefficients and the enrichment of nonsynonymous mutations.
        Combines selection_dist_plot and syn_plot functions.
        correlation_plots determines whether or not the 3 plots containing the correlations are made."""
    
    # loading and processing the data
    df_sel_all = pd.read_csv(selection_file_all, comment='#', memory_map=True)
        
    # PLOT FIGURE
    
    ## set up figure grid
    
    w       = SINGLE_COLUMN
    hshrink = 0.85/0.60
    goldh   = w * hshrink
    fig     = plt.figure(figsize=(w, goldh))

    box_hist = dict(left=0.15, right=0.95, bottom=0.80, top=0.95)
    box_circ = dict(left=0.07, right=0.92, bottom=0.05, top=0.65)
    
    ## a -- selection coefficient histogram
    
    ### plot histogram
    
    s_link    = np.array(df_sel_all['selection coefficient'])
    s_pos_log = np.log10(s_link[s_link>0])
    s_neg_log = np.log10(-s_link[s_link<0])
    
    n_hist = 15
    xmin   = -2.5
    xmax   = -0.5
    x_neg  = np.linspace(xmin, xmax, num=n_hist)
    x_pos  = np.linspace(xmin, xmax, num=n_hist)
    y_neg  = []
    y_pos  = []
    y_neu  = []
    for i in range(len(x_neg)-1):
        y_neg.append(np.sum((s_neg_log>x_neg[i]) * (s_neg_log<=x_neg[i+1])))
    y_neg.append(np.sum(s_neg_log>x_neg[-1]))
    y_neu.append(np.sum(s_neg_log<x_neg[0]))
    for i in range(len(x_pos)-1):
        y_pos.append(np.sum((s_pos_log>x_pos[i]) * (s_pos_log<=x_pos[i+1])))
    y_pos.append(np.sum(s_pos_log>x_pos[-1]))
    y_neu.append(np.sum(s_pos_log<x_pos[0]))
    
    norm  = 4
    c_neg = [hls_to_rgb(0.02, 1., 0.83)]
    c_pos = [hls_to_rgb(0.58, 1., 0.60)]
    for i in range(len(y_neg)):
        t = np.min([(x_neg[i] + norm)/norm, 1])
        c_neg.append(hls_to_rgb(0.58, 0.53 * t + 1. * (1 - t), 0.60))
        t = np.min([(x_pos[i] + norm)/norm, 1])
        c_pos.append(hls_to_rgb(0.02, 0.53 * t + 1. * (1 - t), 0.83))
        
    wspace  = 0.20
    gs_hist = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=wspace, **box_hist)
    
    hist_props = dict(lw=SIZELINE/2, width=1.5/n_hist, align='center', edgecolor=[BKCOLOR], orientation='vertical')
    
    ### negative coefficient subplot
    
    ax_neg = plt.subplot(gs_hist[0, 0])
    
    pprops = { 'colors':      c_neg,
               'xlim':        [xmin-4.5/n_hist, -0.5][::-1],
               'xticks':      [-2.5-3/n_hist, -2, -1, -0.5],
               'xticklabels': ['>-0.3', '-1', '-10', ''],
               'logy':        True,
               'ylim':        [1e0, 1e3],
               'yticks':      [1e0, 1e1, 1e2, 1e3],
               'ylabel':      'Counts',
               'hide':        ['top', 'right'] }
    
    x_neg = [[-2.5-2/n_hist]] + [[i] for i in x_neg]
    y_neg = [[y_neu[0]]] + [[i] for i in y_neg]
    
    mp.plot(type='bar', ax=ax_neg, x=x_neg, y=y_neg, plotprops=hist_props, **pprops)
    
    ### positive coefficient subplot
    
    ax_pos = plt.subplot(gs_hist[0, 1])
    
    pprops = { 'colors':      c_pos,
               'xlim':        [xmin-4.5/n_hist, -0.5],
               'xticks':      [-2.5-3/n_hist, -2, -1, -0.5],
               'xticklabels': ['<0.3', '1', '10', ''],
               'logy':        True,
               'ylim':        [1e0, 1e3],
               'yticks':      [],
               'hide':        ['top', 'right', 'left'] }
               
    x_pos = [[-2.5-2/n_hist]] + [[i] for i in x_pos]
    y_pos = [[y_neu[1]]] + [[i] for i in y_pos]
    
    mp.plot(type='bar', ax=ax_pos, x=x_pos, y=y_pos, plotprops=hist_props, **pprops)
    
    ## b -- circle plot

    gs_circ = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_circ)
    ax_circ = plt.subplot(gs_circ[0, 0])

    df_sel = pd.read_csv(selection_file_ns, comment='#', memory_map=True)
    
    sig_s, sig_site_real, sig_nuc_idx = plot_circle_formatter(ax_circ, df_sel, label2ddr)

    # MAKE LEGEND
    
    ax_neg.text(box_hist['left'] + (box_hist['right']-box_hist['left'])/2, box_hist['bottom']-0.06, 'Inferred selection coefficients, ' + '$\hat{s}$' + ' (%)',
                ha='center', va='top', transform=fig.transFigure, **DEF_LABELPROPS)

    invt = ax_circ.transData.inverted()
    xy1  = invt.transform((0,0))
    xy2  = invt.transform((0,9))
    
    coef_legend_x  = 0.60
    coef_legend_dx = 0.05
    coef_legend_y  = -0.89
    coef_legend_dy = 1.5*(xy1[1]-xy2[1])

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
    ex_s     = [-0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05]
    show_s   = [    1,     0,     0,     0,     0, 1,    0,    0,    0,    0,    1]
    c_s      = [C_DEL, C_BEN]
    for i in range(len(ex_s)):
        plotprops = dict(lw=0, marker='o', s=(np.fabs(ex_s[i]) - DS_NORM)*S_MULT if ex_s[i]!=0 else 0, clip_on=False)
        mp.scatter(ax=ax_circ, x=[[coef_legend_x + i*coef_legend_dx]], y=[[coef_legend_y]], colors=[c_s[ex_s[i]>0]], plotprops=plotprops)
        if show_s[i]:
            ax_circ.text(coef_legend_x + i*coef_legend_dx, coef_legend_y + 0.75*coef_legend_dy, '%d' % (100*ex_s[i]), **txtprops)
    ax_circ.text(coef_legend_x + len(ex_s)*coef_legend_dx/2, coef_legend_y + (2.25*coef_legend_dy), 'Inferred selection\ncoefficient, $\hat{s}$ (%)', **txtprops)
    
    ax_neg.text( 0.05, box_hist['top']+0.01, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_circ.text(0.05, box_circ['top']+0.01, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    # SAVE FIGURE

    plt.savefig('%s/fig-2%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_grant(selection_args, variant_args): #selection_file_all, selection_file_ns, label2ddr=[]):
    """ Makes a plot containing the distribution of inferred coefficients and the enrichment of nonsynonymous mutations.
        Combines selection_dist_plot and syn_plot functions.
        correlation_plots determines whether or not the 3 plots containing the correlations are made."""
    
    # load data
    selection_file_all = selection_args['selection_file_all']
    selection_file_ns  = selection_args['selection_file_ns']
    label2ddr = []
    if 'label2ddr' in selection_args:
        label2ddr = selection_args['label2ddr']

    variant_file = variant_args['variant_file']
    trajectory_file = variant_args['trajectory_file']
    optional_args = ['variant_list', 'variant_names', 's_cutoff', 'time_set']
    default_vals = [[], [], 0.05, 1/1.1, 'late']
    for i in range(len(optional_args)):
        if optional_args[i] not in variant_args:
            variant_args[optional_args[i]] = default_vals[i]

    variant_list = variant_args['variant_list']
    variant_names = variant_args['variant_names']
    s_cutoff = variant_args['s_cutoff']
    time_set = variant_args['time_set']
        
    # PLOT FIGURE
    
    ## set up figure grid

    spc = 0.07
    dx  = 0.02
    hshrink = 0.9
    dy  = (1 - 2*spc) * (1 - 1/hshrink) / 2
    
    w     = 3.35
    goldh = w * hshrink * (1 - 2*spc - 2*dx) / (1 - 2*spc)
    xh    = 0.8 * goldh
    ratio = goldh / (goldh + xh)
    fig   = plt.figure(figsize=(w, goldh + xh))

    box_t    = 0.98
    box_b    = 0.10 
    circ_y   = 1 - 2*spc - 2*dy
    box_circ = dict(left=spc+dx, right=1-spc-dx, bottom=box_t - ratio*circ_y, top=box_t)
    box_traj = dict(left=spc+dx, right=1-spc-dx, bottom=box_b,                top=box_t - ratio*circ_y)
    
    ## circle plot

    gs_circ = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_circ)
    ax_circ = plt.subplot(gs_circ[0, 0])

    df_sel = pd.read_csv(selection_file_ns, comment='#', memory_map=True)
    
    sig_s, sig_site_real, sig_nuc_idx = plot_circle_formatter(ax_circ, df_sel, label2ddr)

    # MAKE LEGEND

    invt = ax_circ.transData.inverted()
    xy1  = invt.transform((0,0))
    xy2  = invt.transform((0,9))
    
    coef_legend_dx = 0.07
    coef_legend_x  = -10 * coef_legend_dx / 2
    coef_legend_y  = 0.12
    coef_legend_dy = 1.5*(xy1[1]-xy2[1])

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
    ex_s     = [-0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05]
    show_s   = [    1,     0,     0,     0,     0, 1,    0,    0,    0,    0,    1]
    c_s      = [C_DEL, C_BEN]
    for i in range(len(ex_s)):
        plotprops = dict(lw=0, marker='o', s=(np.fabs(ex_s[i]) - DS_NORM)*S_MULT if ex_s[i]!=0 else 0, clip_on=False)
        mp.scatter(ax=ax_circ, x=[[coef_legend_x + i*coef_legend_dx]], y=[[coef_legend_y]], colors=[c_s[ex_s[i]>0]], plotprops=plotprops)
        if show_s[i]:
            ax_circ.text(coef_legend_x + i*coef_legend_dx, coef_legend_y + 0.75*coef_legend_dy, '%d' % (100*ex_s[i]), **txtprops)
    ax_circ.text(coef_legend_x + len(ex_s)*coef_legend_dx/2, coef_legend_y + (2.25*coef_legend_dy), 'Inferred selection\ncoefficient (%)', **txtprops)

    # plot trajectories

    # adjust color list if necessary
    local_colors = deepcopy(VAR_COLORS)
    if len(variant_names)>8:
        local_colors = sns.husl_palette(len(variant_names))
    
    # Load data
    df_var = pd.read_csv(variant_file, comment='#', memory_map=True)
    df_sel = 0
    if len(variant_list)>0:
        df_sel = df_var[df_var.variant_names.isin(variant_list)]
        temp_var_list  = list(df_sel.variant_names)
        temp_var_names = []
        for i in range(len(temp_var_list)):
            temp_var_names.append(variant_names[variant_list.index(temp_var_list[i])])
        variant_list  = np.array(temp_var_list)
        variant_names = np.array(temp_var_names)
    else:
        df_sel = df_var[np.fabs(df_var.selection_coefficient)>s_cutoff]
        variant_list  = np.array(list(df_sel.variant_names))
        variant_names = np.array(variant_list)
    
    s_sort = np.argsort(np.fabs(df_sel.selection_coefficient))[::-1]
    
    print('variant\tw')
    for i in range(len(df_sel)):
        print('%s\t%.3f' % (np.array(df_sel.variant_names)[s_sort][i], np.array(np.fabs(df_sel.selection_coefficient))[s_sort][i]))
    
    n_vars        = len(s_sort)
    variant_list  = variant_list[s_sort]
    variant_names = variant_names[s_sort]
    
    # df_traj = pd.read_csv(trajectory_file, comment='#', memory_map=True)
    # df_traj = df_traj[df_traj.variant_names.isin(variant_list)]

    df_traj = pd.read_csv('temp-trajectories.csv', comment='#', memory_map=True)
    df_traj = df_traj[df_traj.variant_names.isin(variant_names)]
    
    # PLOT FIGURE
    
    ## set up figure grid

    gs_traj = gridspec.GridSpec(n_vars, 2, width_ratios=[2, 1], wspace=0.05, **box_traj)
    
    ### trajectories
    
    ax = [[plt.subplot(gs_traj[i, 0]), plt.subplot(gs_traj[i, 1])] for i in range(n_vars)]
    
    legendprops = { 'loc' : 4, 'frameon' : False, 'scatterpoints' : 1, 'handletextpad' : 0.1,
                    'prop' : {'size' : SIZELABEL}, 'ncol' : 1 }
    lineprops   = { 'lw' : SIZELINE, 'linestyle' : '-', 'alpha' : 0.5 }
    
    pprops = { 'xticks' : [],
               'yticks' : [],
               'hide'   : ['top', 'bottom', 'left', 'right'],
               'theme'  : 'open' }
    
    t_max, xticks, xticklabels = 0, [], []
    
    if time_set=='early':
        t_max = 600
        xticks      = [      61,      153,      245,      336,      426,      518, 600]
        xticklabels = ['Mar 20', 'Jun 20', 'Sep 20', 'Dec 20', 'Mar 21', 'Jun 21',  '']

    elif time_set=='full':
        t_max = 900
        xticks      = [      61,      153,      245,      336,      426,      518,       609,      700,      790,      882]
#        xticklabels = ['Mar 20', 'Jun 20', 'Sep 20', 'Dec 20', 'Mar 21', 'Jun 21',  'Sep 21', 'Dec 21', 'Mar 22', 'Jun 22']
        xticklabels = ['Mar 20',       '', 'Sep 20',       '', 'Mar 21',       '',  'Sep 21',       '', 'Mar 22',       '']
    
    elif time_set=='late':
        t_max = 1550
        xticks      = [      61,      245,      426,       609,      791,      975,     1156,     1340,     1493]
        xticklabels = ['Mar 20', 'Sep 20', 'Mar 21',  'Sep 21', 'Mar 22', 'Sep 22', 'Mar 23', 'Sep 23', 'Feb 24']
        
    elif time_set=='vlate':
        t_max = 900
        xticks      = [      336,     426,      518,       609,      700,      790,      882]
        xticklabels = ['Dec 20', 'Mar 21', 'Jun 21',  'Sep 21', 'Dec 21', 'Mar 22',       '']
    
    for i in range(n_vars):
        idx = list(s_sort)[i]
        pprops['colors'] = [local_colors[i]]
        pprops['xlim']   = [  np.min(xticks), t_max]
        pprops['ylim']   = [-0.05,  1.05]
        if (i==n_vars-1):
            pprops['xticks']      = xticks
            pprops['xticklabels'] = xticklabels
            pprops['hide']        = ['top', 'left', 'right']
            pprops['axoffset']    = 0.18
        # df_temp = df_traj[df_traj.variant_names==df_sel.iloc[idx].variant_names]
        vname = variant_names[list(variant_list).index(str(df_sel.iloc[idx].variant_names))]
        df_temp = df_traj[df_traj.variant_names==vname]
        for k in range(len(df_temp)):
            entry = df_temp.iloc[k]
            times = np.array(str(entry.times).split(), float)
            freqs = np.array(str(entry.frequencies).split(), float)
            if k==len(df_temp)-1:
                mp.plot(type='line', ax=ax[i][0], x=[times], y=[freqs], plotprops=lineprops, **pprops)
            else:
                mp.line(             ax=ax[i][0], x=[times], y=[freqs], plotprops=lineprops, **pprops)
            
        txtprops = dict(ha='left', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
        if variant_list[i]=='B.1':
            ax[i][0].text(101, 0.1, variant_names[i], **txtprops)
        elif variant_list[i]=='omicron':
            ax[i][0].text(np.min(xticks), 0.8, 'Omicron\n(BA.1)', **txtprops)
        else:
            ax[i][0].text(np.min(xticks), 0.8, variant_names[i], **txtprops)
    
    txtprops = dict(ha='center', va='center', rotation=90, color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, clip_on=False)
    ax[-1][0].text(box_traj['left']-0.04, box_traj['top']-(box_traj['top']-box_traj['bottom'])/2, 'Variant frequency', transform=fig.transFigure, **txtprops)
    
    for tick in ax[-1][0].get_xticklabels():
        tick.set_rotation(90)
            
    ### selection coefficient estimates
    
    dashlineprops = dict(lw=SIZELINE, ls=':', alpha=0.5, clip_on=False)
    sprops = dict(lw=0, s=9., marker='o')
    
    pprops = { 'yticks': [],
               'xticks': [],
               'hide'  : ['top', 'bottom', 'left', 'right'],
               'theme' : 'open' }
    
    xticks      = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    xticklabels = ['0',  '', '100', '', '200', '', '300']
    for i in range(n_vars):
        idx = list(s_sort)[i]
        pprops['xlim'] = [   0, 3.20]
        pprops['ylim'] = [ 0.5, 1.5]
        ydat           = [1]
        xdat           = [df_sel.iloc[idx].selection_coefficient]
        xerr           = [df_sel.iloc[idx]['theoretical_error']]
        if (i==n_vars-1):
            pprops['xticks']      = xticks
            pprops['xticklabels'] = xticklabels
            # pprops['xlabel']      = 'Inferred selection\ncoefficient, ' + r'$\hat{w}$' + ' (%)'
            pprops['xlabel']      = 'Inferred selection\ncoefficient (%)'
            pprops['hide']        = ['top', 'left', 'right']
            pprops['axoffset']    = 0.3
        mp.plot(type='error', ax=ax[i][1], x=[xdat], y=[ydat], xerr=[xerr], colors=[local_colors[i]], **pprops)
        
    #plt.xticks(rotation = 90)
        
    for i in range(len(xticks)):
        xdat = [xticks[i], xticks[i]]
        ydat = [0.5, n_vars+1.25]
        if n_vars>7:
            ydat[1] += 0.75
        mp.line(ax=ax[-1][1], x=[xdat], y=[ydat], colors=[BKCOLOR], plotprops=dashlineprops)

    ## legends and labels
    
    ax_circ.text(0.05, box_circ['top']-0.02, 'a'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_circ.text(0.05, box_traj['top']-0.01, 'b'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    # SAVE FIGURE

    plt.savefig('%s/fig-grant%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_variant_selection(variant_file, trajectory_file, variant_list=[], variant_names=[], s_cutoff=0.05, h_relative=1/1.1, box_dims=dict(left=0.08, right=0.95, bottom=0.15, top=0.97), time_set='late', filename='fig-3'):
    """ Map out mutations in the major variants, their aggregate inferred selection coefficients, and
        frequency trajectories across regions. """
    
    # Load data
    df_var = pd.read_csv(variant_file, comment='#', memory_map=True)
    df_sel = 0
    if len(variant_list)>0:
        df_sel = df_var[df_var.variant_names.isin(variant_list)]
        temp_var_list  = list(df_sel.variant_names)
        temp_var_names = []
        for i in range(len(temp_var_list)):
            temp_var_names.append(variant_names[variant_list.index(temp_var_list[i])])
        variant_list  = np.array(temp_var_list)
        variant_names = np.array(temp_var_names)
    else:
        df_sel = df_var[np.fabs(df_var.selection_coefficient)>s_cutoff]
        variant_list  = np.array(list(df_sel.variant_names))
        variant_names = np.array(variant_list)
    
    s_sort = np.argsort(np.fabs(df_sel.selection_coefficient))[::-1]
    
    print('variant\tw')
    for i in range(len(df_sel)):
        print('%s\t%.3f' % (np.array(df_sel.variant_names)[s_sort][i], np.array(np.fabs(df_sel.selection_coefficient))[s_sort][i]))
    
    n_vars        = len(s_sort)
    variant_list  = variant_list[s_sort]
    variant_names = variant_names[s_sort]

    # adjust color list if necessary
    has_colors = True
    for i in range(len(variant_names)):
        if variant_names[i] not in VAR_N2C.keys():
            has_colors = False
            break

    local_colors = sns.husl_palette(len(variant_names))
    if has_colors:
        local_colors = [VAR_N2C[c] for c in variant_names]
    
    df_traj = pd.read_csv(trajectory_file, comment='#', memory_map=True)
    df_traj = df_traj[df_traj.variant_names.isin(variant_list)]
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w * h_relative
    fig   = plt.figure(figsize=(w, goldh))

    box_traj = box_dims
    # box_traj = dict(left=0.08, right=0.95, bottom=0.17, top=0.97)  # SLIDES
    #box_sel  = dict(left=0.80, right=0.95, bottom=0.13, top=0.95)

    gs_traj = gridspec.GridSpec(n_vars, 2, width_ratios=[2, 1], wspace=0.05, **box_traj)
    #gs_sel  = gridspec.GridSpec(1, 1, **box_sel)
    
    ### trajectories
    
    ax = [[plt.subplot(gs_traj[i, 0]), plt.subplot(gs_traj[i, 1])] for i in range(n_vars)]
    
    legendprops = { 'loc' : 4, 'frameon' : False, 'scatterpoints' : 1, 'handletextpad' : 0.1,
                    'prop' : {'size' : SIZELABEL}, 'ncol' : 1 }
    lineprops   = { 'lw' : SIZELINE, 'linestyle' : '-', 'alpha' : 0.5 }
    #fillprops   = { 'lw' : 0, 'alpha' : 0.3, 'interpolate' : True }
    
    pprops = { 'xticks' : [],
               'yticks' : [],
               'hide'   : ['top', 'bottom', 'left', 'right'],
               'theme'  : 'open' }
    
    t_max, xticks, xticklabels = 0, [], []
    
    if time_set=='early':
        t_max = 600
        xticks      = [      61,      153,      245,      336,      426,      518, 600]
        xticklabels = ['Mar 20', 'Jun 20', 'Sep 20', 'Dec 20', 'Mar 21', 'Jun 21',  '']

    elif time_set=='full':
        t_max = 900
        xticks      = [      61,      153,      245,      336,      426,      518,       609,      700,      790,      882]
#        xticklabels = ['Mar 20', 'Jun 20', 'Sep 20', 'Dec 20', 'Mar 21', 'Jun 21',  'Sep 21', 'Dec 21', 'Mar 22', 'Jun 22']
        xticklabels = ['Mar 20',       '', 'Sep 20',       '', 'Mar 21',       '',  'Sep 21',       '', 'Mar 22',       '']

    elif time_set=='late':
        t_max = 900
        xticks      = [     245,      336,      426,      518,       609,      700,      790,      882]
        xticklabels = ['Sep 20', 'Dec 20', 'Mar 21', 'Jun 21',  'Sep 21', 'Dec 21', 'Mar 22', 'Jun 22']

    elif time_set=='current':
        t_max = 1550
        xticks      = [     245,      426,       609,      791,      975,     1156,     1340,     1493]
        xticklabels = ['Sep 20', 'Mar 21',  'Sep 21', 'Mar 22', 'Sep 22', 'Mar 23', 'Sep 23', 'Feb 24']
        
    # elif time_set=='current':
    #     t_max = 900
    #     xticks      = [      336,     426,      518,       609,      700,      790,      882]
    #     xticklabels = ['Dec 20', 'Mar 21', 'Jun 21',  'Sep 21', 'Dec 21', 'Mar 22',       '']
    
    for i in range(n_vars):
        idx = list(s_sort)[i]
        pprops['colors'] = [local_colors[i]]
        pprops['xlim']   = [  np.min(xticks), t_max]
        pprops['ylim']   = [-0.05,  1.05]
        if (i==n_vars-1):
            pprops['xticks']      = xticks
            pprops['xticklabels'] = xticklabels
            pprops['hide']        = ['top', 'left', 'right']
            pprops['axoffset']    = 0.18
        df_temp = df_traj[df_traj.variant_names==df_sel.iloc[idx].variant_names]
        for k in range(len(df_temp)):
            entry = df_temp.iloc[k]
            times = np.array(str(entry.times).split(), float)
            freqs = np.array(str(entry.frequencies).split(), float)
            if k==len(df_temp)-1:
                mp.plot(type='line', ax=ax[i][0], x=[times], y=[freqs], plotprops=lineprops, **pprops)
            else:
                mp.line(             ax=ax[i][0], x=[times], y=[freqs], plotprops=lineprops, **pprops)
            
        txtprops = dict(ha='left', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
        if variant_list[i]=='B.1':
            ax[i][0].text(101, 0.1, variant_names[i], **txtprops)
        else:
            ax[i][0].text(np.min(xticks), 0.8, variant_names[i], **txtprops)
    
    txtprops = dict(ha='center', va='center', rotation=90, color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, clip_on=False)
    ax[-1][0].text(box_traj['left']-0.04, box_traj['top']-(box_traj['top']-box_traj['bottom'])/2, 'Frequency', transform=fig.transFigure, **txtprops)
    
    for tick in ax[-1][0].get_xticklabels():
        tick.set_rotation(90)
            
    ### selection coefficient estimates
    
    dashlineprops = dict(lw=SIZELINE, ls=':', alpha=0.25, clip_on=False)
    sprops = dict(lw=0, s=9., marker='o')
    
    pprops = { 'yticks': [],
               'xticks': [],
               'hide'  : ['top', 'bottom', 'left', 'right'],
               'theme' : 'open' }
    
#    xticks      = [0, 0.2, 0.4, 0.6, 0.8]
#    xticklabels = [0, 20, 40, 60, 80]
#    xticks      = [0, 0.25, 0.5, 0.75, 1.0]
#    xticklabels = [0, 25, 50, 75, 100]
#    xticks      = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
#    xticklabels = [0,  50, 100, 150, 200, 250, 300]
    xticks      = [  0, 0.5,  1.0, 1.5,   2.0, 2.5,   3.0, 3.5]
    xticklabels = ['0',  '', '100', '', '200',  '', '300',  '']
    for i in range(n_vars):
        idx = list(s_sort)[i]
        pprops['xlim'] = [   0, np.max(xticks)]
        pprops['ylim'] = [ 0.5, 1.5]
        ydat           = [1]
        xdat           = [df_sel.iloc[idx].selection_coefficient]
        xerr           = [df_sel.iloc[idx]['theoretical_error']]
        if variant_list[i]=='omicron':
            omicron_vars  = ['omicron', 'BA.2', 'BA.4', 'BA.5']
            omicron_names = [   'BA.1', 'BA.2', 'BA.4', 'BA.5']
            position      = [     1.25,   0.70,   1.25,   0.40]
            df_temp = df_var[df_var.variant_names.isin(omicron_vars)]
            xdat = np.array(df_temp.selection_coefficient)
            xerr = np.array(df_temp['theoretical_error'])
            ydat = [1 for ii in range(len(df_temp))]
            txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, clip_on=False)
            for df_iter, df_entry in df_temp.iterrows():
                temp_idx = omicron_vars.index(str(df_entry.variant_names))
                name = omicron_names[temp_idx]
                xx = df_entry.selection_coefficient
                yy = position[temp_idx]
                ax[i][1].text(xx, yy, name, **txtprops)
        
        # Print total selection coefficient / number of mutations
        print('%s\t%.3f' % (variant_names[i], xdat[0]/len(df_sel.iloc[idx]['nucleotides'].split('/'))))

        if (i==n_vars-1):
            pprops['xticks']      = xticks
            pprops['xticklabels'] = xticklabels
            pprops['xlabel']      = 'Inferred selection\ncoefficient, ' + r'$\hat{w}$' + ' (%)'
            pprops['hide']        = ['top', 'left', 'right']
            pprops['axoffset']    = 0.3
        mp.plot(type='error', ax=ax[i][1], x=[xdat], y=[ydat], xerr=[xerr], colors=[local_colors[i]], **pprops)
        
    #plt.xticks(rotation = 90)
        
    for i in range(len(xticks)):
        xdat = [xticks[i], xticks[i]]
        ydat = [0.5, 1.25*n_vars]
        # if n_vars>7:
        #     ydat[1] += 0.75
        mp.line(ax=ax[-1][1], x=[xdat], y=[ydat], colors=[BKCOLOR], plotprops=dashlineprops)

    # SAVE FIGURE

    plt.savefig('%s/%s%s' % (FIG_DIR, filename, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    

def plot_frequency_selection(alpha_file,   alpha_file_subdate,   alpha_start,   alpha_end,
                             delta_file,   delta_file_subdate,   delta_start,   delta_end,
                             omicron_file, omicron_file_subdate, omicron_start, omicron_end):
    """ Plot selection over time for Alpha, Delta, and Omicron variants by submission/collection date. """
    
    # Load data
    df_alpha       = pd.read_csv(alpha_file, comment='#', memory_map=True)
    df_alpha       = df_alpha[(df_alpha['time']>=alpha_start) & (df_alpha['time']<=alpha_end)]
    df_alpha_sub   = pd.read_csv(alpha_file_subdate, comment='#', memory_map=True)
    df_alpha_sub   = df_alpha_sub[(df_alpha_sub['time']>=alpha_start) & (df_alpha_sub['time']<=alpha_end)]
    df_delta       = pd.read_csv(delta_file, comment='#', memory_map=True)
    df_delta       = df_delta[(df_delta['time']>=delta_start) & (df_delta['time']<=delta_end)]
    df_delta_sub   = pd.read_csv(delta_file_subdate, comment='#', memory_map=True)
    df_delta_sub   = df_delta_sub[(df_delta_sub['time']>=delta_start) & (df_delta_sub['time']<=delta_end)]
    df_omicron     = pd.read_csv(omicron_file, comment='#', memory_map=True)
    df_omicron     = df_omicron[(df_omicron['time']>=omicron_start) & (df_omicron['time']<=omicron_end)]
    df_omicron_sub = pd.read_csv(omicron_file_subdate, comment='#', memory_map=True)
    df_omicron_sub = df_omicron_sub[(df_omicron_sub['time']>=omicron_start) & (df_omicron_sub['time']<=omicron_end)]
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = 1.5 * w / 3.8 # 3.8 # 4
    fig   = plt.figure(figsize=(w, goldh))

    left_bd   = 0.09
    right_bd  = 0.98
    h_space   = 0.10
    box_h     = (right_bd - left_bd - 2*h_space)/3
    top_bd    = 0.90
    bottom_bd = 0.13 # 0.30 # 0.18
    v_space   = 0.08
    box_v     = (top_bd - bottom_bd - 1*v_space)/2
    
    box_sel_a  = dict(left=left_bd + 0*box_h + 0*h_space, right=left_bd + 1*box_h + 0*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_traj_a = dict(left=left_bd + 0*box_h + 0*h_space, right=left_bd + 1*box_h + 0*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    box_sel_d  = dict(left=left_bd + 1*box_h + 1*h_space, right=left_bd + 2*box_h + 1*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_traj_d = dict(left=left_bd + 1*box_h + 1*h_space, right=left_bd + 2*box_h + 1*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    box_sel_o  = dict(left=left_bd + 2*box_h + 2*h_space, right=left_bd + 3*box_h + 2*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_traj_o = dict(left=left_bd + 2*box_h + 2*h_space, right=left_bd + 3*box_h + 2*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    
    # a/b/c -- Alpha/Delta/Omicron frequency and inferred selection
    
    gs_sel_a = gridspec.GridSpec(1, 1, **box_sel_a)
    ax_sel_a = plt.subplot(gs_sel_a[0, 0])
    gs_sel_d = gridspec.GridSpec(1, 1, **box_sel_d)
    ax_sel_d = plt.subplot(gs_sel_d[0, 0])
    gs_sel_o = gridspec.GridSpec(1, 1, **box_sel_o)
    ax_sel_o = plt.subplot(gs_sel_o[0, 0])
    
    gs_traj_a = gridspec.GridSpec(1, 1, **box_traj_a)
    ax_traj_a = plt.subplot(gs_traj_a[0, 0])
    gs_traj_d = gridspec.GridSpec(1, 1, **box_traj_d)
    ax_traj_d = plt.subplot(gs_traj_d[0, 0])
    gs_traj_o = gridspec.GridSpec(1, 1, **box_traj_o)
    ax_traj_o = plt.subplot(gs_traj_o[0, 0])
    
    ylima   = ylimd   = ylimo   = [0, 1.20]
    yticka  = ytickd  = yticko  = [0, 0.40, 0.80, 1.20]
    ytickla = ytickld = yticklo = [int(i) for i in 100*np.array(yticka)]
    ymticka = ymtickd = ymticko = [0.20, 0.60, 1.00]
    
    lineprops  = dict(lw=2*SIZELINE, ls='-', alpha=1)
    blineprops = dict(lw=2*SIZELINE, ls='--', alpha=1)
    dashprops  = dict(lw=2*SIZELINE, ls=':', alpha=0.5)
    
    for ax_sel, ax_traj, temp_df, temp_df_2, ylim, yticks, yticklabels, yminorticks, v_color, v_name, v_tag in [
        [ax_sel_a, ax_traj_a, df_alpha,   df_alpha_sub,   ylima, yticka, ytickla, ymticka, C_ALPHA,   'Alpha',   'in Great Britain'],
        [ax_sel_d, ax_traj_d, df_delta,   df_delta_sub,   ylimd, ytickd, ytickld, ymtickd, C_DELTA,   'Delta',   'in Great Britain'],
        [ax_sel_o, ax_traj_o, df_omicron, df_omicron_sub, ylimo, yticko, yticklo, ymticko, C_OMICRON, 'Omicron', 'in South Africa'] ]:
        
        # gather data
        
        times   = np.array(temp_df['time'])
        freqs   = np.array(temp_df['frequency'])
        sels    = np.array(temp_df['selection'])
        times_2 = np.array(temp_df_2['time'])
        sels_2  = np.array(temp_df_2['selection'])
        
        xticks      = np.arange(np.min(times), np.max(times), 14)
        xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
        yearlabel   = xticklabels[0].strftime('%Y')
        xticklabels = [i.strftime('%-d %b') for i in xticklabels]
        
        ## a/b/c.2 -- inferred selection coefficients

        tag = v_tag

        pprops = { 'xlim':        [np.min(times), np.max(times)],
                   'xticks':      xticks,
                   'xticklabels': ['' for xtl in xticklabels],
                   'ylim':        [yy for yy in ylim],
                   'yticks':      yticks,
                   'yticklabels': yticklabels,
                   'yminorticks': yminorticks,
                   'xlabel':      '',
                   'ylabel':      ('Inferred selection coefficient\nfor novel %s mutations\n%s, ' % (v_name, tag)) + r'$\hat{w}$' + ' (%)',
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times]
        ydat = [sels]
        mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[v_color], plotprops=lineprops, **pprops)
        
        xdat = [times_2]
        ydat = [sels_2]
        mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[v_color], plotprops=blineprops, **pprops)

        # xdat = [[t_det, t_det]]
        # ydat = [[0, 1.2]]
        # mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        tdx = -5 if v_name=='Omicron' else 0

        txtprops = dict(ha='right', va='bottom', color=v_color, family=FONTFAMILY, size=SIZELABEL, weight='bold')
        ax_sel.text(xticks[-1]+tdx, 0.01, '%s' % v_name, **txtprops)
        
        ## a/b/c.2 -- frequency trajectories
        
        tag = v_tag
        # if v_name != 'Alpha':
        #     tag = '\n' + tag
        
        pprops = { 'xlim':        [np.min(times), np.max(times)],
                   'xticks':      xticks,
                   'xticklabels': xticklabels,
                   'ylim':        [0, 1.05],
                   'yticks':      [0, 0.5, 1],
                   'yticklabels': [0, 50, 100],
                   'yminorticks': [0.25, 0.75],
                   'xlabel':      yearlabel,
                   'ylabel':      '%s frequency\n%s (%%)' % (v_name, tag),
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times]
        ydat = [freqs]
        mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=[LCOLOR], plotprops=lineprops, **pprops)

        # xdat = [[t_det, t_det]]
        # ydat = [[0, 1]]
        # mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
    ## legends and labels
    
    lineprops['clip_on']  = False
    blineprops['clip_on'] = False
    
    invt = ax_sel_a.transData.inverted()
    xy1  = invt.transform((   0, 0))
    xy2  = invt.transform((7.50, 9))
    xy3  = invt.transform((3.00, 9))
    xy4  = invt.transform((5.25, 9))

    legend_dx1   = (xy1[0]-xy2[0])*2.2
    legend_dx2   = (xy1[0]-xy3[0])
    legend_dx    = (xy1[0]-xy4[0])
    legend_dy    = (xy1[1]-xy2[1])
    times        = np.array(df_alpha['time'])
    legend_x     = np.min(times) + 7
    legend_y     = [1.20, 1.20 + legend_dy]
    legend_t     = ['by collection date', 'by submission date']
    legend_props = [lineprops, blineprops]
    for k in range(len(legend_y)):
        yy = legend_y[k]
        mp.line(ax=ax_sel_a, x=[[legend_x+legend_dx2, legend_x+legend_dx1]], y=[[yy, yy]], colors=[LCOLOR], plotprops=legend_props[k], **pprops)
        ax_sel_a.text(legend_x, yy, legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    dx = -0.06
    dy =  0.05
    ax_sel_a.text(box_sel_a['left'] + dx, box_sel_a['top'] + dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sel_d.text(box_sel_d['left'] + dx, box_sel_d['top'] + dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sel_o.text(box_sel_o['left'] + dx, box_sel_o['top'] + dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/fig-4%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)

    
def plot_early_selection(null_file, selection_file, variant_list=[], variant_name=[], variant_start=[], variant_end=[], filename='fig-early-selection'):
    """ Plot trajectory of inferred selection coefficients for listed variants in all regions, highlighting the first region where
        selection passed the threshold value. """
    
    # Load data
    inferred_null = np.loadtxt(null_file)
    df_s          = pd.read_csv(selection_file, comment='#', memory_map=True)
    df_var        = []
    
    # Null distribution statistics
    max_null    = np.amax(inferred_null)
    min_null    = np.amin(inferred_null)
    num_largest = int(round(len(inferred_null) * 2.5 / 100))    # the number of coefficients in the top 2.5% of the null distribution
    soft_cutoff = np.sort(inferred_null)[::-1][num_largest]
    print('hard w cutoff = %.4f' % max_null)
    print('soft w cutoff = %.4f\n' % soft_cutoff)
    
    # Variant detection times
    t_detect = []
    r_detect = []
    
    print('first detection events\nvariant\ttime\tregion')
    for i in range(len(variant_list)):
        df_temp = df_s[(df_s['variant']==variant_list[i]) & (df_s['selection']>=max_null)]
        if len(df_temp)==0:
            print('no detection events for %s!' % (variant_list[i]))
            t_detect.append[0]
        else:
            first_idx = np.argmin(np.array(df_temp['time']))
            t_detect.append(int(df_temp.iloc[first_idx]['time']))
            r_detect.append(df_temp.iloc[first_idx]['region'])
            print('%s\t%d\t%s' % (variant_list[i], df_temp.iloc[first_idx]['time'], df_temp.iloc[first_idx]['region']))
        df_var.append(df_s[(df_s['variant']==variant_list[i]) & (df_s['time']>=variant_start[i]) & (df_s['time']<=variant_end[i])])
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 2
    fig   = plt.figure(figsize=(w, goldh))

    left_bd   = 0.08
    right_bd  = 0.97
    h_space   = 0.10
    box_h     = (right_bd - left_bd - 2*h_space)/3
    top_bd    = 0.94
    bottom_bd = 0.13
    v_space   = 0.10
    box_v     = (top_bd - bottom_bd - 2.25*v_space)/3
    
    box_traj  = [dict(left=left_bd + k*box_h + k*h_space, right=left_bd + (k+1)*box_h + k*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
                 for k in range(3)]
    box_sel   = [dict(left=left_bd + k*box_h + k*h_space, right=left_bd + (k+1)*box_h + k*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
                 for k in range(3)]
    box_null  = dict(left=left_bd + 0*box_h + 0*h_space, right=left_bd + 1.5*box_h + 1*h_space, top=top_bd - 2*box_v - 2.25*v_space, bottom=top_bd - 3*box_v - 2.25*v_space)
    
    ## d -- null distribution of selection coefficients

    gs_null = gridspec.GridSpec(1, 1, **box_null)
    ax_null = plt.subplot(gs_null[0, 0])
    
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=1, edgecolor='none',
                     orientation='vertical', log=True)
                     
#    pprops = { 'xlim':        [-0.04, 0.045],
#               'xticks':      [-0.04, -0.02, 0., 0.02, 0.04, 0.045],
#               'xticklabels': [r'$-4$', r'$-2$', r'$0$', r'$2$', r'$4$', ''],
#               'ylim':        [1e0, 1e6],
#               'yticks':      [1e0, 1e2, 1e4, 1e6],
#               #'yticklabels': [0, 10, 20, 30, 40, 50],
#               'logy':        True,
#               'ylabel':      'Counts',
#               'xlabel':      'Inferred selection coefficients for neutral variants, ' + r'$\hat{w}$' + ' (%)',
#               'bins':        np.arange(-0.04, 0.045+0.001, 0.001),
#               'weights':     [np.ones_like(inferred_null)],
#               'plotprops':   histprops,
#               'axoffset':    0.1,
#               'theme':       'open' }

    pprops = { 'xlim':        [-0.10, 0.15],
               'xticks':      [-0.10, -0.05, 0., 0.05, 0.10, 0.15],
               'xticklabels': [r'$-10$', r'$-5$', r'$0$', r'$5$', r'$10$', r'$15$'],
               'ylim':        [1e0, 1e6],
               'yticks':      [1e0, 1e2, 1e4, 1e6],
               #'yticklabels': [0, 10, 20, 30, 40, 50],
               'logy':        True,
               'ylabel':      'Counts',
               'xlabel':      'Inferred regional selection coefficients for variants\nwith global selection coefficients inferred <10%, ' + r'$\hat{w}$' + ' (%)',
               'bins':        np.arange(-0.10, 0.15+0.0025, 0.0025),
               'weights':     [np.ones_like(inferred_null)],
               'plotprops':   histprops,
               'axoffset':    0.1,
               'theme':       'open' }

    x = [inferred_null]
    mp.plot(type='hist', ax=ax_null, x=x, colors=[LCOLOR], **pprops)
    
    # a/b/c -- Alpha/Delta/Omicron frequency and inferred selection
    
    gs_traj = [gridspec.GridSpec(1, 1, **box_traj[k]) for k in range(3)]
    ax_traj = [plt.subplot(gs_traj[k][0, 0]) for k in range(3)]

    ylabel = 'Regional\nfrequency (%)'
    
    gs_sel = [gridspec.GridSpec(1, 1, **box_sel[k]) for k in range(3)]
    ax_sel = [plt.subplot(gs_sel[k][0, 0]) for k in range(3)]
    
    ylim        = [0, 0.80]
    yticks      = [0, 0.40, 0.80]
    yticklabels = [int(i) for i in 100*np.array(yticks)]
    yminorticks = [0.20, 0.60]
    
    for k in range(3):
    
        # get color
        v_color = BKCOLOR
        if variant_list[k]=='alpha':
            v_color = C_ALPHA
        elif variant_list[k]=='delta':
            v_color = C_DELTA
        elif variant_list[k]=='omicron':
            v_color = C_OMICRON
    
        ## a/b/c.1 -- frequency trajectory
        
        times = []
        freqs = []
        sels  = []
        
        times_d = []
        freqs_d = []
        sels_d  = []
        
        regions = np.unique(df_var[k]['region'])
        for region in regions:
            temp_df = df_var[k][df_var[k]['region']==region]
            if region==r_detect[k]:
                times_d.append(temp_df['time'])
                #freqs_d.append(temp_df['frequency'])
                sels_d.append(temp_df['selection'])
            else:
                times.append(temp_df['time'])
                #freqs.append(temp_df['frequency'])
                sels.append(temp_df['selection'])
        
        lineprops   = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 1.0 }
        bglineprops = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 0.3 }
        dashprops   = {'lw': 2*SIZELINE, 'ls': ':', 'alpha': 0.5 }
        
        xticks      = np.arange(variant_start[k], variant_end[k], 14)
        xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
        yearlabel   = xticklabels[0].strftime('%Y')
        xticklabels = [i.strftime('%-d %b') for i in xticklabels]
        #xticklabels = [f'{i.month}/{i.day}' for i in xticklabels]
        
        pprops = { 'xlim':        [variant_start[k], variant_end[k]],
                   'xticks':      xticks,
                   'xticklabels': ['' for xtl in xticklabels],
                   'ylim':        [0, 1.05],
                   'yticks':      [0, 0.5, 1],
                   'yticklabels': [0, 50, 100],
                   'yminorticks': [0.25, 0.75],
                   'xlabel':      '',
                   'ylabel':      ylabel,
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        #mp.line(ax=ax_traj[k], x=times,   y=freqs,   colors=[v_color for i in range(len(times))],   plotprops=bglineprops, **pprops)
        #mp.line(ax=ax_traj[k], x=times_d, y=freqs_d, colors=[v_color for i in range(len(times_d))], plotprops=lineprops,   **pprops)
        
#        tobs = 0
#        for i in range(len(ydat[0])):
#            if ydat[0][i]>0:
#                tobs = xdat[0][i]
#                break
#        print('First observed at %d, detected selection at %d\n' % (tobs, t_det))

        xdat = [[t_detect[k], t_detect[k]]]
        ydat = [[0, 1]]
        mp.plot(type='line', ax=ax_traj[k], x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        ## a/b/c.2 -- inferred selection coefficient

        pprops = { 'xlim':        [variant_start[k], variant_end[k]],
                   'xticks':      xticks,
                   'xticklabels': xticklabels,
                   'ylim':        ylim,
                   'yticks':      yticks,
                   'yticklabels': yticklabels,
                   'yminorticks': yminorticks,
                   'xlabel':      yearlabel,
                   'ylabel':      'Inferred selection coefficient\nfor novel SNVs, ' + r'$\hat{w}$' + ' (%)',
#                   'ylabel':      'Inferred selection coefficient\nfor novel %s SNVs, ' % v_name + r'$\hat{w}$' + ' (%)',
#                   'ylabel':      'Inferred selection\ncoefficient for novel\n%s SNVs, ' % v_name + r'$\hat{w}$' + ' (%)',
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        mp.line(ax=ax_sel[k], x=times,   y=sels,   colors=[LCOLOR for i in range(len(times))],    plotprops=bglineprops, **pprops)
        mp.line(ax=ax_sel[k], x=times_d, y=sels_d, colors=[v_color for i in range(len(times_d))], plotprops=lineprops,   **pprops)

        xdat = [[t_detect[k], t_detect[k]]]
        ydat = [[0, 1]]
        mp.plot(type='line', ax=ax_sel[k], x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        if variant_name[k]=='Alpha':
            t_det2string = (dt.timedelta(days=t_detect[k]) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_traj[k].text(t_detect[k]+1.5, 0.98, 'Variant first inferred\nto be significantly\nbeneficial\n%s' % t_det2string, **txtprops)
        else:
            t_det2string = (dt.timedelta(days=t_detect[k]) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_traj[k].text(t_detect[k]+1.5, 0.98, '%s' % t_det2string, **txtprops)
            
        txtprops = dict(ha='right', va='bottom', color=v_color, family=FONTFAMILY, size=SIZELABEL, weight='bold')
        ax_sel[k].text(xticks[-1], 0.01, '%s' % variant_name[k], **txtprops)

    ## legends and labels
    
    sublabels = ['a', 'b', 'c']
    dx = -0.06
    dy =  0.01
    ax_null.text(  box_null['left']   + dx, box_null['top']   + dy, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    for k in range(3):
        ax_traj[k].text(box_traj[k]['left'] + dx, box_traj[k]['top'] + dy, sublabels[k].lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/%s%s' % (FIG_DIR, filename, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    
    
def plot_early_selection_roc(selection_file, variant_list=[], variant_name=[], variant_start=[], variant_end=[], time_set='late', filename='fig-early-selection-roc'):
    """ Plot trajectory of inferred selection coefficients for listed variants in all regions, highlighting the first region where
        selection passed the threshold value. """
    
    # Load data
    df_s   = pd.read_csv(selection_file, comment='#', memory_map=True)
    df_var = []
    
    # Variant detection times
    s_start  = 0.005
    s_end    = 0.15
    s_step   = 0.005
    s_cutoff = np.arange(s_start, s_end+s_step, s_step)
    fpr      = 1 - (s_cutoff/s_end)   # later replace this with a function that computes the fraction of predictions that are false
    
    t_detect_v = []
    t_max      = 9999
    for i in range(len(variant_list)):
        df_var        = df_s[df_s['variant']==variant_list[i]]
        temp_t_detect = []
        for s in s_cutoff:
            df_temp = df_var[df_var['selection']>=s]
            if len(df_temp)==0:
                temp_t_detect.append(t_max)
            else:
                temp_t_detect.append(np.min(np.array(df_temp['time'])))
        t_detect_v.append(temp_t_detect)
        
    # shift detection times relative to a norm
    s_norm = 0.10
    s_idx  = list(s_cutoff).index(s_norm)
    
    if time_set=='relative':
        for i in range(len(t_detect_v)):
            t_norm        = t_detect_v[i][s_idx]
            t_detect_v[i] = np.array(t_detect_v[i])-t_norm
        
    # Variant colors
    v_colors = []
    for k in range(len(variant_list)):
        if variant_list[k]=='alpha':
            v_colors.append(C_ALPHA)
        elif variant_list[k]=='delta':
            v_colors.append(C_DELTA)
        elif variant_list[k]=='omicron':
            v_colors.append(C_OMICRON)
        else:
            v_colors.append(BKCOLOR)
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w
    fig   = plt.figure(figsize=(w, goldh))

    box_roc = dict(left=0.15, right=0.95, bottom=0.15, top=0.95)
    gs_roc  = gridspec.GridSpec(1, 1, wspace=0.05, **box_roc)
    
    ### ROC-style detection curves
    
    ax_roc = plt.subplot(gs_roc[0, 0])
    
    legendprops = { 'loc' : 4, 'frameon' : False, 'scatterpoints' : 1, 'handletextpad' : 0.1,
                    'prop' : {'size' : SIZELABEL}, 'ncol' : 1 }
    lineprops   = { 'lw' : 2*SIZELINE, 'linestyle' : '-', 'alpha' : 1.0 }
    
    t_max, xticks, xticklabels = 0, [], []
    xlabel = ''
    
    if time_set=='early':
        t_max       = 600
        xticks      = [      61,      153,      245,      336,      426,      518, 600]
        xticklabels = ['Mar 20', 'Jun 20', 'Sep 20', 'Dec 20', 'Mar 21', 'Jun 21',  '']

    elif time_set=='full':
        t_max       = 900
        xticks      = [      61,      153,      245,      336,      426,      518,       609,      700,      790,      882]
#        xticklabels = ['Mar 20', 'Jun 20', 'Sep 20', 'Dec 20', 'Mar 21', 'Jun 21',  'Sep 21', 'Dec 21', 'Mar 22', 'Jun 22']
        xticklabels = ['Mar 20',       '', 'Sep 20',       '', 'Mar 21',       '',  'Sep 21',       '', 'Mar 22',       '']
    
    elif time_set=='late':
        t_max       = 900
        xticks      = [     245,      336,      426,      518,       609,      700,      790,      882]
        xticklabels = ['Sep 20', 'Dec 20', 'Mar 21', 'Jun 21',  'Sep 21', 'Dec 21', 'Mar 22', 'Jun 22']
        
    elif time_set=='vlate':
        t_max       = 900
        xticks      = [      336,     426,      518,       609,      700,      790,      882]
        xticklabels = ['Dec 20', 'Mar 21', 'Jun 21',  'Sep 21', 'Dec 21', 'Mar 22',       '']
        
    elif time_set=='relative':
        t_min       = -30
        t_max       =  30
        xticks      = [-30, -20, -10, 0, 10, 20, 30]
        xticklabels = xticks
        xlabel      = 'Relative detection time (days)'
        
    pprops = { 'xlim':        [np.min(xticks), np.max(xticks)],
               'ylim':        [-0.01, 1.01],
               'xticks':      xticks,
               'xticklabels': xticklabels,
               'xlabel':      xlabel,
               'yticks':      [0, 0.5,   1],
               'yticklabels': [0,  50, 100],
               'yminorticks': [0.25, 0.75],
               'ylabel':      'False positive rate (%)',
               'colors':      v_colors,
               'hide':        ['top', 'right'],
               'theme':       'open' }
    
    mp.plot(type='line', ax=ax_roc, x=t_detect_v, y=[fpr for i in range(len(variant_list))], plotprops=lineprops, **pprops)
            
#    txtprops = dict(ha='left', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
#    if variant_list[i]=='B.1':
#        ax[i][0].text(101, 0.1, variant_names[i], **txtprops)
#    else:
#        ax[i][0].text(np.min(xticks), 0.8, variant_names[i], **txtprops)
    
#    txtprops = dict(ha='center', va='center', rotation=90, color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, clip_on=False)
#    ax[-1][0].text(box_traj['left']-0.04, box_traj['top']-(box_traj['top']-box_traj['bottom'])/2, 'Frequency', transform=fig.transFigure, **txtprops)
    
#    for tick in ax[-1][0].get_xticklabels():
#        tick.set_rotation(90)

    # SAVE FIGURE

    plt.savefig('%s/%s%s' % (FIG_DIR, filename, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    
    
def plot_a222v(trajectory_file, alpha_file, eu1_file, variant_id, selection_file):
    """ Map out mutations in the major variants, their aggregate inferred selection coefficients, and
        frequency trajectories across regions. """
        
    # Load data
    df_traj   = pd.read_csv(trajectory_file, comment='#', memory_map=True)
    df_sel    = pd.read_csv(selection_file, comment='#', memory_map=True)
    df_traj_a = pd.read_csv(alpha_file, comment='#', memory_map=True)
    df_traj_e = pd.read_csv(eu1_file, comment='#', memory_map=True)
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.2
    fig   = plt.figure(figsize=(w, goldh))

    scale   = 0.83 / (3 * 0.20 + 0.04)
    box_h   = 0.20 * scale
    box_dh  = 0.04 * scale
    box_top = 0.95

    box_comp = dict(left=0.15, right=0.95, top=box_top - 0*box_h - 0*box_dh, bottom=box_top - 1*box_h - 0*box_dh)
    box_sel  = dict(left=0.15, right=0.95, top=box_top - 1*box_h - 1*box_dh, bottom=box_top - 2*box_h - 1*box_dh)
    box_traj = dict(left=0.15, right=0.95, top=box_top - 2*box_h - 2*box_dh, bottom=box_top - 3*box_h - 2*box_dh)
    
    xticks      = [      61,      153,      245,      336,      426,      518, 600]
    xticklabels = ['Mar 20', 'Jun 20', 'Sep 20', 'Dec 20', 'Mar 21', 'Jun 21',  '']
    
    ## a -- 20E (EU1) vs. Alpha trajectories
    
    gs_comp = gridspec.GridSpec(1, 1, **box_comp)
    ax_comp = plt.subplot(gs_comp[0, 0])
    
    legendprops = { 'loc': 4, 'frameon': False, 'scatterpoints': 1, 'handletextpad': 0.1,
                    'prop': {'size' : SIZELABEL}, 'ncol' : 1 }
    lineprops   = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 1.0 }
    
    t_max  = 600
    pprops = { 'colors':   [C_A222V],
               'xlim':     [61, t_max],
               'xticks':   [],
               'ylim':     [-0.05,  1.05],
               'yticks':   [0, 1],
               'ylabel':   'Frequency in Great Britain',
               'axoffset': 0.1,
               'hide':     ['bottom'],
               'theme':    'open' }
    
    mp.line(ax=ax_comp, x=[df_traj_e.time], y=[df_traj_e.frequency], plotprops=lineprops, **pprops)
        
    pprops['colors'] = [C_ALPHA]
    mp.plot(type='line', ax=ax_comp, x=[df_traj_a.time], y=[df_traj_a.frequency], plotprops=lineprops, **pprops)
    
    ## b -- A222V trajectories
    
    gs_traj = gridspec.GridSpec(1, 1, **box_traj)
    ax_traj = plt.subplot(gs_traj[0, 0])
    
    legendprops = { 'loc': 4, 'frameon': False, 'scatterpoints': 1, 'handletextpad': 0.1,
                    'prop': {'size' : SIZELABEL}, 'ncol' : 1 }
    lineprops   = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.5 }
    
    t_max  = 600
    pprops = { 'colors':      [C_A222V],
               'xlim':        [61, t_max],
               'xticks':      xticks,
               'xticklabels': xticklabels,
               'ylim':        [-0.05,  1.05],
               'yticks':      [0, 1],
               'ylabel':      'S:A22V frequency\nacross regions',
               'axoffset':    0.1,
               #'hide':        ['bottom'],
               'theme':       'open' }
    
    df_temp = df_traj[df_traj.variant_names==variant_id]
    for k in range(len(df_temp)):
        entry = df_temp.iloc[k]
        times = np.array(entry.times.split(), float)
        freqs = np.array(entry.frequencies.split(), float)
        if k==len(df_temp)-1:
            mp.plot(type='line', ax=ax_traj, x=[times], y=[freqs], plotprops=lineprops, **pprops)
        else:
            mp.line(             ax=ax_traj, x=[times], y=[freqs], plotprops=lineprops, **pprops)
    
    ## c -- inferred selection coefficient

    gs_sel = gridspec.GridSpec(1, 1, **box_sel)
    ax_sel = plt.subplot(gs_sel[0, 0])
    
    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 1 }
    
    pprops = { 'xlim':        [61, t_max],
               'xticks':      [],
               'ylim':        [0, 0.04],
               'yticks':      [0, 0.02, 0.04],
               'yticklabels': [0, 2, 4],
               #'xlabel':      'Date',
               'ylabel':      'S:A222V selection\ncoefficient in Great Britain, ' + r'$\hat{s}$' + ' (%)',
               'axoffset':    0.1,
               'hide':        ['bottom'],
               'theme':       'open' }
    
    xdat = [np.array(df_sel['time'])]
    ydat = [np.array(df_sel['selection'])]
    mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[LCOLOR], plotprops=lineprops, **pprops)
    
    # legend and sublabels
    
    dx = -0.11
    dy =  0.01
    
    ax_comp.text(box_comp['left'] + dx, box_comp['top'] + dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_traj.text(box_traj['left'] + dx, box_traj['top'] + dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sel.text( box_sel['left']  + dx, box_sel['top']  + dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    colors    = [C_A222V, C_ALPHA]
    plotprops = DEF_ERRORPROPS.copy()
    plotprops['clip_on'] = False
    
    invt = ax_comp.transData.inverted()
    xy1  = invt.transform((   0, 0))
    xy2  = invt.transform((7.50, 9))
    xy3  = invt.transform((3.00, 9))
    xy4  = invt.transform((5.25, 9))

    legend_dx1 = xy1[0]-xy2[0]
    legend_dx2 = xy1[0]-xy3[0]
    legend_dx  = xy1[0]-xy4[0]
    legend_dy  = xy1[1]-xy2[1]
    legend_x   = 85
    legend_y   = [0.9, 0.9 + legend_dy]
    legend_t   = ['20E (EU1)', 'Alpha']
    for k in range(len(legend_y)):
        mp.error(ax=ax_comp, x=[[legend_x+legend_dx]], y=[[legend_y[k]]], edgecolor=[colors[k]], facecolor=[colors[k]], plotprops=plotprops, **pprops)
        ax_comp.text(legend_x, legend_y[k], legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/fig-4%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    
    
def plot_a222v_slides(trajectory_file, alpha_file, eu1_file, variant_id, selection_file):
    """ Map out mutations in the major variants, their aggregate inferred selection coefficients, and
        frequency trajectories across regions. """
        
    # Load data
    df_traj   = pd.read_csv(trajectory_file, comment='#', memory_map=True)
    df_sel    = pd.read_csv(selection_file, comment='#', memory_map=True)
    df_traj_a = pd.read_csv(alpha_file, comment='#', memory_map=True)
    df_traj_e = pd.read_csv(eu1_file, comment='#', memory_map=True)
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH
    goldh = w / 1.2
    fig   = plt.figure(figsize=(w, goldh))

    scale   = 0.83 / (3 * 0.20 + 0.04)
    box_h   = 0.20 * scale
    box_dh  = 0.04 * scale
    box_top = 0.95

    box_comp = dict(left=0.15, right=0.95, top=box_top - 0*box_h - 0*box_dh, bottom=box_top - 1*box_h - 0*box_dh)
    box_sel  = dict(left=0.15, right=0.95, top=box_top - 1*box_h - 1*box_dh, bottom=box_top - 2*box_h - 1*box_dh)
    box_traj = dict(left=0.15, right=0.95, top=box_top - 2*box_h - 2*box_dh, bottom=box_top - 3*box_h - 2*box_dh)
    
    xticks      = [      61,      153,      245,      336,      426,      518, 600]
    xticklabels = ['Mar 20', 'Jun 20', 'Sep 20', 'Dec 20', 'Mar 21', 'Jun 21',  '']
    
    ## a -- 20E (EU1) vs. Alpha trajectories
    
    gs_comp = gridspec.GridSpec(1, 1, **box_comp)
    ax_comp = plt.subplot(gs_comp[0, 0])
    
    legendprops = { 'loc': 4, 'frameon': False, 'scatterpoints': 1, 'handletextpad': 0.1,
                    'prop': {'size' : SIZELABEL}, 'ncol' : 1 }
    lineprops   = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 1.0 }
    
    t_max  = 600
    pprops = { 'colors':   [C_A222V],
               'xlim':     [61, t_max],
               'xticks':   [],
               'ylim':     [-0.05,  1.05],
               'yticks':   [0, 1],
               'ylabel':   'Frequency in\nGreat Britain',
               'axoffset': 0.1,
               'hide':     ['bottom'],
               'theme':    'open' }
    
    mp.line(ax=ax_comp, x=[df_traj_e.time], y=[df_traj_e.frequency], plotprops=lineprops, **pprops)
        
    pprops['colors'] = [C_ALPHA]
    mp.plot(type='line', ax=ax_comp, x=[df_traj_a.time], y=[df_traj_a.frequency], plotprops=lineprops, **pprops)
    
    ## b -- A222V trajectories
    
    gs_traj = gridspec.GridSpec(1, 1, **box_traj)
    ax_traj = plt.subplot(gs_traj[0, 0])
    
    legendprops = { 'loc': 4, 'frameon': False, 'scatterpoints': 1, 'handletextpad': 0.1,
                    'prop': {'size' : SIZELABEL}, 'ncol' : 1 }
    lineprops   = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.5 }
    
    t_max  = 600
    pprops = { 'colors':      [C_A222V],
               'xlim':        [61, t_max],
               'xticks':      xticks,
               'xticklabels': xticklabels,
               'ylim':        [-0.05,  1.05],
               'yticks':      [0, 1],
               'ylabel':      'S:A22V frequency\nacross regions',
               'axoffset':    0.1,
               #'hide':        ['bottom'],
               'theme':       'open' }
    
    df_temp = df_traj[df_traj.variant_names==variant_id]
    for k in range(len(df_temp)):
        entry = df_temp.iloc[k]
        times = np.array(entry.times.split(), float)
        freqs = np.array(entry.frequencies.split(), float)
        if k==len(df_temp)-1:
            mp.plot(type='line', ax=ax_traj, x=[times], y=[freqs], plotprops=lineprops, **pprops)
        else:
            mp.line(             ax=ax_traj, x=[times], y=[freqs], plotprops=lineprops, **pprops)
    
    ## c -- inferred selection coefficient

    gs_sel = gridspec.GridSpec(1, 1, **box_sel)
    ax_sel = plt.subplot(gs_sel[0, 0])
    
    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 1 }
    
    pprops = { 'xlim':        [61, t_max],
               'xticks':      [],
               'ylim':        [0, 0.04],
               'yticks':      [0, 0.02, 0.04],
               'yticklabels': [0, 2, 4],
               #'xlabel':      'Date',
               'ylabel':      'S:A222V selection\ncoefficient in\nGreat Britain, ' + r'$\hat{s}$' + ' (%)',
               'axoffset':    0.1,
               'hide':        ['bottom'],
               'theme':       'open' }
    
    xdat = [np.array(df_sel['time'])]
    ydat = [np.array(df_sel['selection'])]
    mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[LCOLOR], plotprops=lineprops, **pprops)
    
    # legend and sublabels
    
    dx = -0.11
    dy =  0.01
    
#    ax_comp.text(box_comp['left'] + dx, box_comp['top'] + dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
#    ax_traj.text(box_traj['left'] + dx, box_traj['top'] + dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
#    ax_sel.text( box_sel['left']  + dx, box_sel['top']  + dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    colors    = [C_A222V, C_ALPHA]
    plotprops = DEF_ERRORPROPS.copy()
    plotprops['clip_on'] = False
    
    invt = ax_comp.transData.inverted()
    xy1  = invt.transform((   0, 0))
    xy2  = invt.transform((7.50, 9))
    xy3  = invt.transform((3.00, 9))
    xy4  = invt.transform((5.25, 9))

    legend_dx1 = xy1[0]-xy2[0]
    legend_dx2 = xy1[0]-xy3[0]
    legend_dx  = xy1[0]-xy4[0]
    legend_dy  = xy1[1]-xy2[1]
    legend_x   = 85
    legend_y   = [0.9, 0.9 + legend_dy]
    legend_t   = ['20E (EU1)', 'Alpha']
    labelprops = dict(family=FONTFAMILY, size=SIZELABEL, clip_on=False)

    ax_comp.text(125, 0.25, legend_t[0], color=colors[0], ha='left', va='center', **labelprops)
    ax_comp.text(525, 0.35, legend_t[1], color=colors[1], ha='left', va='center', **labelprops)

    # SAVE FIGURE

    plt.savefig('%s/fig-a222v-slides%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_early_detection_sub(alpha_file_subdate,   alpha_start,   alpha_end,
                             delta_file_subdate,   delta_start,   delta_end,
                             omicron_file_subdate, omicron_start, omicron_end, s_cutoff=0.185):
    """ Plot early detection times for Alpha, Delta, and Omicron variants by submission date. """
    
    # Load data
    df_alpha_sub   = pd.read_csv(alpha_file_subdate, comment='#', memory_map=True)
    df_alpha_sub   = df_alpha_sub[(df_alpha_sub['time']>=alpha_start) & (df_alpha_sub['time']<=alpha_end)]
    df_delta_sub   = pd.read_csv(delta_file_subdate, comment='#', memory_map=True)
    df_delta_sub   = df_delta_sub[(df_delta_sub['time']>=delta_start) & (df_delta_sub['time']<=delta_end)]
    df_omicron_sub = pd.read_csv(omicron_file_subdate, comment='#', memory_map=True)
    df_omicron_sub = df_omicron_sub[(df_omicron_sub['time']>=omicron_start) & (df_omicron_sub['time']<=omicron_end)]
    
    # Get early detection times
    t_alpha   = df_alpha_sub[df_alpha_sub['selection']>s_cutoff].iloc[0]['time']
    #f_alpha   = df_alpha_sub[df_alpha_sub['selection']>s_cutoff].iloc[0]['frequency']
    t_delta   = df_delta_sub[df_delta_sub['selection']>s_cutoff].iloc[0]['time']
    #f_delta   = df_delta_sub[df_delta_sub['selection']>s_cutoff].iloc[0]['frequency']
    t_omicron = df_omicron_sub[df_omicron_sub['selection']>s_cutoff].iloc[0]['time']
    #f_omicron = df_omicron_sub[df_omicron_sub['selection']>s_cutoff].iloc[0]['frequency']
    
    print('by submission date')
    print('cutoff\tstatistic\talpha\tdelta\tomicron')
    print('%.2f\ttime\t\t%d\t%d\t%d'       % (s_cutoff, t_alpha, t_delta, t_omicron))
    #print('%.2f\tfreq\t\t%.3f\t%.3f\t%.3f' % (s_cutoff, f_alpha, f_delta, f_omicron))
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 3.8 # 3.8 # 4
    fig   = plt.figure(figsize=(w, goldh))

    left_bd   = 0.09
    right_bd  = 0.98
    h_space   = 0.10
    box_h     = (right_bd - left_bd - 2*h_space)/3
    top_bd    = 0.94
    bottom_bd = 0.18 # 0.30 # 0.18
    v_space   = 0.10
    box_v     = (top_bd - bottom_bd - 0*v_space)/1
    
    box_sel_a = dict(left=left_bd + 0*box_h + 0*h_space, right=left_bd + 1*box_h + 0*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_sel_d = dict(left=left_bd + 1*box_h + 1*h_space, right=left_bd + 2*box_h + 1*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_sel_o = dict(left=left_bd + 2*box_h + 2*h_space, right=left_bd + 3*box_h + 2*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    
    # a/b/c -- Alpha/Delta/Omicron frequency and inferred selection
    
    gs_sel_a = gridspec.GridSpec(1, 1, **box_sel_a)
    ax_sel_a = plt.subplot(gs_sel_a[0, 0])
    gs_sel_d = gridspec.GridSpec(1, 1, **box_sel_d)
    ax_sel_d = plt.subplot(gs_sel_d[0, 0])
    gs_sel_o = gridspec.GridSpec(1, 1, **box_sel_o)
    ax_sel_o = plt.subplot(gs_sel_o[0, 0])
    
    ylima   = ylimd   = ylimo   = [0, 1.20]
    yticka  = ytickd  = yticko  = [0, 0.40, 0.80, 1.20]
    ytickla = ytickld = yticklo = [int(i) for i in 100*np.array(yticka)]
    ymticka = ymtickd = ymticko = [0.20, 0.60, 1.00]
    
    lineprops  = dict(lw=2*SIZELINE, ls='-', alpha=1)
    blineprops = dict(lw=3*SIZELINE, ls='-', alpha=1)
    dashprops  = dict(lw=2*SIZELINE, ls=':', alpha=0.5)
    
    for ax_sel, temp_df_sub, t_det, ylim, yticks, yticklabels, yminorticks, v_color, v_name, v_region in [
        [ax_sel_a,   df_alpha_sub,   t_alpha, ylima, yticka, ytickla, ymticka, C_ALPHA,   'Alpha',   'Great Britain'],
        [ax_sel_d,   df_delta_sub,   t_delta, ylimd, ytickd, ytickld, ymtickd, C_DELTA,   'Delta',   'Great Britain'],
        [ax_sel_o, df_omicron_sub, t_omicron, ylimo, yticko, yticklo, ymticko, C_OMICRON, 'Omicron', 'South Africa'] ]:
    
        ## a/b/c.1 -- frequency trajectory -- NOT PLOTTED HERE
        
        times_sub = np.array(temp_df_sub['time'])
        sels_sub  = np.array(temp_df_sub['selection'])
        
        xticks      = np.arange(np.min(times_sub), np.max(times_sub), 14)
        xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
        yearlabel   = xticklabels[0].strftime('%Y')
        xticklabels = [i.strftime('%-d %b') for i in xticklabels]
        
        ## a/b/c.2 -- inferred selection coefficient

        pprops = { 'xlim':        [np.min(times_sub), np.max(times_sub)],
                   'xticks':      xticks,
                   'xticklabels': xticklabels,
                   'ylim':        [yy for yy in ylim],
                   'yticks':      yticks,
                   'yticklabels': yticklabels,
                   'yminorticks': yminorticks,
                   'xlabel':      yearlabel,
                   'ylabel':      ('Inferred selection coefficient\nfor novel %s mutations\nin %s, ' % (v_name, v_region)) + r'$\hat{w}$' + ' (%)',
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times_sub]
        ydat = [sels_sub]
        mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[v_color], plotprops=lineprops, **pprops)

        xdat = [[t_det, t_det]]
        ydat = [[0, 1.2]]
        mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        if v_name=='Alpha':
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_sel.text(t_det+1.5, 1.15, 'Variant first inferred\nto be significantly\nbeneficial\n%s' % t_det2string, **txtprops)
        elif v_name=='Delta':
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_sel.text(t_det+1.5, 1.15, '%s' % t_det2string, **txtprops)
        else:
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_sel.text(t_det+1.5, 0.90, '%s' % t_det2string, **txtprops)
            
        txtprops = dict(ha='right', va='bottom', color=v_color, family=FONTFAMILY, size=SIZELABEL, weight='bold')
        ax_sel.text(xticks[-1], 0.01, '%s' % v_name, **txtprops)

    ## legends and labels
    
#    lineprops['clip_on']  = False
#    blineprops['clip_on'] = False
#
#    invt = ax_sel_a.transData.inverted()
#    xy1  = invt.transform((   0, 0))
#    xy2  = invt.transform((7.50, 9))
#    xy3  = invt.transform((3.00, 9))
#    xy4  = invt.transform((5.25, 9))
#
#    legend_dx1   = (xy1[0]-xy2[0])
#    legend_dx2   = (xy1[0]-xy3[0])
#    legend_dx    = (xy1[0]-xy4[0])
#    legend_dy    = (xy1[1]-xy2[1])
#    times        = np.array(df_alpha_sub['time'])
#    legend_x     = np.min(times) + 3
#    legend_y     = [1.35, 1.35 + legend_dy]
#    legend_t     = ['by submission date', 'by collection date']
#    legend_props = [blineprops, lineprops]
#    for k in range(len(legend_y)):
#        yy = legend_y[k]
#        mp.line(ax=ax_sel_a, x=[[legend_x+legend_dx2, legend_x+legend_dx1]], y=[[yy, yy]], colors=[LCOLOR], plotprops=legend_props[k], **pprops)
#        ax_sel_a.text(legend_x, yy, legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    dx = -0.06
    dy =  0.01
    ax_sel_a.text(box_sel_a['left'] + dx, box_sel_a['top'] + dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sel_d.text(box_sel_d['left'] + dx, box_sel_d['top'] + dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sel_o.text(box_sel_o['left'] + dx, box_sel_o['top'] + dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/fig-4%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    
    
def plot_early_detection_col(alpha_file, alpha_file_2,   alpha_start,   alpha_end,
                             delta_file,   delta_start,   delta_end,
                             omicron_file, omicron_start, omicron_end):
    """ Plot early detection times for Alpha, Delta, and Omicron variants by collection date. """
    
    # Load data
    df_alpha   = pd.read_csv(alpha_file, comment='#', memory_map=True)
    df_alpha   = df_alpha[(df_alpha['time']>=alpha_start) & (df_alpha['time']<=alpha_end)]
    df_alpha_2 = pd.read_csv(alpha_file_2, comment='#', memory_map=True)
    df_alpha_2 = df_alpha_2[(df_alpha_2['time']>=alpha_start) & (df_alpha_2['time']<=alpha_end)]
    df_delta   = pd.read_csv(delta_file, comment='#', memory_map=True)
    df_delta   = df_delta[(df_delta['time']>=delta_start) & (df_delta['time']<=delta_end)]
    df_omicron = pd.read_csv(omicron_file, comment='#', memory_map=True)
    df_omicron = df_omicron[(df_omicron['time']>=omicron_start) & (df_omicron['time']<=omicron_end)]
    
    # Get early detection times
    s_cutoff  = 0.185
    t_alpha   = df_alpha[df_alpha['selection']>s_cutoff].iloc[0]['time']
    f_alpha   = df_alpha[df_alpha['selection']>s_cutoff].iloc[0]['frequency']
    t_alpha_2 = df_alpha_2[df_alpha_2['selection']>s_cutoff].iloc[0]['time']
    f_alpha_2 = df_alpha_2[df_alpha_2['selection']>s_cutoff].iloc[0]['frequency']
    t_delta   = df_delta[df_delta['selection']>s_cutoff].iloc[0]['time']
    f_delta   = df_delta[df_delta['selection']>s_cutoff].iloc[0]['frequency']
    t_omicron = df_omicron[df_omicron['selection']>s_cutoff].iloc[0]['time']
    f_omicron = df_omicron[df_omicron['selection']>s_cutoff].iloc[0]['frequency']
   
    print('by collection time')
    print('cutoff\tstatistic\talpha (gb)\talpha (london)\tdelta\tomicron')
    print('%.2f\ttime\t\t%d\t\t%d\t\t%d\t%d'         % (s_cutoff, t_alpha, t_alpha_2, t_delta, t_omicron))
    print('%.2f\tfreq\t\t%.3f\t\t%.3f\t\t%.3f\t%.3f' % (s_cutoff, f_alpha, f_alpha_2, f_delta, f_omicron))
    print('')
    
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = 1.5 * w / 3.8 # 3.8 # 4
    fig   = plt.figure(figsize=(w, goldh))

    left_bd   = 0.09
    right_bd  = 0.98
    h_space   = 0.10
    box_h     = (right_bd - left_bd - 2*h_space)/3
    top_bd    = 0.90
    bottom_bd = 0.13 # 0.30 # 0.18
    v_space   = 0.08
    box_v     = (top_bd - bottom_bd - 1*v_space)/2
    
    box_sel_a  = dict(left=left_bd + 0*box_h + 0*h_space, right=left_bd + 1*box_h + 0*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_traj_a = dict(left=left_bd + 0*box_h + 0*h_space, right=left_bd + 1*box_h + 0*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    box_sel_d  = dict(left=left_bd + 1*box_h + 1*h_space, right=left_bd + 2*box_h + 1*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_traj_d = dict(left=left_bd + 1*box_h + 1*h_space, right=left_bd + 2*box_h + 1*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    box_sel_o  = dict(left=left_bd + 2*box_h + 2*h_space, right=left_bd + 3*box_h + 2*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_traj_o = dict(left=left_bd + 2*box_h + 2*h_space, right=left_bd + 3*box_h + 2*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    
    # a/b/c -- Alpha/Delta/Omicron frequency and inferred selection
    
    gs_sel_a = gridspec.GridSpec(1, 1, **box_sel_a)
    ax_sel_a = plt.subplot(gs_sel_a[0, 0])
    gs_sel_d = gridspec.GridSpec(1, 1, **box_sel_d)
    ax_sel_d = plt.subplot(gs_sel_d[0, 0])
    gs_sel_o = gridspec.GridSpec(1, 1, **box_sel_o)
    ax_sel_o = plt.subplot(gs_sel_o[0, 0])
    
    gs_traj_a = gridspec.GridSpec(1, 1, **box_traj_a)
    ax_traj_a = plt.subplot(gs_traj_a[0, 0])
    gs_traj_d = gridspec.GridSpec(1, 1, **box_traj_d)
    ax_traj_d = plt.subplot(gs_traj_d[0, 0])
    gs_traj_o = gridspec.GridSpec(1, 1, **box_traj_o)
    ax_traj_o = plt.subplot(gs_traj_o[0, 0])
    
    ylima   = ylimd   = ylimo   = [0, 1.20]
    yticka  = ytickd  = yticko  = [0, 0.40, 0.80, 1.20]
    ytickla = ytickld = yticklo = [int(i) for i in 100*np.array(yticka)]
    ymticka = ymtickd = ymticko = [0.20, 0.60, 1.00]
    
    lineprops  = dict(lw=2*SIZELINE, ls='-', alpha=1)
    blineprops = dict(lw=2*SIZELINE, ls='--', alpha=1)
    dashprops  = dict(lw=2*SIZELINE, ls=':', alpha=0.5)
    
    for ax_sel, ax_traj, temp_df, temp_df_2, t_det, t_det_2, ylim, yticks, yticklabels, yminorticks, v_color, v_name, v_tag in [
        [ax_sel_a, ax_traj_a, df_alpha,   df_alpha_2, t_alpha,   t_alpha_2, ylima, yticka, ytickla, ymticka, C_ALPHA,   'Alpha',   ''],
        [ax_sel_d, ax_traj_d, df_delta,   df_delta,   t_delta,   t_delta,   ylimd, ytickd, ytickld, ymtickd, C_DELTA,   'Delta',   'in Great Britain'],
        [ax_sel_o, ax_traj_o, df_omicron, df_omicron, t_omicron, t_omicron, ylimo, yticko, yticklo, ymticko, C_OMICRON, 'Omicron', 'in South Africa'] ]:
        
        # gather data
        
        times   = np.array(temp_df['time'])
        freqs   = np.array(temp_df['frequency'])
        sels    = np.array(temp_df['selection'])
        times_2 = np.array(temp_df_2['time'])
        freqs_2 = np.array(temp_df_2['frequency'])
        sels_2  = np.array(temp_df_2['selection'])
        
        xticks      = np.arange(np.min(times), np.max(times), 14)
        xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
        yearlabel   = xticklabels[0].strftime('%Y')
        xticklabels = [i.strftime('%-d %b') for i in xticklabels]
        
        ## a/b/c.2 -- inferred selection coefficients

        tag = v_tag
        if v_name=='Alpha':
            tag += ',\n'
        else:
            tag = '\n' + tag + ','

        pprops = { 'xlim':        [np.min(times), np.max(times)],
                   'xticks':      xticks,
                   'xticklabels': ['' for xtl in xticklabels],
                   'ylim':        [yy for yy in ylim],
                   'yticks':      yticks,
                   'yticklabels': yticklabels,
                   'yminorticks': yminorticks,
                   'xlabel':      '',
                   'ylabel':      ('Inferred selection coefficient\nfor novel %s mutations%s ' % (v_name, tag)) + r'$\hat{w}$' + ' (%)',
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times]
        ydat = [sels]
        mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[v_color], plotprops=lineprops, **pprops)
        
        if v_name=='Alpha':
            xdat = [times_2]
            ydat = [sels_2]
            mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[v_color], plotprops=blineprops, **pprops)

        xdat = [[t_det, t_det]]
        ydat = [[0, 1.2]]
        mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        if v_name=='Alpha':
            xdat = [[t_det_2, t_det_2]]
            ydat = [[0, 1.2]]
            mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        if v_name=='Alpha':
            t_det2string = (dt.timedelta(days=t_det_2) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_sel.text(t_det_2+1.5, 1.15, 'Variant first inferred\nto be significantly\nbeneficial\n%s' % t_det2string, **txtprops)
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            ax_sel.text(t_det+1.5, 1.15, '%s' % t_det2string, **txtprops)
        elif v_name=='Delta':
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_sel.text(t_det+1.5, 1.15, '%s' % t_det2string, **txtprops)
        else:
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_sel.text(t_det+1.5, 1.15, '%s' % t_det2string, **txtprops)
            
        txtprops = dict(ha='right', va='bottom', color=v_color, family=FONTFAMILY, size=SIZELABEL, weight='bold')
        ax_sel.text(xticks[-1], 0.01, '%s' % v_name, **txtprops)
        
        ## a/b/c.2 -- frequency trajectories
        
        tag = v_tag
        if v_name != 'Alpha':
            tag = '\n' + tag
        
        pprops = { 'xlim':        [np.min(times), np.max(times)],
                   'xticks':      xticks,
                   'xticklabels': xticklabels,
                   'ylim':        [0, 1.05],
                   'yticks':      [0, 0.5, 1],
                   'yticklabels': [0, 50, 100],
                   'yminorticks': [0.25, 0.75],
                   'xlabel':      yearlabel,
                   'ylabel':      'Frequency%s (%%)' % tag,
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times]
        ydat = [freqs]
        mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[LCOLOR], plotprops=lineprops, **pprops)
        
        if v_name=='Alpha':
            xdat = [times_2]
            ydat = [freqs_2]
            mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[LCOLOR], plotprops=blineprops, **pprops)

        xdat = [[t_det, t_det]]
        ydat = [[0, 1]]
        mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        if v_name=='Alpha':
            xdat = [[t_det_2, t_det_2]]
            ydat = [[0, 1.2]]
            mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)

    ## legends and labels
    
    lineprops['clip_on']  = False
    blineprops['clip_on'] = False
    
    invt = ax_sel_a.transData.inverted()
    xy1  = invt.transform((   0, 0))
    xy2  = invt.transform((7.50, 9))
    xy3  = invt.transform((3.00, 9))
    xy4  = invt.transform((5.25, 9))

    legend_dx1   = (xy1[0]-xy2[0])*2.4
    legend_dx2   = (xy1[0]-xy3[0])
    legend_dx    = (xy1[0]-xy4[0])
    legend_dy    = (xy1[1]-xy2[1])
    times        = np.array(df_alpha['time'])
    legend_x     = np.min(times) + 7
    legend_y     = [1.42, 1.42 + legend_dy]
    legend_t     = ['in London', 'in Great Britain']
    legend_props = [blineprops, lineprops]
    for k in range(len(legend_y)):
        yy = legend_y[k]
        mp.line(ax=ax_sel_a, x=[[legend_x+legend_dx2, legend_x+legend_dx1]], y=[[yy, yy]], colors=[LCOLOR], plotprops=legend_props[k], **pprops)
        ax_sel_a.text(legend_x, yy, legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    dx = -0.06
    dy =  0.05
    ax_sel_a.text(box_sel_a['left'] + dx, box_sel_a['top'] + dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sel_d.text(box_sel_d['left'] + dx, box_sel_d['top'] + dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sel_o.text(box_sel_o['left'] + dx, box_sel_o['top'] + dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/fig-s7-detection%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    
    
def plot_precision(detection_file, fig_title='fig-s6-precision'):
    """ Plot precision of concerning variant detection. """
    
    # Load data
    df = pd.read_csv(detection_file, comment='#', memory_map=True)
    
    # Get thresholds and precision values
    thresholds = np.array(df['threshold'])
    precisions = []
    for i in range(len(thresholds)):
        tp = df[df['threshold']==thresholds[i]].iloc[0]['true_positive']
        fp = df[df['threshold']==thresholds[i]].iloc[0]['false_positive']
        pr = tp/(tp+fp)
        precisions.append(pr)
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SINGLE_COLUMN
    goldh = w / 1.2
    fig   = plt.figure(figsize=(w, goldh))

    left_bd   = 0.15
    right_bd  = 0.95
    top_bd    = 0.95
    bottom_bd = 0.18
    
    box = dict(left=left_bd, right=right_bd, top=top_bd, bottom=bottom_bd)
    
    # threshold vs. precision
    
    gs = gridspec.GridSpec(1, 1, **box)
    ax = plt.subplot(gs[0, 0])
    
    lineprops  = dict(lw=2*SIZELINE, ls='-', alpha=1)
    blineprops = dict(lw=3*SIZELINE, ls='-', alpha=1)
    dashprops  = dict(lw=2*SIZELINE, ls=':', alpha=0.5)
    
    pprops = { 'xlim':        [0, 0.25],
               'xticks':      [0, 0.05, 0.10, 0.15, 0.20, 0.25],
               'xticklabels': [0, 5, 10, 15, 20, 25],
               'ylim':        [0, 1.05],
               'yticks':      [0, 0.25, 0.5, 0.75, 1],
               'yminorticks': [0.125, 0.375, 0.625, 0.875],
               'xlabel':      'Selection coefficient threshold for variant\nto be classified as concerning, ' + r'$\theta$' + ' (%)',
               'ylabel':      'Precision',
               'axoffset':    0.1,
               'theme':       'open' }
    
    xdat = [thresholds]
    ydat = [precisions]
    mp.plot(type='line', ax=ax, x=xdat, y=ydat, colors=[COLOR_1], plotprops=lineprops, **pprops)

    dashlineprops = {'lw': SIZELINE*2.0, 'ls': ':', 'alpha': 0.5, 'color': BKCOLOR}
    hlines = [1.0, 0.75, 0.5, 0.25]
    for hline in hlines:
        ax.hlines(hline, 0, 0.25, **dashlineprops)

    # SAVE FIGURE

    plt.savefig('%s/%s%s' % (FIG_DIR, fig_title, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    
    
def plot_early_detection_nonull(null_file, alpha_file, alpha_start, alpha_end, delta_file, delta_start, delta_end, omicron_file, omicron_start, omicron_end):
    """ Plot the null distribution for mutations inferred to be nearly neutral.
        Plot frequencies and early detection times for Alpha and Delta variants. """
    
    # Load data
    inferred_null = np.loadtxt(null_file)
    df_alpha      = pd.read_csv(alpha_file, comment='#', memory_map=True)
    df_alpha      = df_alpha[(df_alpha['time']>=alpha_start) & (df_alpha['time']<=alpha_end)]
    df_delta      = pd.read_csv(delta_file, comment='#', memory_map=True)
    df_delta      = df_delta[(df_delta['time']>=delta_start) & (df_delta['time']<=delta_end)]
    df_omicron    = pd.read_csv(omicron_file, comment='#', memory_map=True)
    df_omicron    = df_omicron[(df_omicron['time']>=omicron_start) & (df_omicron['time']<=omicron_end)]
    
    # Null distribution statistics
    max_null    = np.amax(inferred_null)
    min_null    = np.amin(inferred_null)
    num_largest = int(round(len(inferred_null) * 2.5 / 100))    # the number of coefficients in the top 2.5% of the null distribution
    soft_cutoff = np.sort(inferred_null)[::-1][num_largest]
    print('hard w cutoff = %.4f' % max_null)
    print('soft w cutoff = %.4f\n' % soft_cutoff)
    
    # Get early detection times
    
    t_alpha        = df_alpha[df_alpha['selection']>max_null].iloc[0]['time']
    f_alpha        = df_alpha[df_alpha['selection']>max_null].iloc[0]['frequency']
    t_soft_alpha   = df_alpha[df_alpha['selection']>soft_cutoff].iloc[0]['time']
    f_soft_alpha   = df_alpha[df_alpha['selection']>soft_cutoff].iloc[0]['frequency']
    t_delta        = df_delta[df_delta['selection']>max_null].iloc[0]['time']
    f_delta        = df_delta[df_delta['selection']>max_null].iloc[0]['frequency']
    t_soft_delta   = df_delta[df_delta['selection']>soft_cutoff].iloc[0]['time']
    f_soft_delta   = df_delta[df_delta['selection']>soft_cutoff].iloc[0]['frequency']
    t_omicron      = df_omicron[df_omicron['selection']>max_null].iloc[0]['time']
    f_omicron      = df_omicron[df_omicron['selection']>max_null].iloc[0]['frequency']
    t_soft_omicron = df_omicron[df_omicron['selection']>soft_cutoff].iloc[0]['time']
    f_soft_omicron = df_omicron[df_omicron['selection']>soft_cutoff].iloc[0]['frequency']
   
    print('time')
    print('cutoff\talpha\tdelta\tomicron')
    print('hard\t%d\t%d\t%d' % (t_alpha, t_delta, t_omicron))
    print('soft\t%d\t%d\t%d' % (t_soft_alpha, t_soft_delta, t_soft_omicron))
    
    print('')
    print('frequency')
    print('cutoff\talpha\tdelta\tomicron')
    print('hard\t%.3f\t%.3f\t%.3f' % (f_alpha, f_delta, f_omicron))
    print('soft\t%.3f\t%.3f\t%.3f' % (f_soft_alpha, f_soft_delta, f_soft_omicron))
    print('')
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 3
    fig   = plt.figure(figsize=(w, goldh))

    left_bd   = 0.08
    right_bd  = 0.97
    h_space   = 0.10
    box_h     = (right_bd - left_bd - 2*h_space)/3
    top_bd    = 0.94
    bottom_bd = 0.13
    v_space   = 0.10
    box_v     = (top_bd - bottom_bd - 1*v_space)/1
    
    box_traj_a = dict(left=left_bd + 0*box_h + 0*h_space, right=left_bd + 1*box_h + 0*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_sel_a  = dict(left=left_bd + 0*box_h + 0*h_space, right=left_bd + 1*box_h + 0*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    box_traj_d = dict(left=left_bd + 1*box_h + 1*h_space, right=left_bd + 2*box_h + 1*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_sel_d  = dict(left=left_bd + 1*box_h + 1*h_space, right=left_bd + 2*box_h + 1*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    box_traj_o = dict(left=left_bd + 2*box_h + 2*h_space, right=left_bd + 3*box_h + 2*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_sel_o  = dict(left=left_bd + 2*box_h + 2*h_space, right=left_bd + 3*box_h + 2*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    
    # a/b/c -- Alpha/Delta/Omicron frequency and inferred selection
    
    gs_traj_a = gridspec.GridSpec(1, 1, **box_traj_a)
    ax_traj_a = plt.subplot(gs_traj_a[0, 0])
    gs_traj_d = gridspec.GridSpec(1, 1, **box_traj_d)
    ax_traj_d = plt.subplot(gs_traj_d[0, 0])
    gs_traj_o = gridspec.GridSpec(1, 1, **box_traj_o)
    ax_traj_o = plt.subplot(gs_traj_o[0, 0])
    
#    ylaba = 'Alpha frequency\nin London (%)'
#    ylabd = 'Delta frequency\nin Great Britain (%)'
#    ylabo = 'Omicron frequency\nin South Africa (%)'

    ylaba = 'Frequency in\nLondon (%)'
    ylabd = 'Frequency in\nGreat Britain (%)'
    ylabo = 'Frequency in\nSouth Africa (%)'
    
    gs_sel_a = gridspec.GridSpec(1, 1, **box_sel_a)
    ax_sel_a = plt.subplot(gs_sel_a[0, 0])
    gs_sel_d = gridspec.GridSpec(1, 1, **box_sel_d)
    ax_sel_d = plt.subplot(gs_sel_d[0, 0])
    gs_sel_o = gridspec.GridSpec(1, 1, **box_sel_o)
    ax_sel_o = plt.subplot(gs_sel_o[0, 0])
    
    ylima   = ylimd   = ylimo   = [0, 0.80]
    yticka  = ytickd  = yticko  = [0, 0.40, 0.80]
    ytickla = ytickld = yticklo = [int(i) for i in 100*np.array(yticka)]
    ymticka = ymtickd = ymticko = [0.20, 0.60]
#    ylimd   = [0, 0.80]
#    ytickd  = [0, 0.30, 0.60]
#    ytickld = [int(i) for i in 100*np.array(ytickd)]
#    ymtickd = [0.15, 0.45]
#    ylimo   = [0, 0.60]
#    yticko  = [0, 0.30, 0.60]
#    yticklo = [int(i) for i in 100*np.array(ytickd)]
#    ymticko = [0.15, 0.45]
    
    for ax_traj, ax_sel, temp_df, t_det, ylabel, ylim, yticks, yticklabels, yminorticks, v_color, v_name in [
        [ax_traj_a, ax_sel_a, df_alpha,   t_alpha,   ylaba, ylima, yticka, ytickla, ymticka, C_ALPHA, 'Alpha'],
        [ax_traj_d, ax_sel_d, df_delta,   t_delta,   ylabd, ylimd, ytickd, ytickld, ymtickd, C_DELTA, 'Delta'],
        [ax_traj_o, ax_sel_o, df_omicron, t_omicron, ylabo, ylimo, yticko, yticklo, ymticko, C_OMICRON, 'Omicron'] ]:
    
        ## a/b/c.1 -- frequency trajectory
        
        times   = np.array(temp_df['time'])
        freqs   = np.array(temp_df['frequency'])
        sels    = np.array(temp_df['selection'])
        
        lineprops = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 1 }
        dashprops = {'lw': 2*SIZELINE, 'ls': ':', 'alpha': 0.5 }
        
        xticks      = np.arange(np.min(times), np.max(times), 14)
        xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
        yearlabel   = xticklabels[0].strftime('%Y')
        xticklabels = [i.strftime('%-d %b') for i in xticklabels]
        #xticklabels = [f'{i.month}/{i.day}' for i in xticklabels]
        
        pprops = { 'xlim':        [np.min(times), np.max(times)],
                   'xticks':      xticks,
                   'xticklabels': ['' for xtl in xticklabels],
                   'ylim':        [0, 1.05],
                   'yticks':      [0, 0.5, 1],
                   'yticklabels': [0, 50, 100],
                   'yminorticks': [0.25, 0.75],
                   'xlabel':      '',
                   'ylabel':      ylabel,
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times]
        ydat = [freqs]
        mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[v_color], plotprops=lineprops, **pprops)
        
        tobs = 0
        for i in range(len(ydat[0])):
            if ydat[0][i]>0:
                tobs = xdat[0][i]
                break
        print('First observed at %d, detected selection at %d\n' % (tobs, t_det))

        xdat = [[t_det, t_det]]
        ydat = [[0, 1]]
        mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        ## a/b/c.2 -- inferred selection coefficient

        pprops = { 'xlim':        [np.min(times), np.max(times)],
                   'xticks':      xticks,
                   'xticklabels': xticklabels,
                   'ylim':        ylim,
                   'yticks':      yticks,
                   'yticklabels': yticklabels,
                   'yminorticks': yminorticks,
                   'xlabel':      yearlabel,
                   'ylabel':      'Inferred selection coefficient\nfor novel SNVs, ' + r'$\hat{w}$' + ' (%)',
#                   'ylabel':      'Inferred selection coefficient\nfor novel %s SNVs, ' % v_name + r'$\hat{w}$' + ' (%)',
#                   'ylabel':      'Inferred selection\ncoefficient for novel\n%s SNVs, ' % v_name + r'$\hat{w}$' + ' (%)',
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times]
        ydat = [sels]
        mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[LCOLOR], plotprops=lineprops, **pprops)

        xdat = [[t_det, t_det]]
        ydat = [[0, 1]]
        mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        if v_name=='Alpha':
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_traj.text(t_det+1.5, 0.98, 'Variant first inferred\nto be significantly\nbeneficial\n%s' % t_det2string, **txtprops)
        else:
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_traj.text(t_det+1.5, 0.98, '%s' % t_det2string, **txtprops)
            
        txtprops = dict(ha='right', va='bottom', color=v_color, family=FONTFAMILY, size=SIZELABEL, weight='bold')
        ax_sel.text(xticks[-1], 0.01, '%s' % v_name, **txtprops)

    ## legends and labels
    
    dx = -0.06
    dy =  0.01
    ax_traj_a.text(box_traj_a['left'] + dx, box_traj_a['top'] + dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_traj_d.text(box_traj_d['left'] + dx, box_traj_d['top'] + dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_traj_o.text(box_traj_o['left'] + dx, box_traj_o['top'] + dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/supplementary-fig-%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    
    
def plot_early_detection_nonull_sonly(null_file, alpha_file, alpha_start, alpha_end, delta_file, delta_start, delta_end, omicron_file, omicron_start, omicron_end):
    """ Plot the null distribution for mutations inferred to be nearly neutral.
        Plot frequencies and early detection times for Alpha and Delta variants. """
    
    # Load data
    inferred_null = np.loadtxt(null_file)
    df_alpha      = pd.read_csv(alpha_file, comment='#', memory_map=True)
    df_alpha      = df_alpha[(df_alpha['time']>=alpha_start) & (df_alpha['time']<=alpha_end)]
    df_delta      = pd.read_csv(delta_file, comment='#', memory_map=True)
    df_delta      = df_delta[(df_delta['time']>=delta_start) & (df_delta['time']<=delta_end)]
    df_omicron    = pd.read_csv(omicron_file, comment='#', memory_map=True)
    df_omicron    = df_omicron[(df_omicron['time']>=omicron_start) & (df_omicron['time']<=omicron_end)]
    
    # Null distribution statistics
    max_null    = np.amax(inferred_null)
    min_null    = np.amin(inferred_null)
    num_largest = int(round(len(inferred_null) * 2.5 / 100))    # the number of coefficients in the top 2.5% of the null distribution
    soft_cutoff = np.sort(inferred_null)[::-1][num_largest]
    print('hard w cutoff = %.4f' % max_null)
    print('soft w cutoff = %.4f\n' % soft_cutoff)
    
    # Get early detection times
    
    t_alpha        = df_alpha[df_alpha['selection']>max_null].iloc[0]['time']
    t_soft_alpha   = df_alpha[df_alpha['selection']>soft_cutoff].iloc[0]['time']
    t_delta        = df_delta[df_delta['selection']>max_null].iloc[0]['time']
    t_soft_delta   = df_delta[df_delta['selection']>soft_cutoff].iloc[0]['time']
    t_omicron      = df_omicron[df_omicron['selection']>max_null].iloc[0]['time']
    t_soft_omicron = df_omicron[df_omicron['selection']>soft_cutoff].iloc[0]['time']
   
    print('time')
    print('cutoff\talpha\tdelta\tomicron')
    print('hard\t%d\t%d\t%d' % (t_alpha, t_delta, t_omicron))
    print('soft\t%d\t%d\t%d' % (t_soft_alpha, t_soft_delta, t_soft_omicron))
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 4
    fig   = plt.figure(figsize=(w, goldh))

    left_bd   = 0.08
    right_bd  = 0.97
    h_space   = 0.10
    box_h     = (right_bd - left_bd - 2*h_space)/3
    top_bd    = 0.94
    bottom_bd = 0.18
    v_space   = 0.10
    box_v     = (top_bd - bottom_bd - 0*v_space)/1
    
    box_sel_a = dict(left=left_bd + 0*box_h + 0*h_space, right=left_bd + 1*box_h + 0*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_sel_d = dict(left=left_bd + 1*box_h + 1*h_space, right=left_bd + 2*box_h + 1*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_sel_o = dict(left=left_bd + 2*box_h + 2*h_space, right=left_bd + 3*box_h + 2*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    
    # a/b/c -- Alpha/Delta/Omicron frequency and inferred selection
    
    gs_sel_a = gridspec.GridSpec(1, 1, **box_sel_a)
    ax_sel_a = plt.subplot(gs_sel_a[0, 0])
    gs_sel_d = gridspec.GridSpec(1, 1, **box_sel_d)
    ax_sel_d = plt.subplot(gs_sel_d[0, 0])
    gs_sel_o = gridspec.GridSpec(1, 1, **box_sel_o)
    ax_sel_o = plt.subplot(gs_sel_o[0, 0])
    
    ylima   = ylimd   = ylimo   = [0, 1.20]
    yticka  = ytickd  = yticko  = [0, 0.40, 0.80, 1.20]
    ytickla = ytickld = yticklo = [int(i) for i in 100*np.array(yticka)]
    ymticka = ymtickd = ymticko = [0.20, 0.60, 1.00]
    
    for ax_sel, temp_df, t_det, ylim, yticks, yticklabels, yminorticks, v_color, v_name in [
        [ax_sel_a, df_alpha,   t_alpha,   ylima, yticka, ytickla, ymticka, C_ALPHA, 'Alpha'],
        [ax_sel_d, df_delta,   t_delta,   ylimd, ytickd, ytickld, ymtickd, C_DELTA, 'Delta'],
        [ax_sel_o, df_omicron, t_omicron, ylimo, yticko, yticklo, ymticko, C_OMICRON, 'Omicron'] ]:
    
        ## a/b/c.1 -- frequency trajectory
        
        times = np.array(temp_df['time'])
        sels  = np.array(temp_df['selection'])
        
        lineprops = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 1 }
        dashprops = {'lw': 2*SIZELINE, 'ls': ':', 'alpha': 0.5 }
        
        xticks      = np.arange(np.min(times), np.max(times), 14)
        xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
        yearlabel   = xticklabels[0].strftime('%Y')
        xticklabels = [i.strftime('%-d %b') for i in xticklabels]
        
        print('Detected selection at %d\n' % (t_det))
        
        ## a/b/c.2 -- inferred selection coefficient

        pprops = { 'xlim':        [np.min(times), np.max(times)],
                   'xticks':      xticks,
                   'xticklabels': xticklabels,
                   'ylim':        ylim,
                   'yticks':      yticks,
                   'yticklabels': yticklabels,
                   'yminorticks': yminorticks,
                   'xlabel':      yearlabel,
                   'ylabel':      'Inferred selection coefficient\nfor novel SNVs, ' + r'$\hat{w}$' + ' (%)',
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times]
        ydat = [sels]
        mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[LCOLOR], plotprops=lineprops, **pprops)

        xdat = [[t_det, t_det]]
        ydat = [[0, 1.2]]
        mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        if v_name=='Alpha':
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_sel.text(t_det+1.5, 1.00, 'Variant first inferred\nto be significantly\nbeneficial\n%s' % t_det2string, **txtprops)
        else:
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_sel.text(t_det+1.5, 1.00, '%s' % t_det2string, **txtprops)
            
        txtprops = dict(ha='right', va='bottom', color=v_color, family=FONTFAMILY, size=SIZELABEL, weight='bold')
        ax_sel.text(xticks[-1], 0.01, '%s' % v_name, **txtprops)

    ## legends and labels
    
    dx = -0.06
    dy =  0.01
    ax_sel_a.text(box_sel_a['left'] + dx, box_sel_a['top'] + dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sel_d.text(box_sel_d['left'] + dx, box_sel_d['top'] + dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sel_o.text(box_sel_o['left'] + dx, box_sel_o['top'] + dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/supplementary-fig-early%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    
    
def plot_circle_formatter(ax, df_sel, label2ddr):
    """
    Create a circle plot showing selection and sequence features.
    """
    
#    # Loading covariances for checking significance
#
#    cov  = np.loadtxt('%s/covariance-%s-poly-seq2state.dat' % (HIV_MPL_DIR, tag))
#    num  = np.loadtxt('%s/numerator-%s-poly-seq2state.dat' % (HIV_MPL_DIR, tag))
#    cinv = np.linalg.inv(cov)
#    ds   = cinv / 1e4
    
    # Check for selection coefficients that are significant: |s| > mult * sigma_s
    
#    # QUICK HACK TO GET RID OF INSERTIONS
#    data_columns = ['nucleotide number']
#    temp_df = (df_sel.drop(data_columns, axis=1)
#         .join(df_sel[data_columns].apply(pd.to_numeric, errors='coerce')))
#
#    temp_df = temp_df[temp_df[data_columns].notnull().all(axis=1)]
#    df_sel  = temp_df

    n_sig = 0

    sig_s          = []
    sig_site_real  = []
    sig_nuc_idx    = []
    start_idx      = np.min(df_sel['index'])
    end_idx        = np.max(df_sel['index'])
    start_full_idx = int(df_sel[df_sel['index']==start_idx].iloc[0]['nucleotide number'])
    start_dx       = start_idx - start_full_idx

    for df_iter, df_entry in df_sel.iterrows():
        s_val   = df_entry['selection coefficient']
        #idx     = (df_entry.polymorphic_index * 4) + NUC.index(df_entry.nucleotide) - 1 # find where this is in the cov matrix
        sigma_s = 1 #np.sqrt(ds[idx][idx])
        if np.fabs(s_val)>S_CUTOFF: #mult*sigma_s:
            n_sig += 1
            sig_s.append(s_val)
            sig_site_real.append(df_entry['index']-start_idx)
            sig_nuc_idx.append(NUC.index(df_entry['nucleotide']))
    
    # Sequence landmarks

    seq_range = [[ 265, 805], [805, 2719], [2719, 8554], [8554, 10054], [10054, 10972], [10972, 11842], [11842, 12091]]
    seq_label = [     'NSP1',      'NSP2',       'NSP3',        'NSP4',         'NSP5',         'NSP6',        'NSP7']
    seq_range = seq_range + [[12091, 12685], [12685, 13024], [13024, 13441], [13441, 16236], [16236, 18039]]
    seq_label = seq_label + [        'NSP8',         'NSP9',        'NSP10',        'NSP12',       'NSP13']
    seq_range = seq_range + [[18039, 19620], [19620, 20658], [20658, 21552], [21562, 25384], [25392, 26220]]
    seq_label = seq_label + [       'NSP14',        'NSP15',        'NSP16',            'S',       'ORF3a']
    seq_range = seq_range + [[26244, 26472], [26522, 27191], [27201, 27387], [27393, 27759], [27755, 27887]]
    seq_label = seq_label + [           'E',            'M',         'ORF6',        'ORF7a',       'ORF7b']
    seq_range = seq_range + [[27893, 28259], [28273, 29533], [29557, 29674]]
    seq_label = seq_label + [        'ORF8',            'N',        'ORF10']
    
    seq_range = np.array(seq_range) + 1

    landmark_start = []
    landmark_end   = []
    landmark_label = []
    
    def get_protein_edges(df, protein, num_idx='index', full_idx='nucleotide number'):
        temp_df      = df[df['protein']==protein]
        min_idx      = np.argmin(temp_df[num_idx])
        max_idx      = np.argmax(temp_df[num_idx])
        min_num_idx  = temp_df.iloc[min_idx][num_idx]
        max_num_idx  = temp_df.iloc[max_idx][num_idx]
        min_full_idx = int(temp_df.iloc[min_idx][full_idx])
        max_full_idx = int(temp_df.iloc[max_idx][full_idx])
        return min_num_idx, min_full_idx, max_num_idx, max_full_idx

    for i in range(len(seq_label)):
#        label_min  = df_sel[df_sel['protein']==seq_label[i]].iloc[0]['index']
#        label_start_idx = df_sel[df_sel['nucleotide number']==str()].iloc[0]['index']
#        label_end_idx   = df_sel[df_sel['nucleotide number']==str(seq_range[i][1])].iloc[0]['index']
        min_num_idx, min_full_idx, max_num_idx, max_full_idx = get_protein_edges(df_sel, seq_label[i])
        prot_start = min_num_idx - (min_full_idx - seq_range[i][0])
        prot_end   = max_num_idx + (seq_range[i][1] - max_full_idx)
        landmark_start.append(prot_start - start_idx)
        landmark_end.append(prot_end - start_idx)
#        landmark_start.append(seq_range[i][0]-start_idx)
#        landmark_end.append(seq_range[i][1]-start_idx)
        landmark_label.append(seq_label[i])
        
    # Internal landmarks
    
#    mark_range = [[21563 + 3*(13-1), 21563 + 3*305], [21563 + 3*(319-1), 21563 + 3*541]]
#    mark_label = [                      'Spike NTD',                        'Spike RBD']
    
    mark_range = [[21563 + 3*(14-1), 21563 + 3*685]]
    mark_label = ['S1 subunit']
    
    mark_start = []
    mark_end   = []
    
    for i in range(len(mark_label)):
#        mark_start_idx = df_sel[df_sel['nucleotide number']==str(mark_range[i][0])].iloc[0]['index']
#        mark_end_idx   = df_sel[df_sel['nucleotide number']==str(mark_range[i][1])].iloc[0]['index']
        min_num_idx, min_full_idx, max_num_idx, max_full_idx = get_protein_edges(df_sel, 'S')
        prot_start = min_num_idx - (min_full_idx - mark_range[i][0])
        prot_end   = max_num_idx + (mark_range[i][1] - max_full_idx)
        mark_start.append(prot_start - start_idx)
        mark_end.append(prot_end - start_idx)
#        mark_start.append(mark_range[i][0]-start_idx)
#        mark_end.append(mark_range[i][1]-start_idx)
        
#    # Epitope labels
#
#    seq_range = epitope_range.copy()
#    seq_label = epitope_label.copy()
#
#    epitope_start = []
#    epitope_end   = []
#    epitope_label = []
#    epitope_sites = []
#    site2epitope  = {}
#
#    idx = int(df_index.iloc[0].HXB2)
#    for i in range(len(df_index)):
#        try:
#            idx = int(df_index.iloc[i].HXB2)
#        except:
#            pass
##        if pd.notnull(df_index.iloc[i].HXB2):
##            idx = df_index.iloc[i].HXB2
#        for j in range(len(seq_range)):
#            if idx==seq_range[j][0] or (i==0 and idx>seq_range[j][0] and idx<seq_range[j][1]):
#                epitope_start.append(i)
#                epitope_end.append(i)
#                epitope_label.append(seq_label[j])
#                if seq_label[j]=='DG9':
#                    epitope_start[-1] -= 9 # account for DEP insertion
#            if idx==seq_range[j][1] or (i==len(df_index)-1 and idx>seq_range[j][0] and int(df_index.iloc[0].HXB2)<=seq_range[j][0]
#                                        and idx<seq_range[j][1]):
#                epitope_end[epitope_label.index(seq_label[j])] = i
#                iix = epitope_label.index(seq_label[j])
#                for k in range(epitope_start[iix],epitope_end[iix]):
#                    epitope_sites.append(k)
#                    if k in site2epitope:
#                        print('Unexpected! Overlapping epitopes.')
#                    else:
#                        site2epitope[k] = seq_label[j]
#                print(''.join(list(df_index.iloc[epitope_start[iix]:epitope_end[iix]].TF)))

#    # Populate links
#
#    inv_cov = []
#    idx_1   = []
#    idx_2   = []
#
#    eidx = -1
#    if cov_label:
#        eidx = epitope_label.index(cov_label)
#    cov_cutoff = 0.007
#
#    for i in range(len(cinv)):
#        for j in range(i+1,len(cinv)):
#            if np.fabs(cinv[i][j])>cov_cutoff and (i//len(NUC) != j//len(NUC)):
#                ii = df_info[df_info.polymorphic_index==i//len(NUC)].iloc[0].alignment_index
#                jj = df_info[df_info.polymorphic_index==j//len(NUC)].iloc[0].alignment_index
#                if eidx==-1 or ((ii>=epitope_start[eidx] and ii<=epitope_end[eidx]) or (jj>=epitope_start[eidx] and jj<=epitope_end[eidx])):
#                    inv_cov.append(cinv[i][j] / cov_cutoff)
#                    idx_1.append(ii)
#                    idx_2.append(jj)

    # Dot plot for significant selection coefficients

    c_dot = {True: C_BEN, False: C_DEL}
    
    level_rad = [0.9, 0.85, 0.80, 0.75, 0.70]

    def_buff = 1000
    def site2angle(site, deg=False, buffer=def_buff):
        if deg: return    -360.*(site+(buffer/2))/(end_idx-start_idx+buffer)
        else:   return -2*np.pi*(site+(buffer/2))/(end_idx-start_idx+buffer)

    dot_colors = [c_dot[s>0]                      for s in sig_s]
    dot_sizes  = [S_MULT * (np.fabs(s) - DS_NORM) for s in sig_s]
    dot_x = [level_rad[sig_nuc_idx[i]] * np.cos(site2angle(sig_site_real[i])) for i in range(len(sig_s))]
    dot_y = [level_rad[sig_nuc_idx[i]] * np.sin(site2angle(sig_site_real[i])) for i in range(len(sig_s))]

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=0)
    nuc_spc = 0 #0.01
    nuc_shift = 0 #2 * nuc_spc 
    for i in range(len(NUC)):
        if i==len(NUC)-1:
            ax.text(level_rad[i]+nuc_shift-(i*nuc_spc), 0, '–', **txtprops)
        else:
            ax.text(level_rad[i]+nuc_shift-(i*nuc_spc), 0, NUC[i], **txtprops)

#    # Lines for polymorphic sites
#
#    line_r = [0.62, 0.66]
#    line_x = [[line_r[0] * np.cos(site2angle(i)), line_r[1] * np.cos(site2angle(i))] for i in np.unique(df_sel['nucleotide number'])]
#    line_y = [[line_r[0] * np.sin(site2angle(i)), line_r[1] * np.sin(site2angle(i))] for i in np.unique(df_sel['nucleotide number'])]
#    line_c = [LCOLOR                                                                 for i in np.unique(df_sel['nucleotide number'])]

    # Arcs for sequence landmarks

    arc_r            = [0.96, 0.99, 1.02]
    arc_dr           = 0.01
    track            = 0
    landmark_patches = []
    landmark_text_d  = { }

    for i in range(len(landmark_label)):
        if landmark_label[i]=='ORF7b':
            track = 1
        else:
            track = 0
            
#        if i>0 and landmark_start[i]<landmark_end[i-1]:
#            track = (track + 1)%len(arc_r)
        
        arc_props = dict(center=[0,0], r=arc_r[track], width=arc_dr, lw=AXWIDTH/2, ec=BKCOLOR, fc='none',
                         theta1=site2angle(landmark_end[i], deg=True), theta2=site2angle(landmark_start[i], deg=True))
        landmark_patches.append(matplotlib.patches.Wedge(**arc_props))

        # label with line
        if True:
            txtprops  = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=0)
            plotprops = dict(lw=AXWIDTH/2, ls='-', clip_on=False)

            label_x = [arc_r[track]*np.cos(site2angle(landmark_start[i]))]
            label_y = [arc_r[track]*np.sin(site2angle(landmark_start[i]))]
            ddr     = 0.05 #0.11
            if landmark_label[i] in label2ddr:
                ddr = label2ddr[landmark_label[i]]

            ddx1 =  ddr * np.cos(site2angle(landmark_start[i]))
            ddy  =  ddr * np.sin(site2angle(landmark_start[i]))
            ddx2 = ddx1 + np.sign(ddx1)*0.03

            label_x = label_x + [label_x[0] + ddx1, label_x[0] + ddx2]
            label_y = label_y + [label_y[0] +  ddy, label_y[0] +  ddy]
            if label_x[0]<0:
                txtprops['ha'] = 'right'
            else:
                txtprops['ha'] = 'left'

            ax.text(label_x[-1], label_y[-1], landmark_label[i], **txtprops)
            mp.line(ax=ax, x=[label_x], y=[label_y], colors=[BKCOLOR], plotprops=plotprops)

        # plot normally
        else:
            delta_site = 500
            if landmark_label[i] in ['vif']:
                delta_site += landmark_text_d[landmark_label[i]] + 25
            txt_dr   = 0.04
            txt_ang  = site2angle(landmark_start[i]+delta_site)
            txt_rot  = site2angle(landmark_start[i]+delta_site, deg=True)-90
            txt_x    = (arc_r[track] + (arc_dr/2) + txt_dr)*np.cos(txt_ang)
            txt_y    = (arc_r[track] + (arc_dr/2) + txt_dr)*np.sin(txt_ang)
            txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY,
                            size=SIZELABEL, rotation=txt_rot)
            ax.text(txt_x, txt_y, landmark_label[i], **txtprops)
            
    # Arcs for internal landmarks

    arc_r        = 0.65
    arc_dr       = 0.01
    track        = 0
    mark_patches = []
    mark_text_d  = { }

    for i in range(len(mark_label)):
        
        arc_props = dict(center=[0,0], r=arc_r, width=arc_dr, lw=AXWIDTH/2, ec=BKCOLOR, fc='none',
                         theta1=site2angle(mark_end[i], deg=True), theta2=site2angle(mark_start[i], deg=True))
        mark_patches.append(matplotlib.patches.Wedge(**arc_props))
        
#        arc_props = dict(xy=(0,0), height=arc_r*2, width=arc_r*2, lw=AXWIDTH, color=BKCOLOR,
#                         theta1=site2angle(mark_end[i], deg=True), theta2=site2angle(mark_start[i], deg=True))
#        mark_patches.append(matplotlib.patches.Arc(**arc_props))
        
        # label with line
        if True:
            txtprops  = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=0)
            plotprops = dict(lw=AXWIDTH/2, ls='-', clip_on=False)

            label_x = [arc_r*np.cos(site2angle(mark_start[i]))]
            label_y = [arc_r*np.sin(site2angle(mark_start[i]))]
            ddr     = -0.11
            if mark_label[i] in label2ddr:
                ddr = label2ddr[mark_label[i]]

            ddx1 =  ddr * np.cos(site2angle(mark_start[i]))
            ddy  =  ddr * np.sin(site2angle(mark_start[i]))
            ddx2 = ddx1 - np.sign(ddx1)*0.03

#            label_x = label_x + [label_x[0] + ddx1, label_x[0] + ddx2]
#            label_y = label_y + [label_y[0] +  ddy, label_y[0] +  ddy]
            label_x = label_x + [label_x[0] + ddx1]
            label_y = label_y + [label_y[0] +  ddy]
            if label_x[0]<0:
                txtprops['ha'] = 'left'
            else:
                txtprops['ha'] = 'right'

            ax.text(label_x[-1] + ddx1/3, label_y[-1] + ddy/3, mark_label[i], **txtprops)
            mp.line(ax=ax, x=[label_x], y=[label_y], colors=[BKCOLOR], plotprops=plotprops)

    # Arcs for TGCA selection tracks
    
    for i in range(len(level_rad)):
        # arc_props = dict(center=[0,0], r=level_rad[i], width=0, lw=AXWIDTH/2, ec=C_NEU, fc='none', alpha=0.5, theta1=0, theta2=360)
        arc_props = dict(center=[0,0], r=level_rad[i], width=0, lw=AXWIDTH/2, ec=C_NEU_LT, fc='none', theta1=0, theta2=360)
        landmark_patches.append(matplotlib.patches.Wedge(**arc_props))

#    # Arcs for epitopes
#
#    arc_r           = 1.04
#    arc_dr          = 0.32
#    epitope_patches = []
#
#    for i in range(len(epitope_label)):
#
#        arc_props = dict(center=[0,0], r=arc_r, width=arc_dr, lw=AXWIDTH/2, ec=BKCOLOR, fc='#f2f2f2', alpha=0.8,
#                         theta1=site2angle(epitope_end[i], deg=True), theta2=site2angle(epitope_start[i], deg=True))
#        epitope_patches.append(matplotlib.patches.Wedge(**arc_props))
#
#        # label epitopes
#        if True:
#            mid       = (site2angle(epitope_end[i], deg=True)+site2angle(epitope_start[i], deg=True))/2.
#            label_x   = [arc_r * np.cos(mid * 2 * np.pi / 360.)]
#            label_y   = [arc_r * np.sin(mid * 2 * np.pi / 360.)]
#            ddr       = 0.06
#            if epitope_label[i] in label2ddr:
#                ddr = label2ddr[epitope_label[i]]
#            ddx1 =  ddr * np.cos(mid * 2 * np.pi / 360.)
#            ddy  =  ddr * np.sin(mid * 2 * np.pi / 360.)
#            ddx2 = ddx1 + np.sign(ddx1)*0.03
#            txtprops  = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=0)
#            plotprops = dict(lw=AXWIDTH/2, ls='-', clip_on=False)
#
#            label_x = label_x + [label_x[0] + ddx1, label_x[0] + ddx2]
#            label_y = label_y + [label_y[0] +  ddy, label_y[0] +  ddy]
#            if label_x[0]<0:
#                txtprops['ha'] = 'right'
#            else:
#                txtprops['ha'] = 'left'
#
#            ax.text(label_x[-1], label_y[-1], epitope_label[i], **txtprops)
#            mp.line(ax=ax, x=[label_x], y=[label_y], colors=[BKCOLOR], plotprops=plotprops)

#    # Arc plot for large values of integrated covariance
#
#    c_pos  = LCOLOR
#    c_neg  = LCOLOR
#    c_circ = { True : c_neg, False : c_pos }
#
#    arc_rad   = 0.65
#    arc_mult  = SIZELINE/2
#    arc_alpha = 0.1
#
#    circ_color = [c_circ[ic>0]                                     for ic in inv_cov]
#    #circ_color = [c_circ[idx_1[i] in epitope_sites or idx_2[i] in epitope_sites] for i in range(len(inv_cov))]
#    circ_rad   = [[arc_rad, arc_rad]                               for ic in inv_cov]
#    circ_arc   = [dict(lw=arc_mult * np.fabs(ic), alpha=arc_alpha) for ic in inv_cov]
#    circ_x     = [(i+(def_buff/2)) for i in idx_1]
#    circ_y     = [(i+(def_buff/2)) for i in idx_2]

    # Make plot

    pprops = { #'colors':   circ_color,
               'xlim':    [-1.05, 1.05],
               'ylim':    [-1.05, 1.05],
               'xticks':  [],
               'yticks':  [],
               #'size':     end_idx-start_idx+def_buff,
               #'bezrad':   0.0,
               #'rad':      circ_rad,
               #'arcprops': circ_arc,
               'hide':    ['left','right', 'top', 'bottom'],
               'noaxes':  True }

    for i in range(len(dot_x)):
        plotprops = dict(lw=0, marker='o', s=dot_sizes[i], zorder=9999)
        if i<len(dot_x)-1:
            mp.scatter(             ax=ax, x=[[dot_x[i]]], y=[[dot_y[i]]], colors=[dot_colors[i]], plotprops=plotprops)
        else:
            mp.plot(type='scatter', ax=ax, x=[[dot_x[i]]], y=[[dot_y[i]]], colors=[dot_colors[i]], plotprops=plotprops, **pprops)

#    for i in range(len(np.unique(df_sel['nucleotide number']))):
#        plotprops = dict(lw=AXWIDTH, ls='-')
#        if i==len(np.unique(df_sel['nucleotide number']))-1:
#            mp.plot(type='line', ax=ax, x=[line_x[i]], y=[line_y[i]], colors=[line_c[i]], plotprops=plotprops, **pprops)
#        else:
#            mp.line(             ax=ax, x=[line_x[i]], y=[line_y[i]], colors=[line_c[i]], plotprops=plotprops)

#    mp.plot(type='circos', ax=ax, x=circ_x, y=circ_y, **pprops)

    for patch in landmark_patches:
        ax.add_artist(patch)
        
    for patch in mark_patches:
        ax.add_artist(patch)

#    for patch in epitope_patches:
#        ax.add_artist(patch)

    return sig_s, sig_site_real, sig_nuc_idx #, epitope_start, epitope_end
    
    
#### FIGURES FOR SLIDES ####

def plot_performance_slides(sim_file, inf_file, replicate_file, pop_size, cutoff=0.01):

    # Load simulation data
    sim_data         = np.load(sim_file, allow_pickle=True)
    simulations      = sim_data['simulations']
    actual           = sim_data['selection_all']
    pop_size         = sim_data['pop_size']
    traj             = sim_data['traj_record']
    mutant_sites     = sim_data['mutant_sites']
    mutant_sites_all = sim_data['mutant_sites_all']

    # Load inference data
    inf_data     = np.load(inf_file, allow_pickle=True)
    inferred     = inf_data['selection']
    error        = inf_data['error_bars']
    inferred_ind = inf_data['selection_ind']
    error_ind    = inf_data['errors_ind']

    # Other parameters
    record = np.ones(simulations)
    pop_size_assumed = pop_size
    
    selection = {}
    for i in range(len(mutant_sites_all)):
        selection[mutant_sites_all[i]] = actual[i]

    # Functions 1
    def trajectory_reshape(traj):
        # reshape trajectories
        traj_reshaped = []
        for i in range(len(traj)):
            T_temp = len(traj[i])
            L_temp = len(traj[i][0])
            traj_temp = np.zeros((T_temp, L_temp))
            for j in range(T_temp):
                traj_temp[j] = traj[i][j]
            traj_reshaped.append(np.array(traj_temp))
        return traj_reshaped
    
    def find_significant_sites(traj, mutant_sites, cutoff):
        # find sites whose frequency is (not?) too small
        allele_number = []
        for i in range(len(traj)):
            for j in range(len(traj[i][0])):
                if np.amax(traj[i][:, j]) > cutoff:
                    allele_number.append(mutant_sites[i][j])
        allele_number = np.sort(np.unique(np.array(allele_number)))
        return allele_number
        
    # reorganize arrays and filter low frequency trajectories
    traj = trajectory_reshape(traj)
    
    allele_number = find_significant_sites(traj, mutant_sites, cutoff)
    actual = np.array([selection[i] for i in allele_number])
    L_traj = len(allele_number)
    if len(actual)!=0:
        s_value = np.amax(np.array(actual))
    else:
        s_value = 0
    
    # Functions 2
    def filter_sites(traj, error, inferred, mutant_sites, mutant_sites_all, allele_number):
        # eliminate sites whose frequency is too small
        traj_new = [[] for i in range(simulations)]
        mutant_sites_new = [[] for i in range(simulations)]
        inferred_new = np.zeros(len(allele_number))
        error_bars = np.zeros(len(allele_number))
        for i in range(len(traj)):
            for j in range(len(traj[i][0])):
                if mutant_sites[i][j] in allele_number:
                    traj_new[i].append(np.array(traj[i][:,j]))
                    mutant_sites_new[i].append(mutant_sites[i][j])
        for i in range(len(allele_number)):
            loc = np.where(allele_number[i] == mutant_sites_all)
            inferred_new[i] = inferred[loc]
            error_bars[i] = error[loc]
        return traj_new, inferred_new, error_bars, mutant_sites_new

    def filter_sites_ind(inferred, error):
        # eliminates sites whose frequency is too small in the individually inferred coefficients
        inferred_new = np.zeros((simulations, L_traj))
        error_new    = np.zeros((simulations, L_traj))
        for sim in range(simulations):
            for i in range(L_traj):
                loc = np.where(mutant_sites_all == allele_number[i])
                inferred_new[sim, i] = inferred[sim][loc]
                error_new[sim, i]    = error[sim][loc]
        return inferred_new, error_new
        
    def filter_sites_infer(traj, inferred, mutant_sites, mutant_sites_all, allele_number):
        # eliminate sites whose frequency is too small
        inferred_new = np.zeros(len(allele_number))
        mutant_sites_new = [[] for i in range(len(traj))]
        for i in range(len(traj)):
            for j in range(len(traj[i][0])):
                if mutant_sites[i][j] in allele_number:
                    mutant_sites_new[i].append(mutant_sites[i][j])
        for i in range(len(allele_number)):
            loc = np.where(allele_number[i] == mutant_sites_all)
            inferred_new[i] = inferred[loc]
        return inferred_new
        
    def separate_sign(selection):
        # finds the indices at which mutations are beneficial, neutral, and deleterious
        splus, sminus, szero = {}, {}, {}
        for i in selection:
            if selection[i] > 0:
                splus[i] = selection[i]
            elif selection[i] < 0:
                sminus[i] = selection[i]
            else:
                szero[i] = selection[i]
        mutant_sites = [i for i in selection]
        ind_plus, ind_minus, ind_zero = [], [], []
        for i in splus:
            ind_plus.append(np.where(i == mutant_sites)[0][0])
        for i in sminus:
            ind_minus.append(np.where(i == mutant_sites)[0][0])
        for i in szero:
            ind_zero.append(np.where(i == mutant_sites)[0][0])
        return ind_minus, ind_zero, ind_plus
        
    def sort_by_sign(selection, inferred):
        # sorts the beneficial, deleterious, and neutral mutations
        ind_minus, ind_zero, ind_plus = separate_sign(selection)
        inf_plus  = [inferred[i] for i in ind_plus]
        inf_minus = [inferred[i] for i in ind_minus]
        inf_zero  = [inferred[i] for i in ind_zero]
        return inf_minus, inf_zero, inf_plus
    
    traj, inferred, error, mutant_sites = filter_sites(traj, error, inferred, mutant_sites, mutant_sites_all, allele_number)
    mutant_sites_all = np.sort(np.unique([mutant_sites[j][i] for j in range(len(mutant_sites)) for i in range(len(mutant_sites[j]))]))
    
    # plot combining the frequencies, the selection coefficients, and the histograms
    indices       = np.argsort(actual)[::-1]
    actual        = np.array(actual)[indices]
    inferred      = inferred[indices]
    colors_inf    = np.array([S2COLOR(i) for i in actual], dtype=str)
    allele_number = allele_number[indices]
    
    # Finding the different sets of actual coefficients that are all the same
    actual_unique  = np.unique(actual)
    unique_lengths = [len(np.isin(actual, i).nonzero()[0]) for i in actual_unique]
    colors_unique  = np.array([S2COLOR(i) for i in actual_unique], dtype=str)
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH
    goldh = w / 1.1
    fig   = plt.figure(figsize=(w, goldh))

    box_traj = dict(left=0.13, right=0.90, bottom=0.60, top=0.95)
    box_hist = dict(left=0.13, right=0.70, bottom=0.12, top=0.45)
    
    ## a -- individual beneficial/neutral/deleterious trajectories and selection coefficients

    ### trajectories
    
    n_exs  = len(inferred)
    colors = [C_BEN, C_BEN, C_NEU, C_NEU, C_DEL, C_DEL]
    
    gs = gridspec.GridSpec(n_exs, 2, width_ratios=[1, 1], wspace=0.05, **box_traj)
    ax = [[plt.subplot(gs[i, 0]), plt.subplot(gs[i, 1])] for i in range(n_exs)]
    
    legendprops = { 'loc' : 4, 'frameon' : False, 'scatterpoints' : 1, 'handletextpad' : 0.1,
                    'prop' : {'size' : SIZELABEL}, 'ncol' : 1 }
    lineprops   = { 'lw' : SIZELINE*1.5, 'linestyle' : '-', 'alpha' : 1.0 }
    fillprops   = { 'lw' : 0, 'alpha' : 0.3, 'interpolate' : True }
    
    pprops = { 'xticks' : [],
               'yticks' : [],
               'hide'   : ['top', 'bottom', 'left', 'right'],
               'theme'  : 'open' }
    
    for i in range(n_exs):
        pprops['colors'] = [colors[i]]
        pprops['xlim']   = [    0,  50]
        pprops['ylim']   = [-0.08, 1.08]
        if (i==n_exs-1):
            pprops['xticks']   = [0, 10, 20, 30, 40, 50]
            pprops['xlabel']   = 'Time (serial intervals)'
            pprops['hide']     = ['top', 'left', 'right']
            pprops['axoffset'] = 0.3
        for k in range(simulations):
            if allele_number[i] in mutant_sites[k]:
                loc = np.where(np.array(mutant_sites[k])==allele_number[i])[0][0]
                ydat = traj[k][loc]
                xdat = np.arange(len(traj[k][loc]))*record[k]
            else:
                ydat = np.zeros(len(traj[0]))
                xdat = np.arange(len(traj[0]))*record[k]
        mp.line(             ax=ax[i][0], x=[xdat], y=[ydat], plotprops=lineprops, **pprops)
        mp.plot(type='fill', ax=ax[i][0], x=[xdat], y=[ydat], plotprops=fillprops, **pprops)
    
    ### selection coefficient estimates
    
    dashlineprops = { 'lw' : SIZELINE * 2.0, 'ls' : ':', 'alpha' : 0.5 }
    sprops = dict(lw=0, s=9., marker='o')
    
    pprops = { 'yticks': [],
               'xticks': [],
               'hide'  : ['top', 'bottom', 'left', 'right'],
               'theme' : 'open' }
    
    for i in range(n_exs):
        pprops['xlim'] = [-0.04, 0.04]
        pprops['ylim'] = [  0.5,  1.5]
        ydat           = [1]
        xdat           = [inferred[i]]
        xerr           = error[i]
        if (i==n_exs-1):
            pprops['xticks']      = [-0.04,   -0.02,      0,   0.02,   0.04]
            pprops['xticklabels'] = [  ' ', r'$-2$', r'$0$', r'$2$', r'$4$']
            pprops['xlabel']      = 'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)'
            pprops['hide']        = ['top', 'left', 'right']
            pprops['axoffset']    = 0.3
        mp.plot(type='error', ax=ax[i][1], x=[xdat], y=[ydat], xerr=[xerr], colors=[colors[i]], **pprops)
        ax[i][1].axvline(x=actual_unique[(-(i//2) + 2)%3], color=BKCOLOR, **dashlineprops)
        
    ### add legend
    
    colors    = [C_BEN, C_NEU, C_DEL]
    colors_lt = [C_BEN_LT, C_NEU_LT, C_DEL_LT]
    plotprops = DEF_ERRORPROPS.copy()
    plotprops['clip_on'] = False
    plotprops['markersize'] = SMALLSIZEDOT/4
    
    invt = ax[-1][-1].transData.inverted()
    xy1  = invt.transform((   0, 0))
    xy2  = invt.transform((7.50, 9))
    xy3  = invt.transform((3.00, 9))
    xy4  = invt.transform((5.25, 9))

    legend_dx1 = 5*(xy1[0]-xy2[0])
    legend_dx2 = 5*(xy1[0]-xy3[0])
    legend_dx  = 5*(xy1[0]-xy4[0])
    legend_dy  = 5*(xy1[1]-xy2[1])
    legend_x   =  0.015
    legend_y   = [-4, -4 + legend_dy, -4 + (2*legend_dy)]
    legend_t   = ['Beneficial', 'Neutral', 'Deleterious']
    for k in range(len(legend_y)):
        mp.error(ax=ax[-1][-1], x=[[legend_x+legend_dx]], y=[[legend_y[k]]], edgecolor=[colors[k]], facecolor=[colors_lt[k]], plotprops=plotprops, **pprops)
        ax[-1][-1].text(legend_x, legend_y[k], legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    yy = -4 + (3.5*legend_dy)
    mp.line(ax=ax[-1][-1], x=[[legend_x+legend_dx2, legend_x+legend_dx1]], y=[[yy, yy]], colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':', clip_on=False), **pprops)
    ax[-1][-1].text(legend_x, yy, 'True selection\ncoefficient', ha='left', va='center', **DEF_LABELPROPS)
    
    yy = -4 + (5.25*legend_dy)
    mp.line(ax=ax[-1][-1], x=[[legend_x+legend_dx2, legend_x+legend_dx1]], y=[[yy, yy]], colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, clip_on=False), **pprops)
    ax[-1][-1].text(legend_x, yy, 'Mean inferred\ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    ### bounding boxes

#    ax[0][0].text(box_traj['left']-0.08, box_traj['top']+0.01, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
#    ax[0][0].text(box_hist['left']-0.08, box_hist['top']+0.03, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    lineprops = { 'lw' : AXWIDTH/2., 'ls' : '-', 'alpha' : 1.0 }
    pprops    = { 'xlim' : [0, 1], 'xticks' : [], 'ylim' : [0, 1], 'yticks' : [],
        'hide' : ['top','bottom','left','right'], 'plotprops' : lineprops }
    txtprops = {'ha' : 'right', 'va' : 'center', 'color' : BKCOLOR, 'family' : FONTFAMILY,
        'size' : SIZELABEL, 'rotation' : 90, 'transform' : fig.transFigure}
    ax[0][0].text(box_traj['left']-0.025, box_traj['top'] - (box_traj['top']-box_traj['bottom'])/2., 'Frequency', **txtprops)

#    boxprops = {'ec' : BKCOLOR, 'lw' : SIZELINE/2., 'fc' : 'none', 'clip_on' : False, 'zorder' : -100}
#
#    dx  = 0.005
#    dxl = 0.000
#    dxr = 0.001
#    dy  = 0.003
#    ll = box_traj['left'] + dxl                     # left box left
#    lb = box_traj['bottom'] - dy                      # left box bottom
#    rl = (box_traj['left'] + box_traj['right'])/2. + dx + dxl # right box left
#    wd = (box_traj['right'] - box_traj['left'])/2. - dxr      # box width
#    ht = (box_traj['top'] - box_traj['bottom']) + (2. * dy)   # box height
#
#    recL = matplotlib.patches.Rectangle(xy=(ll, lb), width=wd, height=ht, transform=fig.transFigure, **boxprops)
#    recL = ax[0][0].add_patch(recL)
#    recR = matplotlib.patches.Rectangle(xy=(rl, lb), width=wd, height=ht, transform=fig.transFigure, **boxprops)
#    recR = ax[0][0].add_patch(recR)

    # b -- distribution of inferred selection coefficients
    
    rep_data = np.load(replicate_file, allow_pickle=True)
    positive, negative, zero = [], [], []
    for sim in range(len(rep_data['inferred'])):
        traj     = trajectory_reshape(rep_data['traj'][sim])
        inferred = rep_data['inferred'][sim]
        actual   = rep_data['actual'][sim]
        mutants  = rep_data['mutant_sites'][sim]
        error    = rep_data['errors'][sim]
        mutant_sites_all = np.sort(np.unique(np.array([mutants[i][j] for i in range(len(mutants))
                                               for j in range(len(mutants[i]))])))
        selection = {}
        for i in range(len(mutant_sites_all)):
            selection[mutant_sites_all[i]] = actual[i]
        allele_number = find_significant_sites(traj, mutants, cutoff)
        L_traj = len(allele_number)
        selection_new = {}
        for i in range(L_traj):
            selection_new[allele_number[i]] = selection[allele_number[i]]
        actual           = [selection_new[i] for i in allele_number]
        mutant_sites_all = np.sort(np.unique(np.array([mutants[i][j] for i in range(len(mutants))
                                               for j in range(len(mutants[i]))])))
        inferred  = filter_sites_infer(traj, inferred, mutants, mutant_sites_all, allele_number)
        inf_minus, inf_zero, inf_plus = sort_by_sign(selection_new, inferred)
        positive += inf_plus
        negative += inf_minus
        zero     += inf_zero
    
    ### plot histogram
    
    gs_hist = gridspec.GridSpec(1, 1, **box_hist)
    ax_hist = plt.subplot(gs_hist[0, 0])

    colors     = [C_DEL, C_NEU, C_BEN]
    tags       = ['deleterious', 'neutral', 'beneficial']
    s_true_loc = [-s_value, 0, s_value]

    dashlineprops  = { 'lw' : SIZELINE * 2.0, 'ls' : ':', 'alpha' : 0.5 }
    solidlineprops = { 'lw' : SIZELINE * 2.0, 'alpha' : 0.5 }
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.7, edgecolor='none',
                     orientation='vertical')
    pprops = { 'xlim':        [-0.10, 0.10],
               'xticks':      [   -0.10,    -0.075,  -0.05,     -0.025,     0.,   0.025,   0.05,    0.075,     0.10],
               #'xticklabels': [r'$-10$', r'$-7.5$', r'$-5$', r'$-2.5$', r'$0$', r'$2.5$', r'$5$', r'$7.5$', r'$10$'],
               'xticklabels': [r'$-10$', '', r'$-5$', '', r'$0$', '', r'$5$', '', r'$10$'],
               'ylim':        [0., 0.075],
               'yticks':      [0., 0.025, 0.05, 0.075],
               #'yticklabels': [r'$0$', r'$2.5$', r'$5$', r'$7.5$'],
               'yticklabels': [r'$0$', '', r'$5$', ''],
               'ylabel':      'Frequency (%)',
               'xlabel':      'Inferred selection coefficient, ' + r'$\hat{s}$' + ' (%)',
               'bins':        np.arange(-0.10, 0.10+0.002, 0.002),
               'combine':     True,
               'plotprops':   histprops,
               'axoffset':    0.1,
               'theme':       'open' }

    s_vals = [negative, zero, positive]
    for i in range(len(tags)):
        x = [s_vals[i]]
        ax_hist.axvline(x=s_true_loc[i], color=colors[i], **dashlineprops)
        if i<len(tags)-1: mp.hist(             ax=ax_hist, x=x, colors=[colors[i]], **pprops)
        else:             mp.plot(type='hist', ax=ax_hist, x=x, colors=[colors[i]], **pprops)
        
    for i in range(len(tags)):
        x = np.mean(s_vals[i])
        ax_hist.axvline(x=x, color=colors[i], **solidlineprops)

    # SAVE FIGURE

    plt.savefig('%s/fig-performance-slides%s' % (FIG_DIR, '.pdf'), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    

def plot_finite_sampling_slides(inf_file, filename='fig-finite-sampling-slides', n_per_group=10):

    # Load inference data
    df_inf = pd.read_csv(inf_file, memory_map=True)
    
    s_sort = np.argsort(df_inf['selection coefficient'])[::-1]
    
    topx_s, topx_ds = [], []
    for i in s_sort[:n_per_group]:
        topx_s.append(df_inf.iloc[i]['selection coefficient'])
        topx_ds.append(df_inf.iloc[i]['standard deviation'])
        
    midx_s, midx_ds = [], []
    mid = len(s_sort)//2
    temp_sort = [s_sort[i] for i in range(mid-(n_per_group//2)+1, mid+(n_per_group//2)+1)]
    for i in temp_sort:
        midx_s.append(df_inf.iloc[i]['selection coefficient'])
        midx_ds.append(df_inf.iloc[i]['standard deviation'])
        
    botx_s, botx_ds = [], []
    for i in s_sort[-n_per_group:]:
        botx_s.append(df_inf.iloc[i]['selection coefficient'])
        botx_ds.append(df_inf.iloc[i]['standard deviation'])
    
    # PLOT FIGURE
    
    ## set up figure grid

    
    w     = SLIDE_WIDTH
    goldh = w / 2.5
    fig   = plt.figure(figsize=(w, goldh))

    box = dict(left=0.10, right=0.98, bottom=0.05, top=0.98)
    
    ## a -- individual beneficial/neutral/deleterious selection coefficients
    
    gs = gridspec.GridSpec(1, 3, wspace=0.05, **box)
    ax = [plt.subplot(gs[0, i]) for i in range(3)]
    
    legendprops = { 'loc' : 4, 'frameon' : False, 'scatterpoints' : 1, 'handletextpad' : 0.1,
                    'prop' : {'size' : SIZELABEL}, 'ncol' : 1 }
    lineprops   = { 'lw' : SIZELINE*1.5, 'linestyle' : '-', 'alpha' : 1.0 }
    fillprops   = { 'lw' : 0, 'alpha' : 0.3, 'interpolate' : True }
    sprops = dict(lw=0, s=9., marker='o')
    
    ### deleterious / neutral / beneficial
    
    colors = [[C_DEL for i in range(n_per_group)],
              [C_NEU for i in range(n_per_group)],
              [C_BEN for i in range(n_per_group)]]
              
    xdat = list(range(n_per_group))
    ydat = [ botx_s[::-1],  midx_s[::-1],  topx_s[::-1]]
    yerr = [botx_ds[::-1], midx_ds[::-1], topx_ds[::-1]]
    
    for k in range(3):
        pprops = { 'colors':      colors[k],
                   'xlim':        [-1, n_per_group],
                   'xticks':      [],
                   'ylim':        [-0.1, 0.17],
                   'yticks':      [],
                   'axoffset':    0.3,
                   'hide':        ['top', 'bottom', 'left', 'right'],
                   'theme':       'open' }
    
        if k==0:
            pprops['hide']        = ['top', 'bottom', 'right']
            pprops['yticks']      = [-0.1, -0.05, 0, 0.05, 0.1, 0.15]
            pprops['yticklabels'] = [ -10,    -5, 0,    5,  10,   15]
            
        mp.plot(type='error', ax=ax[k], x=[xdat], y=[ydat[k]], yerr=[yerr[k]], **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s%s' % (FIG_DIR, filename, '.pdf'), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)

    
def plot_selection_statistics_slides(selection_file_all, selection_file_ns, label2ddr=[]):
    """ Makes a plot containing the distribution of inferred coefficients and the enrichment of nonsynonymous mutations.
        Combines selection_dist_plot and syn_plot functions.
        correlation_plots determines whether or not the 3 plots containing the correlations are made."""
    
    # loading and processing the data
    df_sel_all = pd.read_csv(selection_file_all, comment='#', memory_map=True)
        
    # PLOT FIGURE
    
    ## set up figure grid
    
    #w       = SLIDE_WIDTH
    hshrink = (0.15/0.70) * (0.85/0.60)
    goldh   = w * hshrink
    w       = SLIDE_WIDTH * 0.75
    fig     = plt.figure(figsize=(w, goldh))

    box_hist = dict(left=0.15, right=0.95, bottom=0.30, top=1.0)
    
    ## a -- selection coefficient histogram
    
    ### plot histogram
    
    s_link    = np.array(df_sel_all['selection coefficient'])
    s_pos_log = np.log10(s_link[s_link>0])
    s_neg_log = np.log10(-s_link[s_link<0])
    
    n_hist = 15
    xmin   = -2.5
    xmax   = -0.5
    x_neg  = np.linspace(xmin, xmax, num=n_hist)
    x_pos  = np.linspace(xmin, xmax, num=n_hist)
    y_neg  = []
    y_pos  = []
    y_neu  = []
    for i in range(len(x_neg)-1):
        y_neg.append(np.sum((s_neg_log>x_neg[i]) * (s_neg_log<=x_neg[i+1])))
    y_neg.append(np.sum(s_neg_log>x_neg[-1]))
    y_neu.append(np.sum(s_neg_log<x_neg[0]))
    for i in range(len(x_pos)-1):
        y_pos.append(np.sum((s_pos_log>x_pos[i]) * (s_pos_log<=x_pos[i+1])))
    y_pos.append(np.sum(s_pos_log>x_pos[-1]))
    y_neu.append(np.sum(s_pos_log<x_pos[0]))
    
    norm  = 4
    c_neg = [hls_to_rgb(0.02, 1., 0.83)]
    c_pos = [hls_to_rgb(0.58, 1., 0.60)]
    for i in range(len(y_neg)):
        t = np.min([(x_neg[i] + norm)/norm, 1])
        c_neg.append(hls_to_rgb(0.58, 0.53 * t + 1. * (1 - t), 0.60))
        t = np.min([(x_pos[i] + norm)/norm, 1])
        c_pos.append(hls_to_rgb(0.02, 0.53 * t + 1. * (1 - t), 0.83))
        
    wspace  = 0.20
    gs_hist = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=wspace, **box_hist)
    
    hist_props = dict(lw=SIZELINE/2, width=1.5/n_hist, align='center', edgecolor=[BKCOLOR], orientation='vertical')
    
    ### negative coefficient subplot
    
    ax_neg = plt.subplot(gs_hist[0, 0])
    
    pprops = { 'colors':      c_neg,
               'xlim':        [xmin-4.5/n_hist, -0.5][::-1],
               'xticks':      [-2.5-3/n_hist, -2, -1, -0.5],
               'xticklabels': ['>-0.3', '-1', '-10', ''],
               'logy':        True,
               'ylim':        [1e0, 3e2],
               'yticks':      [1e0, 1e1, 1e2],
               'ylabel':      'Counts',
               'hide':        ['top', 'right'] }
    
    x_neg = [[-2.5-2/n_hist]] + [[i] for i in x_neg]
    y_neg = [[y_neu[0]]] + [[i] for i in y_neg]
    
    mp.plot(type='bar', ax=ax_neg, x=x_neg, y=y_neg, plotprops=hist_props, **pprops)
    
    ### positive coefficient subplot
    
    ax_pos = plt.subplot(gs_hist[0, 1])
    
    pprops = { 'colors':      c_pos,
               'xlim':        [xmin-4.5/n_hist, -0.5],
               'xticks':      [-2.5-3/n_hist, -2, -1, -0.5],
               'xticklabels': ['<0.3', '1', '10', ''],
               'logy':        True,
               'ylim':        [1e0, 3e2],
               'yticks':      [],
               'hide':        ['top', 'right', 'left'] }
               
    x_pos = [[-2.5-2/n_hist]] + [[i] for i in x_pos]
    y_pos = [[y_neu[1]]] + [[i] for i in y_pos]
    
    mp.plot(type='bar', ax=ax_pos, x=x_pos, y=y_pos, plotprops=hist_props, **pprops)

    # MAKE LEGEND
    
    ax_neg.text(box_hist['left'] + (box_hist['right']-box_hist['left'])/2, box_hist['bottom']-0.2, 'Inferred selection coefficients, ' + '$\hat{s}$' + ' (%)',
                ha='center', va='top', transform=fig.transFigure, **DEF_LABELPROPS)
    
    # SAVE FIGURE

    plt.savefig('%s/fig-statistics-slides%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    
    
def plot_circle_slides(selection_file, label2ddr=[]):
    """ Ye olde circle plot """
    
    # PLOT FIGURE

    ## set up figure grid

    w     = 8 #3.9 #SINGLE_COLUMN
    goldh = w
    fig   = plt.figure(figsize=(w, goldh))

    d = 0.062 #0.02
    dd = 0.01
    box_circ = dict(left=d-dd, right=1-d-dd, bottom=d, top=1-d)
    gs_circ  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_circ)
    ax_circ  = plt.subplot(gs_circ[0, 0])
    
    ## circle plot

    df_sel = pd.read_csv(selection_file, comment='#', memory_map=True)
    
    sig_s, sig_site_real, sig_nuc_idx = plot_circle_formatter_slides(ax_circ, df_sel, label2ddr)
    
    #ax_circ.text(box_circ['left']-0.02, box_circ['top']+dy, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # MAKE LEGEND

    invt = ax_circ.transData.inverted()
    xy1  = invt.transform((0,0))
    xy2  = invt.transform((0,9))
    
    coef_legend_dx = 0.05 #0.08
    coef_legend_x  = 0 - 5*coef_legend_dx
    coef_legend_y  = 0.1 #0.2
    coef_legend_dy = 5*(xy1[1]-xy2[1]) #3*(xy1[1]-xy2[1])

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
    ex_s     = [-0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05]
    show_s   = [    1,     0,     0,     0,     0, 1,    0,    0,    0,    0,    1]
    c_s      = [C_DEL, C_BEN]
    for i in range(len(ex_s)):
        plotprops = dict(lw=0, marker='o', s=(np.fabs(ex_s[i]) - DS_NORM)*S_MULT if ex_s[i]!=0 else 0, clip_on=False)
        mp.scatter(ax=ax_circ, x=[[coef_legend_x + i*coef_legend_dx]], y=[[coef_legend_y]], colors=[c_s[ex_s[i]>0]], plotprops=plotprops)
        if show_s[i]:
            ax_circ.text(coef_legend_x + i*coef_legend_dx, coef_legend_y + 0.75*coef_legend_dy, '%d' % (100*ex_s[i]), **txtprops)
    ax_circ.text(coef_legend_x + len(ex_s)*coef_legend_dx/2, coef_legend_y + (2.25*coef_legend_dy), 'Inferred selection\ncoefficient (%)', **txtprops)

    # SAVE FIGURE

    plt.savefig('%s/fig-circle%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, transparent=True)
    plt.close(fig)
    
    
def plot_circle_formatter_slides(ax, df_sel, label2ddr):
    """
    Create a circle plot showing selection and sequence features.
    """

    n_sig = 0

    sig_s          = []
    sig_site_real  = []
    sig_nuc_idx    = []
    start_idx      = np.min(df_sel['index'])
    end_idx        = np.max(df_sel['index'])
    start_full_idx = int(df_sel[df_sel['index']==start_idx].iloc[0]['nucleotide number'])
    start_dx       = start_idx - start_full_idx

    for df_iter, df_entry in df_sel.iterrows():
        s_val   = df_entry['selection coefficient']
        #idx     = (df_entry.polymorphic_index * 4) + NUC.index(df_entry.nucleotide) - 1 # find where this is in the cov matrix
        sigma_s = 1 #np.sqrt(ds[idx][idx])
        if np.fabs(s_val)>S_CUTOFF: #mult*sigma_s:
            n_sig += 1
            sig_s.append(s_val)
            sig_site_real.append(df_entry['index']-start_idx)
            sig_nuc_idx.append(NUC.index(df_entry['nucleotide']))
    
    # Sequence landmarks

    seq_range = [[ 265, 805], [805, 2719], [2719, 8554], [8554, 10054], [10054, 10972], [10972, 11842], [11842, 12091]]
    seq_label = [     'NSP1',      'NSP2',       'NSP3',        'NSP4',         'NSP5',         'NSP6',        'NSP7']
    seq_range = seq_range + [[12091, 12685], [12685, 13024], [13024, 13441], [13441, 16236], [16236, 18039]]
    seq_label = seq_label + [        'NSP8',         'NSP9',        'NSP10',        'NSP12',       'NSP13']
    seq_range = seq_range + [[18039, 19620], [19620, 20658], [20658, 21552], [21562, 25384], [25392, 26220]]
    seq_label = seq_label + [       'NSP14',        'NSP15',        'NSP16',            'S',       'ORF3a']
    seq_range = seq_range + [[26244, 26472], [26522, 27191], [27201, 27387], [27393, 27759], [27755, 27887]]
    seq_label = seq_label + [           'E',            'M',         'ORF6',        'ORF7a',       'ORF7b']
    seq_range = seq_range + [[27893, 28259], [28273, 29533], [29557, 29674]]
    seq_label = seq_label + [        'ORF8',            'N',        'ORF10']
    
    seq_range = np.array(seq_range) + 1

    landmark_start = []
    landmark_end   = []
    landmark_label = []
    
    def get_protein_edges(df, protein, num_idx='index', full_idx='nucleotide number'):
        temp_df      = df[df['protein']==protein]
        min_idx      = np.argmin(temp_df[num_idx])
        max_idx      = np.argmax(temp_df[num_idx])
        min_num_idx  = temp_df.iloc[min_idx][num_idx]
        max_num_idx  = temp_df.iloc[max_idx][num_idx]
        min_full_idx = int(temp_df.iloc[min_idx][full_idx])
        max_full_idx = int(temp_df.iloc[max_idx][full_idx])
        return min_num_idx, min_full_idx, max_num_idx, max_full_idx

    for i in range(len(seq_label)):
#        label_min  = df_sel[df_sel['protein']==seq_label[i]].iloc[0]['index']
#        label_start_idx = df_sel[df_sel['nucleotide number']==str()].iloc[0]['index']
#        label_end_idx   = df_sel[df_sel['nucleotide number']==str(seq_range[i][1])].iloc[0]['index']
        min_num_idx, min_full_idx, max_num_idx, max_full_idx = get_protein_edges(df_sel, seq_label[i])
        prot_start = min_num_idx - (min_full_idx - seq_range[i][0])
        prot_end   = max_num_idx + (seq_range[i][1] - max_full_idx)
        landmark_start.append(prot_start - start_idx)
        landmark_end.append(prot_end - start_idx)
#        landmark_start.append(seq_range[i][0]-start_idx)
#        landmark_end.append(seq_range[i][1]-start_idx)
        landmark_label.append(seq_label[i])
        
    # Internal landmarks
    
#    mark_range = [[21563 + 3*(13-1), 21563 + 3*305], [21563 + 3*(319-1), 21563 + 3*541]]
#    mark_label = [                      'Spike NTD',                        'Spike RBD']
    
    mark_range = [[21563 + 3*(14-1), 21563 + 3*685]]
    mark_label = ['S1 subunit']
    
    mark_start = []
    mark_end   = []
    
    for i in range(len(mark_label)):
#        mark_start_idx = df_sel[df_sel['nucleotide number']==str(mark_range[i][0])].iloc[0]['index']
#        mark_end_idx   = df_sel[df_sel['nucleotide number']==str(mark_range[i][1])].iloc[0]['index']
        min_num_idx, min_full_idx, max_num_idx, max_full_idx = get_protein_edges(df_sel, 'S')
        prot_start = min_num_idx - (min_full_idx - mark_range[i][0])
        prot_end   = max_num_idx + (mark_range[i][1] - max_full_idx)
        mark_start.append(prot_start - start_idx)
        mark_end.append(prot_end - start_idx)
#        mark_start.append(mark_range[i][0]-start_idx)
#        mark_end.append(mark_range[i][1]-start_idx)

    # Dot plot for significant selection coefficients

    c_dot = { True : C_BEN, False : C_DEL }
    
    level_rad = [0.9, 0.85, 0.80, 0.75, 0.70]

    def_buff = 1000
    def site2angle(site, deg=False, buffer=def_buff):
        if deg: return    -360.*(site+(buffer/2))/(end_idx-start_idx+buffer)
        else:   return -2*np.pi*(site+(buffer/2))/(end_idx-start_idx+buffer)

    dot_colors = [c_dot[s>0]                      for s in sig_s]
    dot_sizes  = [S_MULT * (np.fabs(s) - DS_NORM) for s in sig_s]
    dot_x = [level_rad[sig_nuc_idx[i]] * np.cos(site2angle(sig_site_real[i])) for i in range(len(sig_s))]
    dot_y = [level_rad[sig_nuc_idx[i]] * np.sin(site2angle(sig_site_real[i])) for i in range(len(sig_s))]

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=0)
    for i in range(len(NUC)):
        if i==len(NUC)-1:
            ax.text(level_rad[i], 0, '–', **txtprops)
        else:
            ax.text(level_rad[i], 0, NUC[i], **txtprops)

    # Arcs for sequence landmarks

    arc_r            = [0.96, 0.99, 1.02]
    arc_dr           = 0.01
    track            = 0
    landmark_patches = []
    landmark_text_d  = { }
    
    #show_labels = ['NSP6', 'S']

    for i in range(len(landmark_label)):
        if landmark_label[i]=='ORF7b':
            track = 1
        else:
            track = 0
        
        arc_props = dict(center=[0,0], r=arc_r[track], width=arc_dr, lw=AXWIDTH/2, ec=BKCOLOR, fc='none',
                         theta1=site2angle(landmark_end[i], deg=True), theta2=site2angle(landmark_start[i], deg=True))
        landmark_patches.append(matplotlib.patches.Wedge(**arc_props))

        # label with line
        if True: #landmark_label[i] in show_labels:
            txtprops  = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=0)
            plotprops = dict(lw=AXWIDTH/2, ls='-', clip_on=False)

            label_x = [arc_r[track]*np.cos(site2angle(landmark_start[i]))]
            label_y = [arc_r[track]*np.sin(site2angle(landmark_start[i]))]
            ddr     = 0.05 #0.11
            if landmark_label[i] in label2ddr:
                ddr = label2ddr[landmark_label[i]]

            ddx1 =  ddr * np.cos(site2angle(landmark_start[i]))
            ddy  =  ddr * np.sin(site2angle(landmark_start[i]))
            ddx2 = ddx1 + np.sign(ddx1)*0.03

            label_x = label_x + [label_x[0] + ddx1, label_x[0] + ddx2]
            label_y = label_y + [label_y[0] +  ddy, label_y[0] +  ddy]
            if label_x[0]<0:
                txtprops['ha'] = 'right'
            else:
                txtprops['ha'] = 'left'

            ax.text(label_x[-1], label_y[-1], landmark_label[i], **txtprops)
            mp.line(ax=ax, x=[label_x], y=[label_y], colors=[BKCOLOR], plotprops=plotprops)

#        # plot normally
#        else:
#            delta_site = 500
#            if landmark_label[i] in ['vif']:
#                delta_site += landmark_text_d[landmark_label[i]] + 25
#            txt_dr   = 0.04
#            txt_ang  = site2angle(landmark_start[i]+delta_site)
#            txt_rot  = site2angle(landmark_start[i]+delta_site, deg=True)-90
#            txt_x    = (arc_r[track] + (arc_dr/2) + txt_dr)*np.cos(txt_ang)
#            txt_y    = (arc_r[track] + (arc_dr/2) + txt_dr)*np.sin(txt_ang)
#            txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY,
#                            size=SIZELABEL, rotation=txt_rot)
#            ax.text(txt_x, txt_y, landmark_label[i], **txtprops)
            
    # Arcs for internal landmarks

    arc_r        = 0.65
    arc_dr       = 0.01
    track        = 0
    mark_patches = []
    mark_text_d  = { }

    for i in range(len(mark_label)):
        
        arc_props = dict(center=[0,0], r=arc_r, width=arc_dr, lw=AXWIDTH/2, ec=BKCOLOR, fc='none',
                         theta1=site2angle(mark_end[i], deg=True), theta2=site2angle(mark_start[i], deg=True))
        mark_patches.append(matplotlib.patches.Wedge(**arc_props))
        
        # label with line
        if True:
            txtprops  = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=0)
            plotprops = dict(lw=AXWIDTH/2, ls='-', clip_on=False)

            label_x = [arc_r*np.cos(site2angle(mark_start[i]))]
            label_y = [arc_r*np.sin(site2angle(mark_start[i]))]
            ddr     = -0.11
            if mark_label[i] in label2ddr:
                ddr = label2ddr[mark_label[i]]

            ddx1 =  ddr * np.cos(site2angle(mark_start[i]))
            ddy  =  ddr * np.sin(site2angle(mark_start[i]))
            ddx2 = ddx1 - np.sign(ddx1)*0.03

            label_x = label_x + [label_x[0] + ddx1]
            label_y = label_y + [label_y[0] +  ddy]
            if label_x[0]<0:
                txtprops['ha'] = 'left'
            else:
                txtprops['ha'] = 'right'

            ax.text(label_x[-1] + ddx1/3, label_y[-1] + ddy/3, mark_label[i], **txtprops)
            mp.line(ax=ax, x=[label_x], y=[label_y], colors=[BKCOLOR], plotprops=plotprops)

    # Arcs for TGCA selection tracks
    
    for i in range(len(level_rad)):
        arc_props = dict(center=[0,0], r=level_rad[i], width=0, lw=AXWIDTH/2, ec=C_NEU_LT, fc='none', theta1=0, theta2=360)
        landmark_patches.append(matplotlib.patches.Wedge(**arc_props))

    # Make plot

    pprops = { 'xlim':    [-1.05, 1.05],
               'ylim':    [-1.05, 1.05],
               'xticks':  [],
               'yticks':  [],
               'hide':    ['left','right', 'top', 'bottom'],
               'noaxes':  True }

    for i in range(len(dot_x)):
        plotprops = dict(lw=0, marker='o', s=dot_sizes[i], zorder=9999)
        if i<len(dot_x)-1:
            mp.scatter(             ax=ax, x=[[dot_x[i]]], y=[[dot_y[i]]], colors=[dot_colors[i]], plotprops=plotprops)
        else:
            mp.plot(type='scatter', ax=ax, x=[[dot_x[i]]], y=[[dot_y[i]]], colors=[dot_colors[i]], plotprops=plotprops, **pprops)

    for patch in landmark_patches:
        ax.add_artist(patch)
        
    for patch in mark_patches:
        ax.add_artist(patch)

    return sig_s, sig_site_real, sig_nuc_idx #, epitope_start, epitope_end
    

def plot_trajectories_slides(trajectory_file, location, f_cutoff=0.05, interpolate=False):
    """ Map out trajectories of all mutations in a specific region over time. """
        
    # Load data
    df_traj = pd.read_csv(trajectory_file, comment='#', memory_map=True)
    df_traj = df_traj[df_traj['location'].str.contains(location)]
    
    muts = np.unique(df_traj['variant_name'])
    xs   = []
    ys   = []
    if interpolate:
        for m in muts:
            df_temp = df_traj[df_traj['variant_name']==m]
            temp_xs = []
            temp_ys = []
            for idx, entry in df_temp.iterrows():
                temp_xs.append(entry['times'])
                temp_ys.append(entry['frequency_trajectory'])
            t_start = np.array([int(x.split()[0]) for x in temp_xs])
            t_order = np.argsort(t_start)
            xs.append(np.array(' '.join(np.array(temp_xs)[t_order]).split(), int))
            ys.append(np.array(' '.join(np.array(temp_ys)[t_order]).split(), float))
    else:
        xs = np.array([np.array(df_traj.iloc[i]['times'].split(), int) for i in range(len(df_traj))], dtype=object)
        ys = np.array([np.array(df_traj.iloc[i]['frequency_trajectory'].split(), float) for i in range(len(df_traj))], dtype=object)
        
    ms = np.array([np.max(y) for y in ys])
    xs = np.array(xs, dtype=object)[ms>f_cutoff]
    ys = np.array(ys, dtype=object)[ms>f_cutoff]
    print('Including %d trajectories' % len(xs))
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH
    goldh = w / 3.3
    fig   = plt.figure(figsize=(w, goldh))

    box_traj = dict(left=0.05, right=0.95, bottom=0.05, top=0.95)
    gs_traj  = gridspec.GridSpec(1, 1, **box_traj)
    
    ### trajectories
    
    ax = plt.subplot(gs_traj[0, 0])
    
    legendprops = { 'loc' : 4, 'frameon' : False, 'scatterpoints' : 1, 'handletextpad' : 0.1,
                    'prop' : {'size' : SIZELABEL}, 'ncol' : 1 }
    lineprops   = { 'lw' : SIZELINE, 'linestyle' : '-', 'alpha' : 0.5 }
    #fillprops   = { 'lw' : 0, 'alpha' : 0.3, 'interpolate' : True }
    
    pprops = { 'xticks' : [],
               'yticks' : [0, 1],
               'hide'   : ['top', 'bottom', 'left', 'right'],
               'theme'  : 'open' }
    
    t_max       = 600
    xticks      = [      61,      153,      245,      336,      426,      518, 600]
    xticklabels = ['Mar 20', 'Jun 20', 'Sep 20', 'Dec 20', 'Mar 21', 'Jun 21',  '']
#    xticklabels = [  '3/20',   '6/20',   '9/20',  '12/20',   '3/21',   '6/21',  '']
    
    colors = sns.hls_palette(len(xs))
    
    for i in range(len(xs)):
        times = xs[i]
        freqs = ys[i]
        
        full_times = times
        full_freqs = freqs
        
#        if interpolate:
#            full_times = np.arange(times[0], times[-1], 1)
#            full_freqs = np.zeros(len(full_times))
#            idx = 0
#            for k in range(len(full_times)):
#                t = full_times[k]
#                if t in times:
#                    idx = times.tolist().index(t)
#                    full_freqs[k] = freqs[idx]
#                else:
#                    f0 = freqs[idx]
#                    f1 = freqs[idx+1]
#                    t0 = times[idx]
#                    t1 = times[idx+1]
#                    x  = (t - t0) / (t1 - t0)
#                    full_freqs[k] = (1 - x)*f0 + x*f1
            
        pprops['colors'] = [colors[i]]
        pprops['xlim']   = [    61, t_max]
        pprops['ylim']   = [-0.005, 1.005]
        if (i==len(xs)-1):
            pprops['xticks']      = xticks
            pprops['xticklabels'] = ['' for xt in xticks] #xticklabels
            pprops['yticklabels'] = ['', '']
            pprops['hide']        = ['top', 'right']
            pprops['axoffset']    = 0.02
            mp.plot(type='line', ax=ax, x=[full_times], y=[full_freqs], plotprops=lineprops, **pprops)
        else:
            mp.line(             ax=ax, x=[full_times], y=[full_freqs], plotprops=lineprops, **pprops)
    
#    txtprops = dict(ha='center', va='center', rotation=90, color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, clip_on=False)
#    ax.text(box_traj['left']-0.04, box_traj['top']-(box_traj['top']-box_traj['bottom'])/2, 'Frequency', transform=fig.transFigure, **txtprops)

    # SAVE FIGURE

    plt.savefig('%s/fig-trajectories-%s-slides%s' % (FIG_DIR, location, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    

def plot_cases_slides(ihme_file, location):
    """ Plot estimated number of SARS-CoV-2 cases in a specific region over time. """
        
    # Load data
    df = pd.read_csv(ihme_file, comment='#', memory_map=True)
    df = df[df['location_name'].str.contains(location)]
    
    dates = [dt.datetime.fromisoformat(df.iloc[i]['date']) for i in range(len(df))]
    xs = [np.array([int((dates[i] - dt.datetime(2020,1,1)).days) for i in range(len(df))])]
    ys = [np.array([df.iloc[i]['inf_mean'] for i in range(len(df))])]
        
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH
    goldh = w / 3.3
    fig   = plt.figure(figsize=(w, goldh))

    box_traj = dict(left=0.10, right=0.95, bottom=0.10, top=0.95)
    gs_traj  = gridspec.GridSpec(1, 1, **box_traj)
    
    ### trajectories
    
    ax = plt.subplot(gs_traj[0, 0])
    
    legendprops = { 'loc' : 4, 'frameon' : False, 'scatterpoints' : 1, 'handletextpad' : 0.1,
                    'prop' : {'size' : SIZELABEL}, 'ncol' : 1 }
    lineprops   = { 'lw' : SIZELINE, 'linestyle' : '-', 'alpha' : 1.0 }
    
    xticks      = [ 50, 100, 150, 200, 250, 300, 350]
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
    xticklabels = [f'{i.month}/{i.day}' for i in xticklabels]
    
    pprops = { 'xticks'     : xticks,
               'xticklabels': xticklabels,
               'yticks'     : [0, 2e4, 4e4, 6e4, 8e4, 1e5],
#               'yticks'     : [1e2, 1e3, 1e4, 1e5],
#               'logy'       : True,
               'hide'       : ['top', 'bottom', 'left', 'right'],
               'theme'      : 'open' }
    
    colors = [COLOR_1]
    
    for i in range(len(xs)):
        times = xs[i]
        cases = ys[i]
        
        pprops['colors'] = [colors[i]]
        pprops['xlim']   = [np.min(xticks), np.max(xticks)]
        if (i==len(xs)-1):
            pprops['hide']     = ['top', 'right']
            pprops['axoffset'] = 0.02
            mp.plot(type='line', ax=ax, x=[times], y=[cases], plotprops=lineprops, **pprops)
        else:
            mp.line(             ax=ax, x=[times], y=[cases], plotprops=lineprops, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/fig-cases-%s-slides%s' % (FIG_DIR, location, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)

    
def plot_variant_selection_slides(variant_file, trajectory_file, variant_list=[], variant_names=[], s_cutoff=0.05):
    """ Map out mutations in the major variants, their aggregate inferred selection coefficients, and
        frequency trajectories across regions. """
        
    # adjust color list if necessary
    local_colors = deepcopy(VAR_COLORS)
    if len(variant_names)>8:
        local_colors = sns.husl_palette(len(variant_names))
        
    # Load data
    df_var = pd.read_csv(variant_file, comment='#', memory_map=True)
    df_sel = 0
    if len(variant_list)>0:
        df_sel = df_var[df_var.variant_names.isin(variant_list)]
        temp_var_list  = list(df_sel.variant_names)
        temp_var_names = []
        for i in range(len(temp_var_list)):
            temp_var_names.append(variant_names[variant_list.index(temp_var_list[i])])
        variant_list  = np.array(temp_var_list)
        variant_names = np.array(temp_var_names)
    else:
        df_sel = df_var[np.fabs(df_var.selection_coefficient)>s_cutoff]
        variant_list  = np.array(list(df_sel.variant_names))
        variant_names = np.array(variant_list)
    
    s_sort = np.argsort(np.fabs(df_sel.selection_coefficient))[::-1]
    print(np.array(np.fabs(df_sel.selection_coefficient))[s_sort])
    print(np.array(df_sel.variant_names)[s_sort])
    
    n_vars        = len(s_sort)
    variant_list  = variant_list[s_sort]
    variant_names = variant_names[s_sort]
    
    df_traj = pd.read_csv(trajectory_file, comment='#', memory_map=True)
    df_traj = df_traj[df_traj.variant_names.isin(variant_list)]
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH #6 #SLIDE_WIDTH
    goldh = w / 1.8 #2.37 #w / 1.8
    fig   = plt.figure(figsize=(w, goldh))

    box_traj = dict(left=0.08, right=0.95, bottom=0.17, top=0.97)
    #box_sel  = dict(left=0.80, right=0.95, bottom=0.13, top=0.95)

    gs_traj = gridspec.GridSpec(n_vars, 2, width_ratios=[2, 1], wspace=0.05, **box_traj)
    #gs_sel  = gridspec.GridSpec(1, 1, **box_sel)
    
    ### trajectories
    
    ax = [[plt.subplot(gs_traj[i, 0]), plt.subplot(gs_traj[i, 1])] for i in range(n_vars)]
    
    legendprops = { 'loc' : 4, 'frameon' : False, 'scatterpoints' : 1, 'handletextpad' : 0.1,
                    'prop' : {'size' : SIZELABEL}, 'ncol' : 1 }
    lineprops   = { 'lw' : SIZELINE, 'linestyle' : '-', 'alpha' : 0.5 }
    #fillprops   = { 'lw' : 0, 'alpha' : 0.3, 'interpolate' : True }
    
    pprops = { 'xticks' : [],
               'yticks' : [],
               'hide'   : ['top', 'bottom', 'left', 'right'],
               'theme'  : 'open' }
    
    t_max = 600
    xticks      = [      61,      153,      245,      336,      426,      518, 600]
    #xticklabels = ['Mar 20', 'Jun 20', 'Sep 20', 'Dec 20', 'Mar 21', 'Jun 21',  '']
    xticklabels = [  '3/20',   '6/20',   '9/20',  '12/20',   '3/21',   '6/21',  '']
    
    for i in range(n_vars):
        idx = list(s_sort)[i]
        pprops['colors'] = [local_colors[i]]
        pprops['xlim']   = [   61, t_max]
        pprops['ylim']   = [-0.05,  1.05]
        if (i==n_vars-1):
#            pprops['xticks']   = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
#            pprops['xlabel']   = 'Time (generations)'
            pprops['xticks']      = xticks
            pprops['xticklabels'] = xticklabels
            pprops['hide']        = ['top', 'left', 'right']
            pprops['axoffset']    = 0.18
        df_temp = df_traj[df_traj.variant_names==df_sel.iloc[idx].variant_names]
        for k in range(len(df_temp)):
            entry = df_temp.iloc[k]
            times = np.array(entry.times.split(), float)
            freqs = np.array(entry.frequencies.split(), float)
            if k==len(df_temp)-1:
                mp.plot(type='line', ax=ax[i][0], x=[times], y=[freqs], plotprops=lineprops, **pprops)
            else:
                mp.line(             ax=ax[i][0], x=[times], y=[freqs], plotprops=lineprops, **pprops)
            
        txtprops = dict(ha='left', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
        if variant_list[i]=='B.1':
            ax[i][0].text(101, 0.1, variant_names[i], **txtprops)
        else:
            ax[i][0].text( 61, 0.8, variant_names[i], **txtprops)
    
    txtprops = dict(ha='center', va='center', rotation=90, color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, clip_on=False)
    ax[-1][0].text(box_traj['left']-0.04, box_traj['top']-(box_traj['top']-box_traj['bottom'])/2, 'Frequency', transform=fig.transFigure, **txtprops)
            
    ### selection coefficient estimates
    
    dashlineprops = dict(lw=SIZELINE, ls=':', alpha=0.25, clip_on=False, zorder=-9999)
    sprops = dict(lw=0, s=9., marker='o')
    
    pprops = { 'yticks': [],
               'xticks': [],
               'hide'  : ['top', 'bottom', 'left', 'right'],
               'theme' : 'open' }
    
#    xticks      = [0, 0.2, 0.4, 0.6, 0.8]
#    xticklabels = [0, 20, 40, 60, 80]
    xticks      = [0, 0.25, 0.5, 0.75, 1.0]
    xticklabels = [0, 25, 50, 75, 100]
    
    for i in range(len(xticks)):
        xdat = [xticks[i], xticks[i]]
        ydat = [0.5, n_vars+2.0]
        mp.line(ax=ax[-1][1], x=[xdat], y=[ydat], colors=[BKCOLOR], plotprops=dashlineprops)
    
    for i in range(n_vars):
        idx = list(s_sort)[i]
        pprops['xlim'] = [   0,   1]
        pprops['ylim'] = [ 0.5, 1.5]
        ydat           = [1]
        xdat           = [df_sel.iloc[idx].selection_coefficient]
        xerr           = [df_sel.iloc[idx]['theoretical_error']]
        if (i==n_vars-1):
            pprops['xticks']      = xticks
            pprops['xticklabels'] = xticklabels
            #pprops['xlabel']      = 'Inferred selection\ncoefficient (%)'
            pprops['hide']        = ['top', 'left', 'right']
            pprops['axoffset']    = 0.3
        mp.plot(type='error', ax=ax[i][1], x=[xdat], y=[ydat], xerr=[xerr], colors=[local_colors[i]], **pprops)

    # SAVE FIGURE

    plt.savefig('%s/fig-variants%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    

def plot_null_slides(null_file):
    """ Plot the null distribution for mutations inferred to be nearly neutral.
        Plot frequencies and early detection times for Alpha and Delta variants. """
    
    # Load data
    inferred_null = np.loadtxt(null_file)
    
    # Null distribution statistics
    max_null    = np.amax(inferred_null)
    min_null    = np.amin(inferred_null)
    num_largest = int(round(len(inferred_null) * 2.5 / 100))    # the number of coefficients in the top 2.5% of the null distribution
    soft_cutoff = np.sort(inferred_null)[::-1][num_largest]
    print('hard w cutoff = %.4f' % max_null)
    print('soft w cutoff = %.4f\n' % soft_cutoff)
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH * 0.75
    goldh = w / 1.5
    fig   = plt.figure(figsize=(w, goldh))

    left_bd   = 0.13
    right_bd  = 0.90
    top_bd    = 0.95
    bottom_bd = 0.25
    
    box_null = dict(left=left_bd, right=right_bd, top=top_bd, bottom=bottom_bd)
    
    ## null distribution of selection coefficients

    gs_null = gridspec.GridSpec(1, 1, **box_null)
    ax_null = plt.subplot(gs_null[0, 0])
    
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=1, edgecolor='none',
                     orientation='vertical', log=True)
                     
#    pprops = { 'xlim':        [-0.04, 0.045],
#               'xticks':      [-0.04, -0.02, 0., 0.02, 0.04, 0.045],
#               'xticklabels': [r'$-4$', r'$-2$', r'$0$', r'$2$', r'$4$', ''],
#               'ylim':        [1e0, 1e6],
#               'yticks':      [1e0, 1e2, 1e4, 1e6],
#               #'yticklabels': [0, 10, 20, 30, 40, 50],
#               'logy':        True,
#               'ylabel':      'Counts',
#               'xlabel':      'Inferred selection coefficients for neutral variants, ' + r'$\hat{w}$' + ' (%)',
#               'bins':        np.arange(-0.04, 0.045+0.001, 0.001),
#               'weights':     [np.ones_like(inferred_null)],
#               'plotprops':   histprops,
#               'axoffset':    0.1,
#               'theme':       'open' }

    pprops = { 'xlim':        [-0.10, 0.15],
               'xticks':      [-0.10, -0.05, 0., 0.05, 0.10, 0.15],
               'xticklabels': [r'$-10$', r'$-5$', r'$0$', r'$5$', r'$10$', r'$15$'],
               'ylim':        [1e0, 1e6],
               'yticks':      [1e0, 1e2, 1e4, 1e6],
               #'yticklabels': [0, 10, 20, 30, 40, 50],
               'logy':        True,
               'ylabel':      'Counts',
               'xlabel':      'Local selection coefficients for mutations with\nglobal selection coefficients inferred <10%, ' + r'$\hat{w}$' + ' (%)',
               'bins':        np.arange(-0.10, 0.15+0.0025, 0.0025),
               'weights':     [np.ones_like(inferred_null)],
               'plotprops':   histprops,
               'axoffset':    0.1,
               'theme':       'open' }

    x = [inferred_null]
    mp.plot(type='hist', ax=ax_null, x=x, colors=[LCOLOR], **pprops)

    # SAVE FIGURE

    plt.savefig('%s/fig-null-slides%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    
    
def plot_early_detection_sub_slides(alpha_file_subdate,   alpha_start,   alpha_end,
                                    delta_file_subdate,   delta_start,   delta_end,
                                    omicron_file_subdate, omicron_start, omicron_end):
    """ Plot early detection times for Alpha, Delta, and Omicron variants by submission date. """
    
    # Load data
    df_alpha_sub   = pd.read_csv(alpha_file_subdate, comment='#', memory_map=True)
    df_alpha_sub   = df_alpha_sub[(df_alpha_sub['time']>=alpha_start) & (df_alpha_sub['time']<=alpha_end)]
    df_delta_sub   = pd.read_csv(delta_file_subdate, comment='#', memory_map=True)
    df_delta_sub   = df_delta_sub[(df_delta_sub['time']>=delta_start) & (df_delta_sub['time']<=delta_end)]
    df_omicron_sub = pd.read_csv(omicron_file_subdate, comment='#', memory_map=True)
    df_omicron_sub = df_omicron_sub[(df_omicron_sub['time']>=omicron_start) & (df_omicron_sub['time']<=omicron_end)]
    
    # Get early detection times
    s_cutoff  = 0.185
    
    t_alpha   = df_alpha_sub[df_alpha_sub['selection']>s_cutoff].iloc[0]['time']
    #f_alpha   = df_alpha_sub[df_alpha_sub['selection']>s_cutoff].iloc[0]['frequency']
    t_delta   = df_delta_sub[df_delta_sub['selection']>s_cutoff].iloc[0]['time']
    #f_delta   = df_delta_sub[df_delta_sub['selection']>s_cutoff].iloc[0]['frequency']
    t_omicron = df_omicron_sub[df_omicron_sub['selection']>s_cutoff].iloc[0]['time']
    #f_omicron = df_omicron_sub[df_omicron_sub['selection']>s_cutoff].iloc[0]['frequency']
    
    print('by submission date')
    print('cutoff\tstatistic\talpha\tdelta\tomicron')
    print('%.2f\ttime\t\t%d\t%d\t%d'       % (s_cutoff, t_alpha, t_delta, t_omicron))
    #print('%.2f\tfreq\t\t%.3f\t%.3f\t%.3f' % (s_cutoff, f_alpha, f_delta, f_omicron))
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH
    goldh = w / 3
    fig   = plt.figure(figsize=(w, goldh))

    left_bd   = 0.15
    right_bd  = 0.95
    top_bd    = 0.85
    bottom_bd = 0.24
    
    box_sel = dict(left=left_bd, right=right_bd, top=top_bd, bottom=bottom_bd)
    
    # a/b/c -- Alpha/Delta/Omicron inferred selection
    
    gs_sel = gridspec.GridSpec(1, 1, **box_sel)
    ax_sel = plt.subplot(gs_sel[0, 0])
    
    ylima   = ylimd   = ylimo   = [0, 1.20]
    yticka  = ytickd  = yticko  = [0, 0.40, 0.80, 1.20]
    ytickla = ytickld = yticklo = [int(i) for i in 100*np.array(yticka)]
    ymticka = ymtickd = ymticko = [0.20, 0.60, 1.00]
    
    lineprops  = dict(lw=2*SIZELINE, ls='-', alpha=1)
    blineprops = dict(lw=3*SIZELINE, ls='-', alpha=1)
    dashprops  = dict(lw=2*SIZELINE, ls=':', alpha=0.5)
    
    for ax_sel, temp_df_sub, t_det, ylim, yticks, yticklabels, yminorticks, v_color, v_name, v_region in [
        [ax_sel,   df_alpha_sub,   t_alpha, ylima, yticka, ytickla, ymticka, C_ALPHA,   'Alpha',   'Great Britain'],
        [ax_sel,   df_delta_sub,   t_delta, ylimd, ytickd, ytickld, ymtickd, C_DELTA,   'Delta',   'Great Britain'],
        [ax_sel, df_omicron_sub, t_omicron, ylimo, yticko, yticklo, ymticko, C_OMICRON, 'Omicron', 'South Africa'] ]:
    
        ## a/b/c.1 -- frequency trajectory -- NOT PLOTTED HERE
        
        times_sub = np.array(temp_df_sub['time'])
        sels_sub  = np.array(temp_df_sub['selection'])
        
        xticks      = np.arange(np.min(times_sub), np.max(times_sub), 14)
        xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
        yearlabel   = xticklabels[0].strftime('%Y')
        xticklabels = [i.strftime('%-d %b') for i in xticklabels]
        
        ## a/b/c.2 -- inferred selection coefficient

        pprops = { 'xlim':        [np.min(times_sub), np.max(times_sub)],
                   'xticks':      xticks,
                   'xticklabels': xticklabels,
                   'ylim':        [yy for yy in ylim],
                   'yticks':      yticks,
                   'yticklabels': yticklabels,
                   'yminorticks': yminorticks,
                   'xlabel':      yearlabel,
                   'ylabel':      '', #('Selection coefficient\nfor %s mutations\nin %s' % (v_name, v_region)) + ' (%)',
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times_sub]
        ydat = [sels_sub]
        mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[v_color], plotprops=lineprops, **pprops)

        xdat = [[t_det, t_det]]
        ydat = [[0, 1.2]]
        mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        if v_name=='Alpha':
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='right', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_sel.text(t_det-1.5, 1.15, 'Variant first inferred\nto be concerning\n%s' % t_det2string, **txtprops)
        elif v_name=='Delta':
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='right', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_sel.text(t_det-1.5, 1.15, '%s' % t_det2string, **txtprops)
        else:
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_sel.text(t_det+1.5, 0.90, '%s' % t_det2string, **txtprops)
            
        txtprops = dict(ha='right', va='bottom', color=v_color, family=FONTFAMILY, size=SIZELABEL, weight='bold')
        ax_sel.text(xticks[-1], 0.01, '%s' % v_name, **txtprops)
        
        plt.savefig('%s/fig-early-detection-subdate-%s-slides%s' % (FIG_DIR, v_name, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        ax_sel.clear()

    plt.close(fig)
    
    
def plot_early_detection_slides(alpha_file,   alpha_file_subdate,   alpha_start,   alpha_end,
                                delta_file,   delta_file_subdate,   delta_start,   delta_end,
                                omicron_file, omicron_file_subdate, omicron_start, omicron_end):
    """ Plot early detection times for Alpha, Delta, and Omicron variants by submission and collection time. """
    
    # Load data
    df_alpha   = pd.read_csv(alpha_file, comment='#', memory_map=True)
    df_alpha   = df_alpha[(df_alpha['time']>=alpha_start) & (df_alpha['time']<=alpha_end)]
    df_delta   = pd.read_csv(delta_file, comment='#', memory_map=True)
    df_delta   = df_delta[(df_delta['time']>=delta_start) & (df_delta['time']<=delta_end)]
    df_omicron = pd.read_csv(omicron_file, comment='#', memory_map=True)
    df_omicron = df_omicron[(df_omicron['time']>=omicron_start) & (df_omicron['time']<=omicron_end)]
    
    df_alpha_sub   = pd.read_csv(alpha_file_subdate, comment='#', memory_map=True)
    df_alpha_sub   = df_alpha_sub[(df_alpha_sub['time']>=alpha_start) & (df_alpha_sub['time']<=alpha_end)]
    df_delta_sub   = pd.read_csv(delta_file_subdate, comment='#', memory_map=True)
    df_delta_sub   = df_delta_sub[(df_delta_sub['time']>=delta_start) & (df_delta_sub['time']<=delta_end)]
    df_omicron_sub = pd.read_csv(omicron_file_subdate, comment='#', memory_map=True)
    df_omicron_sub = df_omicron_sub[(df_omicron_sub['time']>=omicron_start) & (df_omicron_sub['time']<=omicron_end)]
    
    # Get early detection times
    s_cutoff  = 0.185
    t_alpha   = df_alpha[df_alpha['selection']>s_cutoff].iloc[0]['time']
    f_alpha   = df_alpha[df_alpha['selection']>s_cutoff].iloc[0]['frequency']
    t_delta   = df_delta[df_delta['selection']>s_cutoff].iloc[0]['time']
    f_delta   = df_delta[df_delta['selection']>s_cutoff].iloc[0]['frequency']
    t_omicron = df_omicron[df_omicron['selection']>s_cutoff].iloc[0]['time']
    f_omicron = df_omicron[df_omicron['selection']>s_cutoff].iloc[0]['frequency']
   
    print('by collection time')
    print('cutoff\tstatistic\talpha\tdelta\tomicron')
    print('%.2f\ttime\t\t%d\t%d\t%d'       % (s_cutoff, t_alpha, t_delta, t_omicron))
    print('%.2f\tfreq\t\t%.3f\t%.3f\t%.3f' % (s_cutoff, f_alpha, f_delta, f_omicron))
    print('')
    
    t_alpha   = df_alpha_sub[df_alpha_sub['selection']>s_cutoff].iloc[0]['time']
    #f_alpha   = df_alpha_sub[df_alpha_sub['selection']>s_cutoff].iloc[0]['frequency']
    t_delta   = df_delta_sub[df_delta_sub['selection']>s_cutoff].iloc[0]['time']
    #f_delta   = df_delta_sub[df_delta_sub['selection']>s_cutoff].iloc[0]['frequency']
    t_omicron = df_omicron_sub[df_omicron_sub['selection']>s_cutoff].iloc[0]['time']
    #f_omicron = df_omicron_sub[df_omicron_sub['selection']>s_cutoff].iloc[0]['frequency']
    
    print('by submission date')
    print('cutoff\tstatistic\talpha\tdelta\tomicron')
    print('%.2f\ttime\t\t%d\t%d\t%d'       % (s_cutoff, t_alpha, t_delta, t_omicron))
    #print('%.2f\tfreq\t\t%.3f\t%.3f\t%.3f' % (s_cutoff, f_alpha, f_delta, f_omicron))
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH
    goldh = w / 3
    fig   = plt.figure(figsize=(w, goldh))

    left_bd   = 0.15
    right_bd  = 0.95
    top_bd    = 0.85
    bottom_bd = 0.24
    
    box_sel = dict(left=left_bd, right=right_bd, top=top_bd, bottom=bottom_bd)
    
    # a/b/c -- Alpha/Delta/Omicron frequency and inferred selection
    
    gs_sel = gridspec.GridSpec(1, 1, **box_sel)
    ax_sel = plt.subplot(gs_sel[0, 0])
    
    ylima   = ylimd   = ylimo   = [0, 1.20]
    yticka  = ytickd  = yticko  = [0, 0.40, 0.80, 1.20]
    ytickla = ytickld = yticklo = [int(i) for i in 100*np.array(yticka)]
    ymticka = ymtickd = ymticko = [0.20, 0.60, 1.00]
    
    lineprops  = dict(lw=2*SIZELINE, ls='-', alpha=1)
    blineprops = dict(lw=3*SIZELINE, ls='-', alpha=1)
    dashprops  = dict(lw=2*SIZELINE, ls=':', alpha=0.5)
    
    for ax_sel, temp_df, temp_df_sub, t_det, ylim, yticks, yticklabels, yminorticks, v_color, v_name, v_region in [
        [ax_sel, df_alpha,   df_alpha_sub,   t_alpha,   ylima, yticka, ytickla, ymticka, C_ALPHA,   'Alpha',   'Great Britain'],
        [ax_sel, df_delta,   df_delta_sub,   t_delta,   ylimd, ytickd, ytickld, ymtickd, C_DELTA,   'Delta',   'Great Britain'],
        [ax_sel, df_omicron, df_omicron_sub, t_omicron, ylimo, yticko, yticklo, ymticko, C_OMICRON, 'Omicron', 'South Africa'] ]:
    
        ## a/b/c.1 -- frequency trajectory -- NOT PLOTTED HERE
        
        times     = np.array(temp_df['time'])
        sels      = np.array(temp_df['selection'])
        times_sub = np.array(temp_df_sub['time'])
        sels_sub  = np.array(temp_df_sub['selection'])
        
        xticks      = np.arange(np.min(times), np.max(times), 14)
        xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
        yearlabel   = xticklabels[0].strftime('%Y')
        xticklabels = [i.strftime('%-d %b') for i in xticklabels]
        
        ## a/b/c.2 -- inferred selection coefficient

        pprops = { 'xlim':        [np.min(times), np.max(times)],
                   'xticks':      xticks,
                   'xticklabels': xticklabels,
                   'ylim':        [yy for yy in ylim],
                   'yticks':      yticks,
                   'yticklabels': yticklabels,
                   'yminorticks': yminorticks,
                   'xlabel':      yearlabel,
                   'ylabel':      ('Selection coefficient\nin %s, ' % (v_region)) + r'$\hat{w}$' + ' (%)',
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times]
        ydat = [sels]
        mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[LCOLOR], plotprops=lineprops, **pprops)
        
        xdat = [times_sub]
        ydat = [sels_sub]
        mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[v_color], plotprops=blineprops, **pprops)

        xdat = [[t_det, t_det]]
        ydat = [[0, 1.2]]
        mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        if v_name=='Alpha':
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='right', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_sel.text(t_det-1.5, 1.20, 'Variant first inferred\nto be significantly\nbeneficial\n%s' % t_det2string, **txtprops)
        elif v_name=='Delta':
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='right', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_sel.text(t_det-1.5, 1.20, '%s' % t_det2string, **txtprops)
        else:
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='right', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_sel.text(t_det-1.5, 0.90, '%s' % t_det2string, **txtprops)
            
        txtprops = dict(ha='right', va='bottom', color=v_color, family=FONTFAMILY, size=SIZELABEL, weight='bold')
        ax_sel.text(xticks[-1], 0.01, '%s' % v_name, **txtprops)
        
        plt.savefig('%s/fig-early-detection-%s-slides%s' % (FIG_DIR, v_name, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        ax_sel.clear()

#    ## legends and labels
#
#    lineprops['clip_on']  = False
#    blineprops['clip_on'] = False
#
#    invt = ax_sel_a.transData.inverted()
#    xy1  = invt.transform((   0, 0))
#    xy2  = invt.transform((7.50, 9))
#    xy3  = invt.transform((3.00, 9))
#    xy4  = invt.transform((5.25, 9))
#
#    legend_dx1   = (xy1[0]-xy2[0])
#    legend_dx2   = (xy1[0]-xy3[0])
#    legend_dx    = (xy1[0]-xy4[0])
#    legend_dy    = (xy1[1]-xy2[1])
#    times        = np.array(df_alpha['time'])
#    legend_x     = np.min(times) + 3
#    legend_y     = [1.35, 1.35 + legend_dy]
#    legend_t     = ['by submission date', 'by collection date']
#    legend_props = [blineprops, lineprops]
#    for k in range(len(legend_y)):
#        yy = legend_y[k]
#        mp.line(ax=ax_sel_a, x=[[legend_x+legend_dx2, legend_x+legend_dx1]], y=[[yy, yy]], colors=[LCOLOR], plotprops=legend_props[k], **pprops)
#        ax_sel_a.text(legend_x, yy, legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

#    dx = -0.06
#    dy =  0.01
#    ax_sel_a.text(box_sel_a['left'] + dx, box_sel_a['top'] + dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
#    ax_sel_d.text(box_sel_d['left'] + dx, box_sel_d['top'] + dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
#    ax_sel_o.text(box_sel_o['left'] + dx, box_sel_o['top'] + dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

#    plt.savefig('%s/fig-early-detection-slides%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


#### OLD VERSIONS ####

def plot_early_detection_slides_old(null_file, alpha_file, alpha_start, alpha_end, delta_file, delta_start, delta_end, omicron_file, omicron_start, omicron_end, variant='Alpha'):
    """ Plot the null distribution for mutations inferred to be nearly neutral.
        Plot frequencies and early detection times for Alpha and Delta variants. """
    
    # Load data
    inferred_null = np.loadtxt(null_file)
    df_alpha      = pd.read_csv(alpha_file, comment='#', memory_map=True)
    df_alpha      = df_alpha[(df_alpha['time']>=alpha_start) & (df_alpha['time']<=alpha_end)]
    df_delta      = pd.read_csv(delta_file, comment='#', memory_map=True)
    df_delta      = df_delta[(df_delta['time']>=delta_start) & (df_delta['time']<=delta_end)]
    df_omicron    = pd.read_csv(omicron_file, comment='#', memory_map=True)
    df_omicron    = df_omicron[(df_omicron['time']>=omicron_start) & (df_omicron['time']<=omicron_end)]
    
    # Null distribution statistics
    max_null    = np.amax(inferred_null)
    min_null    = np.amin(inferred_null)
    num_largest = int(round(len(inferred_null) * 2.5 / 100))    # the number of coefficients in the top 2.5% of the null distribution
    soft_cutoff = np.sort(inferred_null)[::-1][num_largest]
    print('hard w cutoff = %.4f' % max_null)
    print('soft w cutoff = %.4f\n' % soft_cutoff)
    
    # Get early detection times
    
    t_alpha        = df_alpha[df_alpha['selection']>max_null].iloc[0]['time']
    f_alpha        = df_alpha[df_alpha['selection']>max_null].iloc[0]['frequency']
    t_soft_alpha   = df_alpha[df_alpha['selection']>soft_cutoff].iloc[0]['time']
    f_soft_alpha   = df_alpha[df_alpha['selection']>soft_cutoff].iloc[0]['frequency']
    t_delta        = df_delta[df_delta['selection']>max_null].iloc[0]['time']
    f_delta        = df_delta[df_delta['selection']>max_null].iloc[0]['frequency']
    t_soft_delta   = df_delta[df_delta['selection']>soft_cutoff].iloc[0]['time']
    f_soft_delta   = df_delta[df_delta['selection']>soft_cutoff].iloc[0]['frequency']
    t_omicron      = df_omicron[df_omicron['selection']>max_null].iloc[0]['time']
    f_omicron      = df_omicron[df_omicron['selection']>max_null].iloc[0]['frequency']
    t_soft_omicron = df_omicron[df_omicron['selection']>soft_cutoff].iloc[0]['time']
    f_soft_omicron = df_omicron[df_omicron['selection']>soft_cutoff].iloc[0]['frequency']
   
#    print('time')
#    print('cutoff\talpha\tdelta\tomicron')
#    print('hard\t%d\t%d\t%d' % (t_alpha, t_delta, t_omicron))
#    print('soft\t%d\t%d\t%d' % (t_soft_alpha, t_soft_delta, t_soft_omicron))
#
#    print('')
#    print('frequency')
#    print('cutoff\talpha\tdelta\tomicron')
#    print('hard\t%.3f\t%.3f\t%.3f' % (f_alpha, f_delta, f_omicron))
#    print('soft\t%.3f\t%.3f\t%.3f' % (f_soft_alpha, f_soft_delta, f_soft_omicron))
#    print('')
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH
    goldh = w / 2.5
    fig   = plt.figure(figsize=(w, goldh))

    ## POSTER
    left_bd   = 0.30
    right_bd  = 0.95
    h_space   = 0.15
    box_h     = 0.22
    top_bd    = 0.95
    bottom_bd = 0.20
    v_space   = 0.10
    box_v     = (top_bd - bottom_bd - 1*v_space)/2
    
#    ## SLIDES
#    left_bd   = 0.08
#    right_bd  = 0.97
#    h_space   = 0.15
#    box_h     = 0.22
#    top_bd    = 0.92
#    bottom_bd = 0.15
#    v_space   = 0.10
#    box_v     = (top_bd - bottom_bd - 1*v_space)/2
    
    box_traj = dict(left=left_bd, right=right_bd, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_sel  = dict(left=left_bd, right=right_bd, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    
    # variant frequency and inferred selection
    
    gs_traj = gridspec.GridSpec(1, 1, **box_traj)
    ax_traj = plt.subplot(gs_traj[0, 0])
    
    gs_sel = gridspec.GridSpec(1, 1, **box_sel)
    ax_sel = plt.subplot(gs_sel[0, 0])
    
    ylim        = [0, 0.80]
    yticks      = [0, 0.40, 0.80]
    yticklabels = [int(i) for i in 100*np.array(yticks)]
    yminorticks = [0.20, 0.60]
    
    temp_df = df_alpha
    t_det   = t_alpha
    ylabel  = 'Frequency in\nLondon (%)'
    v_color = C_ALPHA
    if variant=='Delta':
        temp_df = df_delta
        t_det   = t_delta
        ylabel  = 'Frequency in\nGreat Britain (%)'
        v_color = C_DELTA
    elif variant=='Omicron':
        temp_df = df_omicron
        t_det   = t_omicron
        ylabel  = 'Frequency in\nSouth Africa (%)'
        v_color = C_OMICRON
        
    ## a/b/c.1 -- frequency trajectory
        
    times   = np.array(temp_df['time'])
    freqs   = np.array(temp_df['frequency'])
    sels    = np.array(temp_df['selection'])
    
    lineprops = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 1 }
    dashprops = {'lw': 2*SIZELINE, 'ls': ':', 'alpha': 0.5 }
    
    xticks      = np.arange(np.min(times), np.max(times), 14)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
    yearlabel   = xticklabels[0].strftime('%Y')
    xticklabels = [i.strftime('%-d %b') for i in xticklabels]
    #xticklabels = [f'{i.month}/{i.day}' for i in xticklabels]
    
    pprops = { 'xlim':        [np.min(times), np.max(times)],
               'xticks':      xticks,
               'xticklabels': ['' for xtl in xticklabels],
               'ylim':        [0, 1.05],
               'yticks':      [0, 0.5, 1],
               'yticklabels': [0, 50, 100],
               'yminorticks': [0.25, 0.75],
               'xlabel':      '',
               'ylabel':      '',
#               'ylabel':      ylabel,
               'axoffset':    0.1,
               'theme':       'open' }
    
    xdat = [times]
    ydat = [freqs]
    mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[v_color], plotprops=lineprops, **pprops)
    
    tobs = 0
    for i in range(len(ydat[0])):
        if ydat[0][i]>0:
            tobs = xdat[0][i]
            break
    print('First observed at %d, detected selection at %d\n' % (tobs, t_det))

    xdat = [[t_det, t_det]]
    ydat = [[0, 1]]
    mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
    
    ## a/b/c.2 -- inferred selection coefficient

    pprops = { 'xlim':        [np.min(times), np.max(times)],
               'xticks':      xticks,
               'xticklabels': xticklabels,
               'ylim':        ylim,
               'yticks':      yticks,
               'yticklabels': yticklabels,
               'yminorticks': yminorticks,
               'xlabel':      yearlabel,
#               'ylabel':      'Inferred selection coefficient\nfor novel mutations (%)',
               'ylabel':      '',
#               'ylabel':      'Inferred selection coefficient\nfor novel SNVs, ' + r'$\hat{w}$' + ' (%)',
               'axoffset':    0.1,
               'theme':       'open' }
    
    xdat = [times]
    ydat = [sels]
    mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[LCOLOR], plotprops=lineprops, **pprops)

    xdat = [[t_det, t_det]]
    ydat = [[0, 1]]
    mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
    
    if False: #variant=='Alpha':
        t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
        txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
        ax_traj.text(t_det+1, 0.98, 'Variant first inferred\nto be significantly\nbeneficial\n%s' % t_det2string, **txtprops)
    else:
        t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
        txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
        ax_traj.text(t_det+1, 0.98, '%s' % t_det2string, **txtprops)
        
    txtprops = dict(ha='right', va='bottom', color=v_color, family=FONTFAMILY, size=SIZELABEL, weight='bold')
    ax_sel.text(xticks[-1], 0.01, '%s' % variant, **txtprops)
    
#    ## 1 -- frequency trajectory
#
#    times   = np.array(temp_df['time'])
#    freqs   = np.array(temp_df['frequency'])
#    sels    = np.array(temp_df['selection'])
#
#    lineprops = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 1 }
#    dashprops = {'lw': 2*SIZELINE, 'ls': ':', 'alpha': 0.5 }
#
#    xticks      = np.arange(np.min(times), np.max(times), 14)
#    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
#    xticklabels = [f'{i.month}/{i.day}' for i in xticklabels]
#
#    pprops = { 'xlim':        [np.min(times), np.max(times)],
#               'xticks':      xticks,
#               'xticklabels': ['' for xtl in xticklabels],
#               'ylim':        [0, 1.05],
#               'yticks':      [0, 0.5, 1],
#               'yticklabels': [0, 50, 100],
#               'yminorticks': [0.25, 0.75],
#               'xlabel':      '',
#               #'ylabel':      ylabel,
#               'axoffset':    0.1,
#               'theme':       'open' }
#
#    xdat = [times]
#    ydat = [freqs]
#    mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[v_color], plotprops=lineprops, **pprops)
#
#    tobs = 0
#    for i in range(len(ydat[0])):
#        if ydat[0][i]>0:
#            tobs = xdat[0][i]
#            break
#    print('First observed at %d, detected selection at %d\n' % (tobs, t_det))
#
#    xdat = [[t_det, t_det]]
#    ydat = [[0, 1]]
#    mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
#
#    ## 2 -- inferred selection coefficient
#
#    pprops = { 'xlim':        [np.min(times), np.max(times)],
#               'xticks':      xticks,
#               'xticklabels': xticklabels,
#               'ylim':        ylim,
#               'yticks':      yticks,
#               'yticklabels': yticklabels,
#               'yminorticks': yminorticks,
#               #'xlabel':      'Date',
#               #'ylabel':      'Inferred selection\ncoefficient (%)',
#               'axoffset':    0.1,
#               'theme':       'open' }
#
#    xdat = [times]
#    ydat = [sels]
#    mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[LCOLOR], plotprops=lineprops, **pprops)
#
#    xdat = [[t_det, t_det]]
#    ydat = [[0, 1]]
#    mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
#
#    txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
#    ax_traj.text(t_det+1.5, 0.98, 'Variant first inferred\nto be significantly\nbeneficial', **txtprops)

    ## legends and labels
    
#    dx = -0.06
#    dy =  0.01
#    ax_null.text(  box_null['left']   + dx, box_null['top']   + dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
#    ax_traj_a.text(box_traj_a['left'] + dx, box_traj_a['top'] + dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
#    ax_traj_d.text(box_traj_d['left'] + dx, box_traj_d['top'] + dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/fig-early-%s%s' % (FIG_DIR, variant, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    
    
def plot_early_detection_slides_delta(null_file, alpha_file, alpha_start, alpha_end, delta_file, delta_start, delta_end):
    """ Plot the null distribution for mutations inferred to be nearly neutral.
        Plot frequencies and early detection times for Alpha and Delta variants. """
    
    # Load data
    inferred_null = np.loadtxt(null_file)
    df_alpha      = pd.read_csv(alpha_file, comment='#', memory_map=True)
    df_alpha      = df_alpha[(df_alpha['time']>=alpha_start) & (df_alpha['time']<=alpha_end)]
    df_delta      = pd.read_csv(delta_file, comment='#', memory_map=True)
    df_delta      = df_delta[(df_delta['time']>=delta_start) & (df_delta['time']<=delta_end)]
    
    # Null distribution statistics
    max_null    = np.amax(inferred_null)
    min_null    = np.amin(inferred_null)
    num_largest = int(round(len(inferred_null) * 2.5 / 100))    # the number of coefficients in the top 2.5% of the null distribution
    soft_cutoff = np.sort(inferred_null)[::-1][num_largest]
    print('hard cutoff = %.4f' % max_null)
    print('soft cutoff = %.4f\n' % soft_cutoff)
    
    # Get early detection times
#    t_alpha      = np.min(df_alpha[df_alpha['selection']>max_null]['time'])
#    t_soft_alpha = np.min(df_alpha[df_alpha['selection']>soft_cutoff]['time'])
#    t_delta      = np.min(df_delta[df_delta['selection']>max_null]['time'])
#    t_soft_delta = np.min(df_delta[df_delta['selection']>soft_cutoff]['time'])
    
    t_alpha      = df_alpha[df_alpha['selection']>max_null].iloc[0]['time']
    f_alpha      = df_alpha[df_alpha['selection']>max_null].iloc[0]['frequency']
    t_soft_alpha = df_alpha[df_alpha['selection']>soft_cutoff].iloc[0]['time']
    f_soft_alpha = df_alpha[df_alpha['selection']>soft_cutoff].iloc[0]['frequency']
    t_delta      = df_delta[df_delta['selection']>max_null].iloc[0]['time']
    f_delta      = df_delta[df_delta['selection']>max_null].iloc[0]['frequency']
    t_soft_delta = df_delta[df_delta['selection']>soft_cutoff].iloc[0]['time']
    f_soft_delta = df_delta[df_delta['selection']>soft_cutoff].iloc[0]['frequency']
    
#    t_alpha      = 0
#    f_alpha      = 0
#    t_soft_alpha = 0
#    f_soft_alpha = 0
   
    print('time')
    print('cutoff\talpha\tdelta')
    print('hard\t%d\t%d' % (t_alpha, t_delta))
    print('soft\t%d\t%d' % (t_soft_alpha, t_soft_delta))
    
    print('')
    print('frequency')
    print('cutoff\talpha\tdelta')
    print('hard\t%.3f\t%.3f' % (f_alpha, f_delta))
    print('soft\t%.3f\t%.3f' % (f_soft_alpha, f_soft_delta))
    print('')
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = 12.0
    goldh = (w / 2.5) * (SLIDE_WIDTH / 12.0)
    fig   = plt.figure(figsize=(w, goldh))

#    left_bd   = 0.08
#    right_bd  = 0.97
#    h_space   = 0.10
#    box_h     = (right_bd - left_bd - 2*h_space)/3
#    top_bd    = 0.92
#    bottom_bd = 0.15
#    v_space   = 0.10
#    box_v     = (top_bd - bottom_bd - 1*v_space)/2
#
#    box_null   = dict(left=left_bd + 0*box_h + 0*h_space, right=left_bd + 1*box_h + 0*h_space, top=top_bd, bottom=bottom_bd)
#    box_traj_a = dict(left=left_bd + 1*box_h + 1*h_space, right=left_bd + 2*box_h + 1*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
#    box_sel_a  = dict(left=left_bd + 1*box_h + 1*h_space, right=left_bd + 2*box_h + 1*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
#    box_traj_d = dict(left=left_bd + 2*box_h + 2*h_space, right=left_bd + 3*box_h + 2*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
#    box_sel_d  = dict(left=left_bd + 2*box_h + 2*h_space, right=left_bd + 3*box_h + 2*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)

    left_bd   = 0.08
    right_bd  = 0.97
    h_space   = 0.15
    box_h     = 0.22
    top_bd    = 0.92
    bottom_bd = 0.15
    v_space   = 0.10
    box_v     = (top_bd - bottom_bd - 1*v_space)/2
    
    box_null   = dict(left=left_bd, right=left_bd + box_h, top=top_bd, bottom=bottom_bd)
    box_traj_d = dict(left=left_bd + box_h + h_space, right=right_bd, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_sel_d  = dict(left=left_bd + box_h + h_space, right=right_bd, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    
    ## a -- null distribution of selection coefficients

    gs_null = gridspec.GridSpec(1, 1, **box_null)
    ax_null = plt.subplot(gs_null[0, 0])
    
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=1, edgecolor='none',
                     orientation='vertical', log=True)
                     
    pprops = { 'xlim':        [-0.04, 0.045],
               'xticks':      [-0.04, -0.02, 0., 0.02, 0.04, 0.045],
               'xticklabels': [r'$-4$', r'$-2$', r'$0$', r'$2$', r'$4$', ''],
               'ylim':        [1e0, 1e6],
               'yticks':      [1e0, 1e2, 1e4, 1e6],
               #'yticklabels': [0, 10, 20, 30, 40, 50],
               'logy':        True,
               #'ylabel':      'Counts',
               #'xlabel':      'Inferred selection coefficients for neutral variants (%)',
               'bins':        np.arange(-0.04, 0.045+0.001, 0.001),
               'weights':     [np.ones_like(inferred_null)],
               'plotprops':   histprops,
               'axoffset':    0.1,
               'theme':       'open' }

    x = [inferred_null]
    mp.plot(type='hist', ax=ax_null, x=x, colors=[LCOLOR], **pprops)
    #ax_null.set_yscale('log', nonpositive='clip')
    
    # b/c -- Alpha/Delta frequency and inferred selection
    
#    gs_traj_a = gridspec.GridSpec(1, 1, **box_traj_a)
#    ax_traj_a = plt.subplot(gs_traj_a[0, 0])
    gs_traj_d = gridspec.GridSpec(1, 1, **box_traj_d)
    ax_traj_d = plt.subplot(gs_traj_d[0, 0])
    
    ylaba = 'Alpha frequency\nin London (%)'
    ylabd = 'Delta frequency\nin Great Britain (%)'
    
#    gs_sel_a = gridspec.GridSpec(1, 1, **box_sel_a)
#    ax_sel_a = plt.subplot(gs_sel_a[0, 0])
    gs_sel_d = gridspec.GridSpec(1, 1, **box_sel_d)
    ax_sel_d = plt.subplot(gs_sel_d[0, 0])
    
    ylima   = [0, 0.75]
    yticka  = [0, 0.25, 0.50, 0.75]
    ytickla = [int(i) for i in 100*np.array(yticka)]
    ymticka = []
    ylimd   = [0, 0.75]
    ytickd  = [0, 0.25, 0.50, 0.75]
    ytickld = [int(i) for i in 100*np.array(yticka)]
    ymtickd = []
    
    for ax_traj, ax_sel, temp_df, t_det, ylabel, ylim, yticks, yticklabels, yminorticks, v_color in [
        #[ax_traj_a, ax_sel_a, df_alpha, t_alpha, ylaba, ylima, yticka, ytickla, ymticka, C_ALPHA],
        [ax_traj_d, ax_sel_d, df_delta, t_delta, ylabd, ylimd, ytickd, ytickld, ymtickd, C_DELTA] ]:
    
        ## b/c.1 -- frequency trajectory
        
        times   = np.array(temp_df['time'])
        freqs   = np.array(temp_df['frequency'])
        sels    = np.array(temp_df['selection'])
        
        lineprops = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 1 }
        dashprops = {'lw': 2*SIZELINE, 'ls': ':', 'alpha': 0.5 }
        
        xticks      = np.arange(np.min(times), np.max(times), 14)
        xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
        xticklabels = [f'{i.month}/{i.day}' for i in xticklabels]
        
        pprops = { 'xlim':        [np.min(times), np.max(times)],
                   'xticks':      xticks,
                   'xticklabels': ['' for xtl in xticklabels],
                   'ylim':        [0, 1.05],
                   'yticks':      [0, 0.5, 1],
                   'yticklabels': [0, 50, 100],
                   'yminorticks': [0.25, 0.75],
                   'xlabel':      '',
                   #'ylabel':      ylabel,
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times]
        ydat = [freqs]
        mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[v_color], plotprops=lineprops, **pprops)
        
        tobs = 0
        for i in range(len(ydat[0])):
            if ydat[0][i]>0:
                tobs = xdat[0][i]
                break
        print('First observed at %d, detected selection at %d\n' % (tobs, t_det))

        xdat = [[t_det, t_det]]
        ydat = [[0, 1]]
        mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        ## b/c.2 -- inferred selection coefficient

        pprops = { 'xlim':        [np.min(times), np.max(times)],
                   'xticks':      xticks,
                   'xticklabels': xticklabels,
                   'ylim':        ylim,
                   'yticks':      yticks,
                   'yticklabels': yticklabels,
                   'yminorticks': yminorticks,
                   #'xlabel':      'Date',
                   #'ylabel':      'Inferred selection\ncoefficient (%)',
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times]
        ydat = [sels]
        mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[LCOLOR], plotprops=lineprops, **pprops)

        xdat = [[t_det, t_det]]
        ydat = [[0, 1]]
        mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
        ax_traj.text(t_det+1.5, 0.98, 'Variant first reliably\ninferred to be\nbeneficial', **txtprops)

    ## legends and labels
    
#    dx = -0.06
#    dy =  0.01
#    ax_null.text(  box_null['left']   + dx, box_null['top']   + dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
#    ax_traj_a.text(box_traj_a['left'] + dx, box_traj_a['top'] + dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
#    ax_traj_d.text(box_traj_d['left'] + dx, box_traj_d['top'] + dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/fig-early-delta%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    

def plot_early_detection_old(null_file, alpha_file, alpha_start, alpha_end, delta_file, delta_start, delta_end, omicron_file, omicron_start, omicron_end):
    """ Plot the null distribution for mutations inferred to be nearly neutral.
        Plot frequencies and early detection times for Alpha and Delta variants. """
    
    # Load data
    inferred_null = np.loadtxt(null_file)
    df_alpha      = pd.read_csv(alpha_file, comment='#', memory_map=True)
    df_alpha      = df_alpha[(df_alpha['time']>=alpha_start) & (df_alpha['time']<=alpha_end)]
    df_delta      = pd.read_csv(delta_file, comment='#', memory_map=True)
    df_delta      = df_delta[(df_delta['time']>=delta_start) & (df_delta['time']<=delta_end)]
    df_omicron    = pd.read_csv(omicron_file, comment='#', memory_map=True)
    df_omicron    = df_omicron[(df_omicron['time']>=omicron_start) & (df_omicron['time']<=omicron_end)]
    
    # Null distribution statistics
    max_null    = np.amax(inferred_null)
    min_null    = np.amin(inferred_null)
    num_largest = int(round(len(inferred_null) * 2.5 / 100))    # the number of coefficients in the top 2.5% of the null distribution
    soft_cutoff = np.sort(inferred_null)[::-1][num_largest]
    print('hard w cutoff = %.4f' % max_null)
    print('soft w cutoff = %.4f\n' % soft_cutoff)
    
    # Get early detection times
    
    t_alpha        = df_alpha[df_alpha['selection']>max_null].iloc[0]['time']
    f_alpha        = df_alpha[df_alpha['selection']>max_null].iloc[0]['frequency']
    t_soft_alpha   = df_alpha[df_alpha['selection']>soft_cutoff].iloc[0]['time']
    f_soft_alpha   = df_alpha[df_alpha['selection']>soft_cutoff].iloc[0]['frequency']
    t_delta        = df_delta[df_delta['selection']>max_null].iloc[0]['time']
    f_delta        = df_delta[df_delta['selection']>max_null].iloc[0]['frequency']
    t_soft_delta   = df_delta[df_delta['selection']>soft_cutoff].iloc[0]['time']
    f_soft_delta   = df_delta[df_delta['selection']>soft_cutoff].iloc[0]['frequency']
    t_omicron      = df_omicron[df_omicron['selection']>max_null].iloc[0]['time']
    f_omicron      = df_omicron[df_omicron['selection']>max_null].iloc[0]['frequency']
    t_soft_omicron = df_omicron[df_omicron['selection']>soft_cutoff].iloc[0]['time']
    f_soft_omicron = df_omicron[df_omicron['selection']>soft_cutoff].iloc[0]['frequency']
   
    print('time')
    print('cutoff\talpha\tdelta\tomicron')
    print('hard\t%d\t%d\t%d' % (t_alpha, t_delta, t_omicron))
    print('soft\t%d\t%d\t%d' % (t_soft_alpha, t_soft_delta, t_soft_omicron))
    
    print('')
    print('frequency')
    print('cutoff\talpha\tdelta\tomicron')
    print('hard\t%.3f\t%.3f\t%.3f' % (f_alpha, f_delta, f_omicron))
    print('soft\t%.3f\t%.3f\t%.3f' % (f_soft_alpha, f_soft_delta, f_soft_omicron))
    print('')
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 2
    fig   = plt.figure(figsize=(w, goldh))

    left_bd   = 0.08
    right_bd  = 0.97
    h_space   = 0.10
    box_h     = (right_bd - left_bd - 2*h_space)/3
    top_bd    = 0.94
    bottom_bd = 0.13
    v_space   = 0.10
    box_v     = (top_bd - bottom_bd - 2.25*v_space)/3
    
    box_traj_a = dict(left=left_bd + 0*box_h + 0*h_space, right=left_bd + 1*box_h + 0*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_sel_a  = dict(left=left_bd + 0*box_h + 0*h_space, right=left_bd + 1*box_h + 0*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    box_traj_d = dict(left=left_bd + 1*box_h + 1*h_space, right=left_bd + 2*box_h + 1*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_sel_d  = dict(left=left_bd + 1*box_h + 1*h_space, right=left_bd + 2*box_h + 1*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    box_traj_o = dict(left=left_bd + 2*box_h + 2*h_space, right=left_bd + 3*box_h + 2*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_sel_o  = dict(left=left_bd + 2*box_h + 2*h_space, right=left_bd + 3*box_h + 2*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    box_null   = dict(left=left_bd + 0*box_h + 0*h_space, right=left_bd + 1.5*box_h + 1*h_space, top=top_bd - 2*box_v - 2.25*v_space, bottom=top_bd - 3*box_v - 2.25*v_space)
    
    ## d -- null distribution of selection coefficients

    gs_null = gridspec.GridSpec(1, 1, **box_null)
    ax_null = plt.subplot(gs_null[0, 0])
    
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=1, edgecolor='none',
                     orientation='vertical', log=True)
                     
#    pprops = { 'xlim':        [-0.04, 0.045],
#               'xticks':      [-0.04, -0.02, 0., 0.02, 0.04, 0.045],
#               'xticklabels': [r'$-4$', r'$-2$', r'$0$', r'$2$', r'$4$', ''],
#               'ylim':        [1e0, 1e6],
#               'yticks':      [1e0, 1e2, 1e4, 1e6],
#               #'yticklabels': [0, 10, 20, 30, 40, 50],
#               'logy':        True,
#               'ylabel':      'Counts',
#               'xlabel':      'Inferred selection coefficients for neutral variants, ' + r'$\hat{w}$' + ' (%)',
#               'bins':        np.arange(-0.04, 0.045+0.001, 0.001),
#               'weights':     [np.ones_like(inferred_null)],
#               'plotprops':   histprops,
#               'axoffset':    0.1,
#               'theme':       'open' }

    pprops = { 'xlim':        [-0.10, 0.15],
               'xticks':      [-0.10, -0.05, 0., 0.05, 0.10, 0.15],
               'xticklabels': [r'$-10$', r'$-5$', r'$0$', r'$5$', r'$10$', r'$15$'],
               'ylim':        [1e0, 1e6],
               'yticks':      [1e0, 1e2, 1e4, 1e6],
               #'yticklabels': [0, 10, 20, 30, 40, 50],
               'logy':        True,
               'ylabel':      'Counts',
               'xlabel':      'Inferred regional selection coefficients for variants\nwith global selection coefficients inferred <10%, ' + r'$\hat{w}$' + ' (%)',
               'bins':        np.arange(-0.10, 0.15+0.0025, 0.0025),
               'weights':     [np.ones_like(inferred_null)],
               'plotprops':   histprops,
               'axoffset':    0.1,
               'theme':       'open' }

    x = [inferred_null]
    mp.plot(type='hist', ax=ax_null, x=x, colors=[LCOLOR], **pprops)
    
    # a/b/c -- Alpha/Delta/Omicron frequency and inferred selection
    
    gs_traj_a = gridspec.GridSpec(1, 1, **box_traj_a)
    ax_traj_a = plt.subplot(gs_traj_a[0, 0])
    gs_traj_d = gridspec.GridSpec(1, 1, **box_traj_d)
    ax_traj_d = plt.subplot(gs_traj_d[0, 0])
    gs_traj_o = gridspec.GridSpec(1, 1, **box_traj_o)
    ax_traj_o = plt.subplot(gs_traj_o[0, 0])
    
#    ylaba = 'Alpha frequency\nin London (%)'
#    ylabd = 'Delta frequency\nin Great Britain (%)'
#    ylabo = 'Omicron frequency\nin South Africa (%)'

    ylaba = 'Frequency in\nLondon (%)'
    ylabd = 'Frequency in\nGreat Britain (%)'
    ylabo = 'Frequency in\nSouth Africa (%)'
    
    gs_sel_a = gridspec.GridSpec(1, 1, **box_sel_a)
    ax_sel_a = plt.subplot(gs_sel_a[0, 0])
    gs_sel_d = gridspec.GridSpec(1, 1, **box_sel_d)
    ax_sel_d = plt.subplot(gs_sel_d[0, 0])
    gs_sel_o = gridspec.GridSpec(1, 1, **box_sel_o)
    ax_sel_o = plt.subplot(gs_sel_o[0, 0])
    
    ylima   = ylimd   = ylimo   = [0, 0.80]
    yticka  = ytickd  = yticko  = [0, 0.40, 0.80]
    ytickla = ytickld = yticklo = [int(i) for i in 100*np.array(yticka)]
    ymticka = ymtickd = ymticko = [0.20, 0.60]
#    ylimd   = [0, 0.80]
#    ytickd  = [0, 0.30, 0.60]
#    ytickld = [int(i) for i in 100*np.array(ytickd)]
#    ymtickd = [0.15, 0.45]
#    ylimo   = [0, 0.60]
#    yticko  = [0, 0.30, 0.60]
#    yticklo = [int(i) for i in 100*np.array(ytickd)]
#    ymticko = [0.15, 0.45]
    
    for ax_traj, ax_sel, temp_df, t_det, ylabel, ylim, yticks, yticklabels, yminorticks, v_color, v_name in [
        [ax_traj_a, ax_sel_a, df_alpha,   t_alpha,   ylaba, ylima, yticka, ytickla, ymticka, C_ALPHA, 'Alpha'],
        [ax_traj_d, ax_sel_d, df_delta,   t_delta,   ylabd, ylimd, ytickd, ytickld, ymtickd, C_DELTA, 'Delta'],
        [ax_traj_o, ax_sel_o, df_omicron, t_omicron, ylabo, ylimo, yticko, yticklo, ymticko, C_OMICRON, 'Omicron'] ]:
    
        ## a/b/c.1 -- frequency trajectory
        
        times   = np.array(temp_df['time'])
        freqs   = np.array(temp_df['frequency'])
        sels    = np.array(temp_df['selection'])
        
        lineprops = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 1 }
        dashprops = {'lw': 2*SIZELINE, 'ls': ':', 'alpha': 0.5 }
        
        xticks      = np.arange(np.min(times), np.max(times), 14)
        xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
        yearlabel   = xticklabels[0].strftime('%Y')
        xticklabels = [i.strftime('%-d %b') for i in xticklabels]
        #xticklabels = [f'{i.month}/{i.day}' for i in xticklabels]
        
        pprops = { 'xlim':        [np.min(times), np.max(times)],
                   'xticks':      xticks,
                   'xticklabels': ['' for xtl in xticklabels],
                   'ylim':        [0, 1.05],
                   'yticks':      [0, 0.5, 1],
                   'yticklabels': [0, 50, 100],
                   'yminorticks': [0.25, 0.75],
                   'xlabel':      '',
                   'ylabel':      ylabel,
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times]
        ydat = [freqs]
        mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[v_color], plotprops=lineprops, **pprops)
        
        tobs = 0
        for i in range(len(ydat[0])):
            if ydat[0][i]>0:
                tobs = xdat[0][i]
                break
        print('First observed at %d, detected selection at %d\n' % (tobs, t_det))

        xdat = [[t_det, t_det]]
        ydat = [[0, 1]]
        mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        ## a/b/c.2 -- inferred selection coefficient

        pprops = { 'xlim':        [np.min(times), np.max(times)],
                   'xticks':      xticks,
                   'xticklabels': xticklabels,
                   'ylim':        ylim,
                   'yticks':      yticks,
                   'yticklabels': yticklabels,
                   'yminorticks': yminorticks,
                   'xlabel':      yearlabel,
                   'ylabel':      'Inferred selection coefficient\nfor novel SNVs, ' + r'$\hat{w}$' + ' (%)',
#                   'ylabel':      'Inferred selection coefficient\nfor novel %s SNVs, ' % v_name + r'$\hat{w}$' + ' (%)',
#                   'ylabel':      'Inferred selection\ncoefficient for novel\n%s SNVs, ' % v_name + r'$\hat{w}$' + ' (%)',
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times]
        ydat = [sels]
        mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[LCOLOR], plotprops=lineprops, **pprops)

        xdat = [[t_det, t_det]]
        ydat = [[0, 1]]
        mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        if v_name=='Alpha':
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_traj.text(t_det+1.5, 0.98, 'Variant first inferred\nto be significantly\nbeneficial\n%s' % t_det2string, **txtprops)
        else:
            t_det2string = (dt.timedelta(days=t_det) + dt.datetime(2020,1,1)).strftime('%-d %b')
            txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
            ax_traj.text(t_det+1.5, 0.98, '%s' % t_det2string, **txtprops)
            
        txtprops = dict(ha='right', va='bottom', color=v_color, family=FONTFAMILY, size=SIZELABEL, weight='bold')
        ax_sel.text(xticks[-1], 0.01, '%s' % v_name, **txtprops)

    ## legends and labels
    
    dx = -0.06
    dy =  0.01
    ax_null.text(  box_null['left']   + dx, box_null['top']   + dy, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_traj_a.text(box_traj_a['left'] + dx, box_traj_a['top'] + dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_traj_d.text(box_traj_d['left'] + dx, box_traj_d['top'] + dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_traj_o.text(box_traj_o['left'] + dx, box_traj_o['top'] + dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/fig-4%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_a222v_old(trajectory_file, variant_id, selection_file):
    """ Map out mutations in the major variants, their aggregate inferred selection coefficients, and
        frequency trajectories across regions. """
        
    # Load data
    df_traj = pd.read_csv(trajectory_file, comment='#', memory_map=True)
    df_sel  = pd.read_csv(selection_file, comment='#', memory_map=True)
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.8
    fig   = plt.figure(figsize=(w, goldh))

    scale   = 0.83 / (2 * 0.20 + 0.03)
    box_h   = 0.20 * scale
    box_dh  = 0.03 * scale
    box_top = 0.95

    box_traj = dict(left=0.15, right=0.95, top=box_top - 0*box_h - 0*box_dh, bottom=box_top - 1*box_h - 0*box_dh)
    box_sel  = dict(left=0.15, right=0.95, top=box_top - 1*box_h - 1*box_dh, bottom=box_top - 2*box_h - 0*box_dh)
    
    ## a -- A222V trajectories
    
    gs_traj = gridspec.GridSpec(1, 1, **box_traj)
    ax_traj = plt.subplot(gs_traj[0, 0])
    
    legendprops = { 'loc': 4, 'frameon': False, 'scatterpoints': 1, 'handletextpad': 0.1,
                    'prop': {'size' : SIZELABEL}, 'ncol' : 1 }
    lineprops   = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.5 }
    
    t_max  = 600
    pprops = { 'colors':   [C_A222V],
               'xlim':     [61, t_max],
               'xticks':   [],
               'ylim':     [-0.05,  1.05],
               'yticks':   [0, 1],
               'ylabel':   'S:A22V\nfrequency',
               'axoffset': 0.1,
               'hide':     ['bottom'],
               'theme':    'open' }
    
    df_temp = df_traj[df_traj.variant_names==variant_id]
    for k in range(len(df_temp)):
        entry = df_temp.iloc[k]
        times = np.array(str(entry.times).split(), float)
        freqs = np.array(str(entry.frequencies).split(), float)
        if k==len(df_temp)-1:
            mp.plot(type='line', ax=ax_traj, x=[times], y=[freqs], plotprops=lineprops, **pprops)
        else:
            mp.line(             ax=ax_traj, x=[times], y=[freqs], plotprops=lineprops, **pprops)
    
    ## b -- inferred selection coefficient

    gs_sel = gridspec.GridSpec(1, 1, **box_sel)
    ax_sel = plt.subplot(gs_sel[0, 0])
    
    xticks      = [      61,      153,      245,      336,      426,      518, 600]
    xticklabels = ['Mar 20', 'Jun 20', 'Sep 20', 'Dec 20', 'Mar 21', 'Jun 21',  '']
    lineprops   = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 1 }
    
    pprops = { 'xlim':        [61, t_max],
               'xticks':      xticks,
               'xticklabels': xticklabels,
               'ylim':        [0, 0.04],
               'yticks':      [0, 0.02, 0.04],
               'yticklabels': [0, 2, 4],
               #'xlabel':      'Date',
               'ylabel':      'Inferred selection\ncoefficient in Great Britain, ' + r'$\hat{s}$' + ' (%)',
               'axoffset':    0.1,
               'theme':       'open' }
    
    xdat = [np.array(df_sel['time'])]
    ydat = [np.array(df_sel['selection'])]
    mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[LCOLOR], plotprops=lineprops, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/fig-4%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_early_detection_older(inf_file, tv_file, link_file, full_tv_file=None, start=355, end_time=370, window_size=10):
    """ Plots the trajectory and the inferred coefficient over time for the B.1.1.7 group. Also compares to the null distribution."""
    
    site      = 'S-143-2'
    inf_data  = np.load(inf_file, allow_pickle=True)
    inferred  = inf_data['selection']
    alleles   = inf_data['allele_number']
    labels    = [GET_LABEL(i) for i in alleles]
    traj      = inf_data['traj']
    muts      = inf_data['mutant_sites']
    times     = inf_data['times']
    locs      = inf_data['locations']
    tv_data   = np.load(tv_file, allow_pickle=True)
    
    if False:
        traj  = inf_data['traj_nosmooth']
        times = inf_data['times_full']

    # Finding the B.1.1.7 trajectory
    site_traj  = []
    site_times = []
    for i in range(len(traj)):
        muts_temp = [GET_LABEL(muts[i][j]) for j in range(len(muts[i]))]
        if site in muts_temp:
            site_traj.append(traj[i][:, muts_temp.index(site)])
            site_times.append(times[i])

    # Finding the inferred coefficient for the linked group over all time
    labels_link, inf_link, err_link = CALCULATE_LINKED_COEFFICIENTS(tv_file, link_file, tv=True)
    site_index = []
    for i in range(len(labels_link)):
        labels_temp = [labels_link[i][j] for j in range(len(labels_link[i]))]
        if site in labels_temp:
            site_index.append(i)
    if len(site_index)>1:
        print('site appears in multiple places', site_index)
    site_inf = inf_link[:, site_index[0]]
    
    # getting the times corresponding to the inference for the region of interest
    if 'times_all' in tv_data:
        times_all  = tv_data['times_all']    # the time corresponding to the inference. Will be cut to be between start and end_time.
        start_time = times_all[0]
    else:
        start_time = 14
        times_all  = np.arange(start_time, start_time + len(site_inf))
        
    if start != start_time:
        site_inf   = site_inf[list(times_all).index(start):]
        times_all  = times_all[list(times_all).index(start):]
        #times_all  = times_all[start-start_time:]
        full_inf   = inf_link[:, site_index[0]]
        #site_inf   = site_inf[start-start_time:]
        start_time = start
        
    if end_time != times_all[-1]:
        d_t        = times_all[-1] - times_all[list(times_all).index(end_time)]
        site_inf   = site_inf[:list(times_all).index(end_time)]
        times_all  = times_all[:list(times_all).index(end_time)]
        
    # getting the times corresponding to the inference for the full time series
    if 'times_all' in tv_data:
        full_times = tv_data['times_all']    # the time corresponding to the full inference
        start_full = full_times[0]
    else:
        start_full = 14
        full_times = np.arange(start_full, start_full + len(full_inf))
    
    # find null distribution
    if not full_tv_file:
        inferred_null = FIND_NULL_DISTRIBUTION(tv_file, link_file)
    else:
        inferred_null = FIND_NULL_DISTRIBUTION(full_tv_file, link_file)
    max_null    = np.amax(inferred_null)
    min_null    = np.amin(inferred_null)
    num_largest = int(round(len(inferred_null) * 2.5 / 100))    # the number of coefficients in the top 2.5% of the null distribution
    soft_cutoff = np.sort(inferred_null)[::-1][num_largest]
    print('soft cutoff =', soft_cutoff)
    
    ### Find when the inferred coefficient exceeds the null distribution
    locs_sep     = [i.split('-')[3] for i in locs]
    #print(np.array(site_times)[np.array(locs_sep)=='lond'])
    uk_index     = locs_sep.index('lond')
    #uk_index     = np.argmax([np.amax(i) for i in site_traj])
    uk_traj      = site_traj[uk_index][:-d_t]
    uk_traj_full = site_traj[uk_index]
    uk_time      = site_times[uk_index]
    #print(uk_time)
    
    # Trying to fix time interval for trajectory
    if True:
        uk_time_new  = np.array(uk_time) + int(window_size/2)
        uk_cutoff    = uk_time_new[list(uk_time_new).index(start_time): list(uk_time_new).index(end_time)+1]
        uk_time      = uk_cutoff
        uk_traj      = uk_traj_full[list(uk_time_new).index(start_time): list(uk_time_new).index(end_time)+1]
    inf_detected  = site_inf[site_inf>max_null][0]
    t_detected    = times_all[site_inf>max_null][0]
    t_soft_detect = times_all[site_inf>soft_cutoff][0]
    #traj_detected = uk_time[list(uk_time).index(t_detected)]
    delta_t = times_all[-1] - t_detected + 1
    soft_delta_t = times_all[-1] - t_soft_detect + 1
    traj_detected = uk_traj[-delta_t]
    #print(t_detected)
    print('hard detection time:', uk_time[-delta_t])
    print('soft detection time:', uk_time[-soft_delta_t])
    print('frequency at detected time:', traj_detected)

    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w
    fig   = plt.figure(figsize=(w, goldh))

    box_null = dict(left=0.15, right=0.95, bottom=0.70, top=0.95)
    box_traj = dict(left=0.15, right=0.95, bottom=0.35, top=0.55)
    box_sel  = dict(left=0.15, right=0.95, bottom=0.12, top=0.32)
    
    ## a -- null distribution of selection coefficients

    gs_null = gridspec.GridSpec(1, 1, **box_null)
    ax_null = plt.subplot(gs_null[0, 0])
    
    print(np.max(inferred_null), len(inferred_null))
    
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.7, edgecolor='none',
                     orientation='vertical', log=True)
                     
    pprops = { 'xlim':        [-0.02, 0.02],
               'xticks':      [-0.02, -0.01, 0., 0.01, 0.02],
               'xticklabels': [r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'],
               'ylim':        [1e0, 1e4],
               'yticks':      [1e0, 1e1, 1e2, 1e3, 1e4],
               #'yticklabels': [0, 10, 20, 30, 40, 50],
               'logy':        True,
               'ylabel':      'Counts',
               'xlabel':      'Inferred selection coefficients for neutral variants (%)',
               'bins':        np.arange(-0.02, 0.02+0.001, 0.001),
               'weights':     [np.ones_like(inferred_null)],
               'plotprops':   histprops,
               'axoffset':    0.1,
               'theme':       'open' }

    x = [inferred_null]
    mp.plot(type='hist', ax=ax_null, x=x, colors=[LCOLOR], **pprops)
    #ax_null.set_yscale('log', nonpositive='clip')
    
    ## b -- frequency trajectory

    gs_traj = gridspec.GridSpec(1, 1, **box_traj)
    ax_traj = plt.subplot(gs_traj[0, 0])
    
    lineprops = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 1 }
    dashprops = {'lw': 2*SIZELINE, 'ls': ':', 'alpha': 0.5 }
    
    end_idx = -24
#    end_idx = -66
    
    xticks      = np.arange(full_times[0], full_times[end_idx], 14)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
    xticklabels = [f'{i.month}/{i.day}' for i in xticklabels]
    
    pprops = { 'xlim':        [full_times[0], full_times[end_idx]],
               'xticks':      xticks,
               'xticklabels': ['' for xtl in xticklabels],
               'ylim':        [0, 1.05],
               'yticks':      [0, 0.5, 1],
               'yticklabels': [0, 50, 100],
               'yminorticks': [0.25, 0.75],
#               'ylim':        [0, 0.30],
#               'yticks':      [0, 0.1, 0.2, 0.3],
#               'yticklabels': [0, 10, 20, 30],
#               'yminorticks': [0.05, 0.15, 0.25],
               'xlabel':      '',
               'ylabel':      'Variant\nfrequency (%)',
               'axoffset':    0.1,
               'theme':       'open' }

#    xdat = [np.arange(start_time, start_time + len(site_inf))]
#    ydat = [uk_traj[-len(site_inf):]]
    
    xdat = [full_times[-len(uk_traj_full):end_idx] - int(window_size/2)]
    ydat = [uk_traj_full[:end_idx]]
    mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[COLOR_1], plotprops=lineprops, **pprops)
    
    tdetect = 0
    for i in range(len(ydat[0])):
        if ydat[0][i]>0:
            tdetect = xdat[0][i]
            break
    print('First observed at %d, detected selection at %d' % (tdetect, t_detected))

    xdat = [[t_detected, t_detected]]
    ydat = [[0, 1]]
    mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
    
    ## c -- inferred selection coefficient

    gs_sel = gridspec.GridSpec(1, 1, **box_sel)
    ax_sel = plt.subplot(gs_sel[0, 0])
    
    pprops = { 'xlim':        [full_times[0], full_times[end_idx]],
               'xticks':      xticks,
               'xticklabels': xticklabels,
               'ylim':        [0, 0.47],
               'yticks':      [0, 0.20, 0.40],
               'yticklabels': [0, 20, 40],
               'yminorticks': [0.1, 0.3],
#               'ylim':        [0, 0.22],
#               'yticks':      [0, 0.10, 0.20],
#               'yticklabels': [0, 10, 20],
#               'yminorticks': [0.05, 0.15],
               'xlabel':      'Date',
               'ylabel':      'Inferred selection\ncoefficient (%)',
               'axoffset':    0.1,
               'theme':       'open' }
    
    xdat = [full_times[:end_idx]]
    ydat = [full_inf[:end_idx]]
    mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[COLOR_2], plotprops=lineprops, **pprops)

    xdat = [[t_detected, t_detected]]
    ydat = [[0, 1]]
    mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)

    ## legends and stuff
    
    ax_null.text(box_null['left']-0.10, box_null['top']+0.01, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_traj.text(box_traj['left']-0.10, box_traj['top']+0.01, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
    ax_traj.text(t_detected+1.5, 0.98, 'Variant first reliably\ninferred to be\nbeneficial', **txtprops)

    # SAVE FIGURE

    plt.savefig('%s/fig-4%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)
    
    
def plot_early_detection_draft(null_file, alpha_file, alpha_start, alpha_end, delta_file, delta_start, delta_end):
    """ Plot the null distribution for mutations inferred to be nearly neutral.
        Plot frequencies and early detection times for Alpha and Delta variants. """
    
    # Load data
    inferred_null = np.loadtxt(null_file)
    df_alpha      = pd.read_csv(alpha_file, comment='#', memory_map=True)
    df_alpha      = df_alpha[(df_alpha['time']>=alpha_start) & (df_alpha['time']<=alpha_end)]
    df_delta      = pd.read_csv(delta_file, comment='#', memory_map=True)
    df_delta      = df_delta[(df_delta['time']>=delta_start) & (df_delta['time']<=delta_end)]
    
    # Null distribution statistics
    max_null    = np.amax(inferred_null)
    min_null    = np.amin(inferred_null)
    num_largest = int(round(len(inferred_null) * 2.5 / 100))    # the number of coefficients in the top 2.5% of the null distribution
    soft_cutoff = np.sort(inferred_null)[::-1][num_largest]
    print('hard w cutoff = %.4f' % max_null)
    print('soft w cutoff = %.4f\n' % soft_cutoff)
    
    # Get early detection times
    
    t_alpha      = df_alpha[df_alpha['selection']>max_null].iloc[0]['time']
    f_alpha      = df_alpha[df_alpha['selection']>max_null].iloc[0]['frequency']
    t_soft_alpha = df_alpha[df_alpha['selection']>soft_cutoff].iloc[0]['time']
    f_soft_alpha = df_alpha[df_alpha['selection']>soft_cutoff].iloc[0]['frequency']
    t_delta      = df_delta[df_delta['selection']>max_null].iloc[0]['time']
    f_delta      = df_delta[df_delta['selection']>max_null].iloc[0]['frequency']
    t_soft_delta = df_delta[df_delta['selection']>soft_cutoff].iloc[0]['time']
    f_soft_delta = df_delta[df_delta['selection']>soft_cutoff].iloc[0]['frequency']
   
    print('time')
    print('cutoff\talpha\tdelta')
    print('hard\t%d\t%d' % (t_alpha, t_delta))
    print('soft\t%d\t%d' % (t_soft_alpha, t_soft_delta))
    
    print('')
    print('frequency')
    print('cutoff\talpha\tdelta')
    print('hard\t%.3f\t%.3f' % (f_alpha, f_delta))
    print('soft\t%.3f\t%.3f' % (f_soft_alpha, f_soft_delta))
    print('')
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 3
    fig   = plt.figure(figsize=(w, goldh))

    left_bd   = 0.08
    right_bd  = 0.97
    h_space   = 0.10
    box_h     = (right_bd - left_bd - 2*h_space)/3
    top_bd    = 0.92
    bottom_bd = 0.15
    v_space   = 0.10
    box_v     = (top_bd - bottom_bd - 1*v_space)/2
    
    box_null   = dict(left=left_bd + 0*box_h + 0*h_space, right=left_bd + 1*box_h + 0*h_space, top=top_bd, bottom=bottom_bd)
    box_traj_a = dict(left=left_bd + 1*box_h + 1*h_space, right=left_bd + 2*box_h + 1*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_sel_a  = dict(left=left_bd + 1*box_h + 1*h_space, right=left_bd + 2*box_h + 1*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    box_traj_d = dict(left=left_bd + 2*box_h + 2*h_space, right=left_bd + 3*box_h + 2*h_space, top=top_bd - 0*box_v - 0*v_space, bottom=top_bd - 1*box_v - 0*v_space)
    box_sel_d  = dict(left=left_bd + 2*box_h + 2*h_space, right=left_bd + 3*box_h + 2*h_space, top=top_bd - 1*box_v - 1*v_space, bottom=top_bd - 2*box_v - 1*v_space)
    
    ## a -- null distribution of selection coefficients

    gs_null = gridspec.GridSpec(1, 1, **box_null)
    ax_null = plt.subplot(gs_null[0, 0])
    
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=1, edgecolor='none',
                     orientation='vertical', log=True)
                     
#    pprops = { 'xlim':        [-0.04, 0.045],
#               'xticks':      [-0.04, -0.02, 0., 0.02, 0.04, 0.045],
#               'xticklabels': [r'$-4$', r'$-2$', r'$0$', r'$2$', r'$4$', ''],
#               'ylim':        [1e0, 1e6],
#               'yticks':      [1e0, 1e2, 1e4, 1e6],
#               #'yticklabels': [0, 10, 20, 30, 40, 50],
#               'logy':        True,
#               'ylabel':      'Counts',
#               'xlabel':      'Inferred selection coefficients for neutral variants, ' + r'$\hat{w}$' + ' (%)',
#               'bins':        np.arange(-0.04, 0.045+0.001, 0.001),
#               'weights':     [np.ones_like(inferred_null)],
#               'plotprops':   histprops,
#               'axoffset':    0.1,
#               'theme':       'open' }

    pprops = { 'xlim':        [-0.10, 0.15],
               'xticks':      [-0.10, -0.05, 0., 0.05, 0.10, 0.15],
               'xticklabels': [r'$-10$', r'$-5$', r'$0$', r'$5$', r'$10$', r'$15$'],
               'ylim':        [1e0, 1e6],
               'yticks':      [1e0, 1e2, 1e4, 1e6],
               #'yticklabels': [0, 10, 20, 30, 40, 50],
               'logy':        True,
               'ylabel':      'Counts',
               'xlabel':      'Inferred selection coefficients for quasi-neutral variants, ' + r'$\hat{w}$' + ' (%)',
               'bins':        np.arange(-0.10, 0.15+0.0025, 0.0025),
               'weights':     [np.ones_like(inferred_null)],
               'plotprops':   histprops,
               'axoffset':    0.1,
               'theme':       'open' }

    x = [inferred_null]
    mp.plot(type='hist', ax=ax_null, x=x, colors=[LCOLOR], **pprops)
    
    # b/c -- Alpha/Delta frequency and inferred selection
    
    gs_traj_a = gridspec.GridSpec(1, 1, **box_traj_a)
    ax_traj_a = plt.subplot(gs_traj_a[0, 0])
    gs_traj_d = gridspec.GridSpec(1, 1, **box_traj_d)
    ax_traj_d = plt.subplot(gs_traj_d[0, 0])
    
    ylaba = 'Alpha frequency\nin London (%)'
    ylabd = 'Delta frequency\nin Great Britain (%)'
    
    gs_sel_a = gridspec.GridSpec(1, 1, **box_sel_a)
    ax_sel_a = plt.subplot(gs_sel_a[0, 0])
    gs_sel_d = gridspec.GridSpec(1, 1, **box_sel_d)
    ax_sel_d = plt.subplot(gs_sel_d[0, 0])
    
    ylima   = [0, 0.60]
    yticka  = [0, 0.30, 0.60]
    ytickla = [int(i) for i in 100*np.array(yticka)]
    ymticka = [0.15, 0.45]
    ylimd   = [0, 0.60]
    ytickd  = [0, 0.30, 0.60]
    ytickld = [int(i) for i in 100*np.array(ytickd)]
    ymtickd = [0.15, 0.45]
    
    for ax_traj, ax_sel, temp_df, t_det, ylabel, ylim, yticks, yticklabels, yminorticks, v_color in [
        [ax_traj_a, ax_sel_a, df_alpha, t_alpha, ylaba, ylima, yticka, ytickla, ymticka, C_ALPHA],
        [ax_traj_d, ax_sel_d, df_delta, t_delta, ylabd, ylimd, ytickd, ytickld, ymtickd, C_DELTA] ]:
    
        ## b/c.1 -- frequency trajectory
        
        times   = np.array(temp_df['time'])
        freqs   = np.array(temp_df['frequency'])
        sels    = np.array(temp_df['selection'])
        
        lineprops = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 1 }
        dashprops = {'lw': 2*SIZELINE, 'ls': ':', 'alpha': 0.5 }
        
        xticks      = np.arange(np.min(times), np.max(times), 14)
        xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
        xticklabels = [f'{i.month}/{i.day}' for i in xticklabels]
        
        pprops = { 'xlim':        [np.min(times), np.max(times)],
                   'xticks':      xticks,
                   'xticklabels': ['' for xtl in xticklabels],
                   'ylim':        [0, 1.05],
                   'yticks':      [0, 0.5, 1],
                   'yticklabels': [0, 50, 100],
                   'yminorticks': [0.25, 0.75],
                   'xlabel':      '',
                   'ylabel':      ylabel,
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times]
        ydat = [freqs]
        mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[v_color], plotprops=lineprops, **pprops)
        
        tobs = 0
        for i in range(len(ydat[0])):
            if ydat[0][i]>0:
                tobs = xdat[0][i]
                break
        print('First observed at %d, detected selection at %d\n' % (tobs, t_det))

        xdat = [[t_det, t_det]]
        ydat = [[0, 1]]
        mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        ## b/c.2 -- inferred selection coefficient

        pprops = { 'xlim':        [np.min(times), np.max(times)],
                   'xticks':      xticks,
                   'xticklabels': xticklabels,
                   'ylim':        ylim,
                   'yticks':      yticks,
                   'yticklabels': yticklabels,
                   'yminorticks': yminorticks,
                   #'xlabel':      'Date',
                   'ylabel':      'Inferred selection\ncoefficient, ' + r'$\hat{w}$' + ' (%)',
                   'axoffset':    0.1,
                   'theme':       'open' }
        
        xdat = [times]
        ydat = [sels]
        mp.line(ax=ax_sel, x=xdat, y=ydat, colors=[LCOLOR], plotprops=lineprops, **pprops)

        xdat = [[t_det, t_det]]
        ydat = [[0, 1]]
        mp.plot(type='line', ax=ax_sel, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
        
        txtprops = dict(ha='left', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
        ax_traj.text(t_det+1.5, 0.98, 'Variant first reliably\ninferred to be\nbeneficial', **txtprops)

    ## legends and labels
    
    dx = -0.06
    dy =  0.01
    ax_null.text(  box_null['left']   + dx, box_null['top']   + dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_traj_a.text(box_traj_a['left'] + dx, box_traj_a['top'] + dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_traj_d.text(box_traj_d['left'] + dx, box_traj_d['top'] + dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/fig-5%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_selection_statistics_old(infer_file, link_file, link_file_all, syn_file, index_file=None, full_tv_file=None, syn_lower_limit=55):
    """ Makes a plot containing the distribution of inferred coefficients and the enrichment of nonsynonymous mutations.
    Combines selection_dist_plot and syn_plot functions.
    correlation_plots determines whether or not the 3 plots containing the correlations are made."""
    
    # loading and processing the data
    data            = np.load(infer_file, allow_pickle=True)
    linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in EVERY region
    linked_anywhere = np.load(link_file_all, allow_pickle=True)    # sites that are fully linked in ANY region
    syn_data        = np.load(syn_file, allow_pickle=True)
    
    allele_number = data['allele_number']
    labels        = [GET_LABEL(i) for i in allele_number]
    types_temp    = syn_data['types']
    type_locs     = syn_data['locations']
    linked_all    = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    inferred      = data['selection']
    labels        = [GET_LABEL(i) for i in data['allele_number']]
    s_link        = np.zeros(len(linked_sites))
    digits        = len(str(len(inferred)))
    print('number of sites:', len(inferred))
    
    mutant_types  = []
    for i in allele_number:
        if i in type_locs:
            mutant_types.append(types_temp[list(type_locs).index(i)])
        else:
            mutant_types.append('unknown')
        
    inferred_new  = []    # the sum of the inferred coefficients for the linked sites
    for i in range(len(linked_sites)):
        for j in range(len(linked_sites[i])):
            if np.any(linked_sites[i][j]==np.array(labels)):
                s_link[i] += inferred[np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
    for i in range(len(labels)):
        if labels[i] not in linked_all:
            inferred_new.append(inferred[i])
    s_link = inferred_new + list(s_link)
    
    # null distribution
    if full_tv_file:
        inferred_null = FIND_NULL_DISTRIBUTION(full_tv_file, link_file)
        
    # find enrichment in nonsynonymous sites
    site_types = []
    for group in linked_anywhere:
        site_types_temp = []
        for site in group:
            if np.any(site == np.array(labels)):
                site_types_temp.append(mutant_types[list(labels).index(site)])
            else:
                site_types_temp.append('null')
        site_types.append(site_types_temp)

    nonsyn_nonlinked  = np.zeros(len(inferred) - 1)
    for i in range(len(inferred)-1):
        nonsyn_nonlinked[i] = NUMBER_NONSYNONYMOUS_NONLINKED(inferred, i+1, linked_anywhere, site_types, labels, mutant_types)
        
    # find the percentage nonsynonymous with selection coefficients between given intervals
    intervals  = np.logspace(-4, 0, num=12)
    percent_ns = []
    for i in range(len(intervals)):
        if i==0:
            idxs_temp  = np.nonzero(np.logical_and(np.fabs(inferred)>0, np.fabs(inferred)<=intervals[i]))
        else:
            idxs_temp  = np.nonzero(np.logical_and(np.fabs(inferred)>intervals[i-1], np.fabs(inferred)<=intervals[i]))
#        if i==0:
#            idxs_temp  = np.nonzero(np.logical_and(inferred>0, inferred<=intervals[i]))
#        else:
#            idxs_temp  = np.nonzero(np.logical_and(inferred>intervals[i-1], inferred<=intervals[i]))
        #idxs_temp  = np.nonzero(np.logical_and(inferred<intervals[i], inferred>=intervals[i+1]))
        sites_temp = np.array(labels)[idxs_temp]
        types_temp = np.array(mutant_types)[idxs_temp]
        num_nonsyn = NUMBER_NONSYNONYMOUS(linked_anywhere, site_types, sites_temp, types_temp)
        percent_ns.append(num_nonsyn)
        
    # change negative values to zeros
    for i in range(len(percent_ns)):
        if percent_ns[i]<0:
            percent_ns[i] = 0
        
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.1
    fig   = plt.figure(figsize=(w, goldh))

#    box_hist = dict(left=0.15, right=0.55, bottom=0.60, top=0.95)
#    box_sns  = dict(left=0.70, right=0.97, bottom=0.60, top=0.95)
#    box_frac = dict(left=0.15, right=0.72, bottom=0.16, top=0.45)

    box_hist = dict(left=0.15, right=0.95, bottom=0.64, top=0.95)
    box_sns  = dict(left=0.15, right=0.46, bottom=0.15, top=0.45)
    box_frac = dict(left=0.64, right=0.95, bottom=0.16, top=0.45)
    
    ## a -- selection coefficient histogram
    
    ### plot histogram
    
    s_link    = np.array(s_link)
    s_pos_log = np.log10(s_link[s_link>0])
    s_neg_log = np.log10(-s_link[s_link<0])
    
    n_hist = 15
    xmin   = -3
    xmax   = 0
    x_neg  = np.linspace(xmin, xmax, num=n_hist)
    x_pos  = np.linspace(xmin, xmax, num=n_hist)
    y_neg  = []
    y_pos  = []
    y_neu  = []
    for i in range(len(x_neg)-1):
        y_neg.append(np.sum((s_neg_log>x_neg[i]) * (s_neg_log<=x_neg[i+1])))
    y_neg.append(np.sum(s_neg_log>x_neg[-1]))
    y_neu.append(np.sum(s_neg_log<x_neg[0]))
    for i in range(len(x_pos)-1):
        y_pos.append(np.sum((s_pos_log>x_pos[i]) * (s_pos_log<=x_pos[i+1])))
    y_pos.append(np.sum(s_pos_log>x_pos[-1]))
    y_neu.append(np.sum(s_pos_log<x_pos[0]))
    
    norm  = 4
    c_neg = [hls_to_rgb(0.02, 1., 0.83)]
    c_pos = [hls_to_rgb(0.58, 1., 0.60)]
    for i in range(len(y_neg)):
        t = np.min([(x_neg[i] + norm)/norm, 1])
        c_neg.append(hls_to_rgb(0.58, 0.53 * t + 1. * (1 - t), 0.60))
        t = np.min([(x_pos[i] + norm)/norm, 1])
        c_pos.append(hls_to_rgb(0.02, 0.53 * t + 1. * (1 - t), 0.83))
        
    
#    wspace  = 0.20
#    gs_hist = gridspec.GridSpec(1, 3, width_ratios=[n_hist*2/3, 2, n_hist], wspace=wspace, **box_hist)
#
#    hist_props = dict(lw=AXWIDTH/2, width=2.4/n_hist, align='edge', edgecolor=[BKCOLOR], orientation='vertical')
#
#    ### negative coefficient subplot
#
#    ax_neg = plt.subplot(gs_hist[0, 0])
#
#    pprops = { 'colors':      [C_DEL_LT],
#               'xlim':        [xmin, -1][::-1],
#               'xticks':      [-3, -2, -1],
#               'xticklabels': [r'$-0.1$', r'$-1$', r'$-10$'],
#               'ylim':        [0, 750],
#               'yticks':      [0, 250, 500, 750],
#               'ylabel':      'Counts',
#               'hide':        ['top', 'right'] }
#
#    mp.plot(type='bar', ax=ax_neg, x=[x_neg], y=[y_neg], plotprops=hist_props, **pprops)
#
#    ### neutral coefficient subplot
#
#    ax_neu = plt.subplot(gs_hist[0, 1])
#
#    pprops = { 'colors':      [C_NEU_LT],
#               'xlim':        [xmin/n_hist, -xmin/n_hist],
#               'xticks':      [0],
#               'ylim':        [0, 750],
#               'yticks':      [],
#               'hide':        ['top', 'right', 'left'],
#               'xaxstart':    xmin/n_hist,
#               'xaxend':      -xmin/n_hist }
#
#    mp.plot(type='bar', ax=ax_neu, x=[[xmin/n_hist, 0]], y=[y_neu], plotprops=hist_props, **pprops)
#
#    ### positive coefficient subplot
#
#    ax_pos = plt.subplot(gs_hist[0, 2])
#
#    pprops = { 'colors':      [C_BEN_LT],
#               'xlim':        [xmin, xmax],
#               'xticks':      [-3, -2, -1, 0],
#               'xticklabels': [r'$0.1$', r'$1$', r'$10$', r'$100$'],
#               'ylim':        [0, 750],
#               'yticks':      [],
#               'hide':        ['top', 'right', 'left'] }
#
#    mp.plot(type='bar', ax=ax_pos, x=[x_pos], y=[y_pos], plotprops=hist_props, **pprops)


    wspace  = 0.20
    gs_hist = gridspec.GridSpec(1, 2, width_ratios=[(n_hist*2/3)+1, n_hist+1], wspace=wspace, **box_hist)
    
    hist_props = dict(lw=SIZELINE/2, width=2.4/n_hist, align='center', edgecolor=[BKCOLOR], orientation='vertical')
    
    ### negative coefficient subplot
    
    ax_neg = plt.subplot(gs_hist[0, 0])
    
    pprops = { 'colors':      c_neg,
               'xlim':        [xmin-4.5/n_hist, -1][::-1],
               'xticks':      [-3-3/n_hist, -2, -1],
               'xticklabels': ['>-0.1', '-1', '-10'],
               'ylim':        [0, 750],
               'yticks':      [0, 250, 500, 750],
               'ylabel':      'Counts',
               'hide':        ['top', 'right'] }
    
    x_neg = [[-3-3/n_hist]] + [[i] for i in x_neg]
    y_neg = [[y_neu[0]]] + [[i] for i in y_neg]
    
    #x_neg = np.array(x_neg) + 3/n_hist
    
    mp.plot(type='bar', ax=ax_neg, x=x_neg, y=y_neg, plotprops=hist_props, **pprops)
    
    ### positive coefficient subplot
    
    ax_pos = plt.subplot(gs_hist[0, 1])
    
    pprops = { 'colors':      c_pos,
               'xlim':        [xmin-4.5/n_hist, xmax],
               'xticks':      [-3-3/n_hist, -2, -1, 0],
               'xticklabels': ['<0.1', '1', '10', '100'],
               'ylim':        [0, 750],
               'yticks':      [],
               'hide':        ['top', 'right', 'left'] }
               
    x_pos = [[-3-3/n_hist]] + [[i] for i in x_pos]
    y_pos = [[y_neu[1]]] + [[i] for i in y_pos]
    
    #x_pos = np.array(x_pos) + 3/n_hist
    
    mp.plot(type='bar', ax=ax_pos, x=x_pos, y=y_pos, plotprops=hist_props, **pprops)

    ## b -- fraction of nonsynonymous mutations by selection coefficient rank
        
    ### plot lines
    
    gs_sns = gridspec.GridSpec(1, 1, **box_sns)
    ax_sns = plt.subplot(gs_sns[0, 0])
    
    lineprops = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 1 }
    dashprops = {'lw': 2*SIZELINE, 'ls': ':', 'alpha': 0.5 }
    
    pprops = { 'xlim':        [1, 5000],
               'ylim':        [0, 1.05],
               'xticks':      [1, 10, 100, 1000, 5000],
               'xticklabels': ['1', '10', '100', '1000', ''],
               'logx':        True,
               'yticks':      [0, 0.5, 1],
               'yticklabels': [0, 50, 100],
               'yminorticks': [0.25, 0.75],
               'xlabel':      'Number of largest\nselection coefficients',
               'ylabel':      'Fraction\nnonsynonymous (%)',
               'axoffset':    0.1,
               'theme':       'open' }

    xdat = [np.arange(len(nonsyn_nonlinked)) + 1]
    ydat = [nonsyn_nonlinked]
    mp.line(ax=ax_sns, x=xdat, y=ydat, colors=[COLOR_2], plotprops=lineprops, **pprops)

    xdat = [np.arange(len(nonsyn_nonlinked)) + 1]
    ydat = [[nonsyn_nonlinked[-1] for i in range(len(nonsyn_nonlinked))]]
    mp.plot(type='line', ax=ax_sns, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
    
    ## c - fraction of nonsynonymous mutations by selection coefficient size
    
    ### plot fractions
    
    gs_frac = gridspec.GridSpec(1, 1, **box_frac)
    ax_frac = plt.subplot(gs_frac[0, 0])
    
    lineprops = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 1, 'drawstyle': 'steps-post'}
    dashprops = {'lw': 2*SIZELINE, 'ls': ':', 'alpha': 0.5 }
    histprops = dict(lw=SIZELINE/2, width=0.28, ls='solid', alpha=0.7, edgecolor=[BKCOLOR], orientation='vertical')
    
    pprops = { #'xlim':        [1e-4, 1e-0],
               #'xticks':      [1e-4, 1e-3, 1e-2, 1e-1, 1e0],
               'xlim':        [-4, 0],
               'xticks':      [-4, -3, -2, -1, 0],
               'xticklabels': ['0.01', '0.1', '1', '10', '100'],
               'logx':        False,
               'ylim':        [0, 1.05],
               'yticks':      [0, 0.5, 1],
               'yticklabels': [0, 50, 100],
               'yminorticks': [0.25, 0.75],
               'xlabel':      'Magnitude of selection\ncoefficients (%)',
               'ylabel':      'Fraction\nnonsynonymous (%)',
               'axoffset':    0.1,
               'theme':       'open' }

    xdat = [np.log10(intervals)]
    ydat = [percent_ns]
#    mp.plot(type='line', ax=ax_frac, x=xdat, y=ydat, colors=[COLOR_1], plotprops=lineprops, **pprops)
    mp.plot(type='bar', ax=ax_frac, x=xdat, y=ydat, colors=[COLOR_2], plotprops=histprops, **pprops)

    print('non-synonymous percentage:', percent_ns)
    print('intervals:', intervals)
    
    # labels and legends
    
    ax_sns.text(box_hist['left']-0.11, box_hist['top']+0.01, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sns.text(box_sns['left'] -0.11, box_sns['top'] +0.01, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sns.text(box_frac['left']-0.11, box_frac['top']+0.01, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    ax_sns.text(box_hist['left'] + (box_hist['right']-box_hist['left'])/2, box_hist['bottom']-0.08, 'Inferred selection coefficients, ' + '$\hat{s}$' + ' (%)',
                ha='center', va='top', transform=fig.transFigure, **DEF_LABELPROPS)
#    ax_sns.text(box_sns['left']  + (box_sns['right'] - box_sns['left'])/2,  box_sns['bottom']-0.08, 'Number of largest\nselection coefficients',
#                ha='center', va='top', transform=fig.transFigure, **DEF_LABELPROPS)
                
#    ax_neg.text(xmin-1.5/n_hist, 740, '$-0.1<\hat{s}$', ha='center', va='bottom', **DEF_LABELPROPS)
#    ax_pos.text(xmin-1.5/n_hist, 710, '$\hat{s}<0.1$', ha='center', va='bottom', **DEF_LABELPROPS)

#    lineprops = { 'lw' : AXWIDTH/2., 'ls' : '-', 'alpha' : 1.0 }
#    pprops    = { 'xlim' : [0, 1], 'xticks' : [], 'ylim' : [0, 1], 'yticks' : [],
#        'hide' : ['top','bottom','left','right'], 'plotprops' : lineprops }
#    txtprops = {'ha' : 'right', 'va' : 'center', 'color' : BKCOLOR, 'family' : FONTFAMILY,
#        'size' : SIZELABEL, 'rotation' : 90, 'transform' : fig.transFigure}
#    ax[0][0].text(box_traj['left']-0.025, box_traj['top'] - (box_traj['top']-box_traj['bottom'])/2., 'Frequency', **txtprops)

#    # Plotting the enrichment of nonsynonymous mutations vs. number of coefficients
#    ax1  = fig.add_subplot(grid[0, 1])
#    ax1.set_ylim(syn_lower_limit, 101)
#    ax1.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW, length=2)
#    for line in ['right',  'top']:
#        ax1.spines[line].set_visible(False)
#    for line in ['left', 'bottom']:
#        ax1.spines[line].set_linewidth(SPINE_LW)
#    sns.lineplot(np.arange(len(nonsyn_nonlinked))+1, 100*nonsyn_nonlinked, ax=ax1, lw=1, color=MAIN_COLOR)
#    ax1.axhline(100*nonsyn_nonlinked[-1], lw=1, color=COMP_COLOR, alpha=0.75)
#    ax1.text(0.25, (100*nonsyn_nonlinked[-1]-syn_lower_limit)/(100-syn_lower_limit) - 0.05, 'Background', fontsize=6, transform=ax1.transAxes, ha='center', va='center')
#    ax1.set_xscale('log')
#    ax1.set_xlim(1, len(nonsyn_nonlinked)+1)
#    ax1.set_ylabel('Nonsynonymous (%)', fontsize=AXES_FONTSIZE)
#    ax1.set_xlabel("Number of largest coefficients", fontsize=AXES_FONTSIZE)
#    print('background:', 100*nonsyn_nonlinked[-1])
    
    # SAVE FIGURE

    plt.savefig('%s/fig-2%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_variant_selection_extended_old(variant_file, trajectory_file, variant_list, variant_names, selection_file, label2ddr=[]):
    """ Map out mutations in the major variants, their aggregate inferred selection coefficients, and
        frequency trajectories across regions. """
        
    # Load data
    df_var = pd.read_csv(variant_file, comment='#', memory_map=True)
    df_sel = 0
    if len(variant_list)>0:
        df_sel = df_var[df_var.variant_names.isin(variant_list)]
        temp_var_list  = list(df_sel.variant_names)
        temp_var_names = []
        for i in range(len(temp_var_list)):
            temp_var_names.append(variant_names[variant_list.index(temp_var_list[i])])
        variant_list  = np.array(temp_var_list)
        variant_names = np.array(temp_var_names)
    else:
        df_sel = df_var[np.fabs(df_var.selection_coefficient)>s_cutoff]
        variant_list  = np.array(list(df_sel.variant_names))
        variant_names = np.array(variant_list)
    
    s_sort = np.argsort(np.fabs(df_sel.selection_coefficient))[::-1]
    print(np.array(np.fabs(df_sel.selection_coefficient))[s_sort])
    print(np.array(df_sel.variant_names)[s_sort])
    
    n_vars        = len(s_sort)
    variant_list  = variant_list[s_sort]
    variant_names = variant_names[s_sort]
    
    df_traj = pd.read_csv(trajectory_file, comment='#', memory_map=True)
    df_traj = df_traj[df_traj.variant_names.isin(variant_list)]
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w       = DOUBLE_COLUMN #SLIDE_WIDTH
    hshrink = 0.5
    goldh   = w * hshrink
    fig     = plt.figure(figsize=(w, goldh))

    box_traj = dict(left=0.06, right=0.45, bottom=0.14, top=0.95)
    box_circ = dict(left=0.95-0.85*hshrink, right=0.95, bottom=0.10, top=0.95)

    gs_traj = gridspec.GridSpec(n_vars, 2, width_ratios=[2, 1], wspace=0.05, **box_traj)
    gs_circ = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_circ)
    
    ### trajectories
    
    ax = [[plt.subplot(gs_traj[i, 0]), plt.subplot(gs_traj[i, 1])] for i in range(n_vars)]
    
    var_colors = sns.husl_palette(n_vars)
    
    legendprops = { 'loc' : 4, 'frameon' : False, 'scatterpoints' : 1, 'handletextpad' : 0.1,
                    'prop' : {'size' : SIZELABEL}, 'ncol' : 1 }
    lineprops   = { 'lw' : SIZELINE, 'linestyle' : '-', 'alpha' : 0.5 }
    #fillprops   = { 'lw' : 0, 'alpha' : 0.3, 'interpolate' : True }
    
    pprops = { 'xticks' : [],
               'yticks' : [],
               'hide'   : ['top', 'bottom', 'left', 'right'],
               'theme'  : 'open' }
    
    t_max = 600
    xticks      = [      61,      153,      245,      336,      426,      518, 600]
    xticklabels = ['Mar 20', 'Jun 20', 'Sep 20', 'Dec 20', 'Mar 21', 'Jun 21',  '']
    
    for i in range(n_vars):
        idx = list(s_sort)[i]
        pprops['colors'] = [var_colors[i]]
        pprops['xlim']   = [   61, t_max]
        pprops['ylim']   = [-0.05,  1.05]
        if (i==n_vars-1):
            pprops['xticks']      = xticks
            pprops['xticklabels'] = xticklabels
            pprops['hide']        = ['top', 'left', 'right']
            pprops['axoffset']    = 0.18
        df_temp = df_traj[df_traj.variant_names==df_sel.iloc[idx].variant_names]
        for k in range(len(df_temp)):
            entry = df_temp.iloc[k]
            times = np.array(str(entry.times).split(), float)
            freqs = np.array(str(entry.frequencies).split(), float)
            if k==len(df_temp)-1:
                mp.plot(type='line', ax=ax[i][0], x=[times], y=[freqs], plotprops=lineprops, **pprops)
            else:
                mp.line(             ax=ax[i][0], x=[times], y=[freqs], plotprops=lineprops, **pprops)
            
        txtprops = dict(ha='left', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
        if variant_names[i]=='B.1':
            ax[i][0].text(101, 0.1, variant_names[i], **txtprops)
        else:
            ax[i][0].text( 61, 0.8, variant_names[i], **txtprops)
    
    txtprops = dict(ha='center', va='center', rotation=90, color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, clip_on=False)
    ax[-1][0].text(box_traj['left']-0.04, box_traj['top']-(box_traj['top']-box_traj['bottom'])/2, 'Frequency', transform=fig.transFigure, **txtprops)
            
    ### selection coefficient estimates
    
    dashlineprops = dict(lw=SIZELINE, ls=':', alpha=0.25, clip_on=False)
    sprops = dict(lw=0, s=9., marker='o')
    
    pprops = { 'yticks': [],
               'xticks': [],
               'hide'  : ['top', 'bottom', 'left', 'right'],
               'theme' : 'open' }

    xticks      = [0, 0.25, 0.5, 0.75, 1.0]
    xticklabels = [0, 25, 50, 75, 100]
    for i in range(n_vars):
        idx = list(s_sort)[i]
        pprops['xlim'] = [   0,   1]
        pprops['ylim'] = [ 0.5, 1.5]
        ydat           = [1]
        xdat           = [df_sel.iloc[idx].selection_coefficient]
        xerr           = [df_sel.iloc[idx]['theoretical_error']]
        if (i==n_vars-1):
            pprops['xticks']      = xticks
            pprops['xticklabels'] = xticklabels
            pprops['xlabel']      = 'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)'
            pprops['hide']        = ['top', 'left', 'right']
            pprops['axoffset']    = 0.3
        mp.plot(type='error', ax=ax[i][1], x=[xdat], y=[ydat], xerr=[xerr], colors=[var_colors[i]], **pprops)
        
    for i in range(len(xticks)):
        xdat = [xticks[i], xticks[i]]
        ydat = [0.5, n_vars+2.0]
        mp.line(ax=ax[-1][1], x=[xdat], y=[ydat], colors=[BKCOLOR], plotprops=dashlineprops)
    
    ## circle plot

    ax_circ = plt.subplot(gs_circ[0, 0])

    df_sel = pd.read_csv(selection_file, comment='#', memory_map=True)
    
    sig_s, sig_site_real, sig_nuc_idx = plot_circle_formatter(ax_circ, df_sel, label2ddr)

    # MAKE LEGEND

    invt = ax_circ.transData.inverted()
    xy1  = invt.transform((0,0))
    xy2  = invt.transform((0,9))
    
    coef_legend_x  = 0.60
    coef_legend_dx = 0.05
    coef_legend_y  = -0.89
    coef_legend_dy = (xy1[1]-xy2[1])

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
    ex_s     = [-0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04]
    show_s   = [    1,     0,     0,     0, 1,    0,    0,    0,    1]
    c_s      = [C_DEL, C_BEN]
    for i in range(len(ex_s)):
        plotprops = dict(lw=0, marker='o', s=np.fabs(ex_s[i])*S_MULT, clip_on=False)
        mp.scatter(ax=ax_circ, x=[[coef_legend_x + i*coef_legend_dx]], y=[[coef_legend_y]], colors=[c_s[ex_s[i]>0]], plotprops=plotprops)
        if show_s[i]:
            ax_circ.text(coef_legend_x + i*coef_legend_dx, coef_legend_y + 0.75*coef_legend_dy, '%d' % (100*ex_s[i]), **txtprops)
    ax_circ.text(coef_legend_x + len(ex_s)*coef_legend_dx/2, coef_legend_y + (2.25*coef_legend_dy), 'Inferred selection\ncoefficient, $\hat{s}$ (%)', **txtprops)
    
    ax[0][0].text(box_traj['left']-0.02, box_traj['top'], 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_circ.text( box_circ['left']-0.02, box_circ['top'], 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/fig-3-circle%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_selection_statistics_extended_old(selection_file_all, selection_file_ns, label2ddr=[]):
    """ Makes a plot containing the distribution of inferred coefficients and the enrichment of nonsynonymous mutations.
    Combines selection_dist_plot and syn_plot functions.
    correlation_plots determines whether or not the 3 plots containing the correlations are made."""
    
    # loading and processing the data
    df_sel_all = pd.read_csv(selection_file_all, comment='#', memory_map=True)
        
    # find enrichment in nonsynonymous sites
    ns_linked = np.array(df_sel_all['synonymous']=='NS') | np.array(df_sel_all['type of linked group']=='NS')
    df_sel_all['ns_linked'] = ns_linked
    
    s_sort = np.argsort(np.fabs(df_sel_all['selection coefficient']))[::-1]
    ns_link_sorted = ns_linked[s_sort]
    print('Top coefficient: %.3f' % df_sel_all.iloc[list(s_sort)[0]]['selection coefficient'])
    
#    s_sort = np.argsort(np.fabs(df_sel_all['total coefficient for linked group']))[::-1]
#    ns_link_sorted = ns_linked[s_sort]
#    print('Top coefficient: %.3f' % df_sel_all.iloc[list(s_sort)[0]]['total coefficient for linked group'])
    
    fraction_ns = np.zeros(len(ns_link_sorted))
    for i in range(len(ns_link_sorted)):
        num = np.sum(ns_link_sorted[:i+1])
        den = i+1
        fraction_ns[i] = num / den
        
    # find the percentage nonsynonymous with selection coefficients between given intervals
    intervals  = np.logspace(-4, -1, num=12)
    percent_ns = []
    for i in range(len(intervals)):
        df_temp = 0
        if i==0:
            df_temp = df_sel_all[np.fabs(df_sel_all['selection coefficient'])<=intervals[i]]
        else:
            df_temp = df_sel_all[(np.fabs(df_sel_all['selection coefficient'])<=intervals[i]) & (np.fabs(df_sel_all['selection coefficient'])>intervals[i-1])]
#        if i==0:
#            df_temp = df_sel_all[np.fabs(df_sel_all['total coefficient for linked group'])<=intervals[i]]
#        else:
#            df_temp = df_sel_all[(np.fabs(df_sel_all['total coefficient for linked group'])<=intervals[i]) & (np.fabs(df_sel_all['total coefficient for linked group'])>intervals[i-1])]
        percent_ns.append(np.sum(df_temp['ns_linked'])/len(df_temp))
        
    # PLOT FIGURE
    
    ## set up figure grid
    
    w       = DOUBLE_COLUMN #SLIDE_WIDTH
    hshrink = 0.5
    goldh   = w * hshrink
    fig     = plt.figure(figsize=(w, goldh))

    box_hist = dict(left=0.07, right=0.45, bottom=0.64, top=0.95)
    box_sns  = dict(left=0.07, right=0.22, bottom=0.15, top=0.45)
    box_frac = dict(left=0.30, right=0.45, bottom=0.16, top=0.45)
    box_circ = dict(left=0.95-0.85*hshrink, right=0.95, bottom=0.10, top=0.95)
    
    ## a -- selection coefficient histogram
    
    ### plot histogram
    
    s_link    = np.array(df_sel_all['selection coefficient'])
    s_pos_log = np.log10(s_link[s_link>0])
    s_neg_log = np.log10(-s_link[s_link<0])
    
    n_hist = 15
    xmin   = -3
    xmax   = -1
    x_neg  = np.linspace(xmin, xmax, num=n_hist)
    x_pos  = np.linspace(xmin, xmax, num=n_hist)
    y_neg  = []
    y_pos  = []
    y_neu  = []
    for i in range(len(x_neg)-1):
        y_neg.append(np.sum((s_neg_log>x_neg[i]) * (s_neg_log<=x_neg[i+1])))
    y_neg.append(np.sum(s_neg_log>x_neg[-1]))
    y_neu.append(np.sum(s_neg_log<x_neg[0]))
    for i in range(len(x_pos)-1):
        y_pos.append(np.sum((s_pos_log>x_pos[i]) * (s_pos_log<=x_pos[i+1])))
    y_pos.append(np.sum(s_pos_log>x_pos[-1]))
    y_neu.append(np.sum(s_pos_log<x_pos[0]))
    
    norm  = 4
    c_neg = [hls_to_rgb(0.02, 1., 0.83)]
    c_pos = [hls_to_rgb(0.58, 1., 0.60)]
    for i in range(len(y_neg)):
        t = np.min([(x_neg[i] + norm)/norm, 1])
        c_neg.append(hls_to_rgb(0.58, 0.53 * t + 1. * (1 - t), 0.60))
        t = np.min([(x_pos[i] + norm)/norm, 1])
        c_pos.append(hls_to_rgb(0.02, 0.53 * t + 1. * (1 - t), 0.83))
        
    wspace  = 0.20
#    gs_hist = gridspec.GridSpec(1, 2, width_ratios=[(n_hist*2/3)+1, n_hist+1], wspace=wspace, **box_hist)
    gs_hist = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=wspace, **box_hist)
    
    hist_props = dict(lw=SIZELINE/2, width=1.5/n_hist, align='center', edgecolor=[BKCOLOR], orientation='vertical')
    
    ### negative coefficient subplot
    
    ax_neg = plt.subplot(gs_hist[0, 0])
    
    pprops = { 'colors':      c_neg,
               'xlim':        [xmin-4.5/n_hist, -1][::-1],
               'xticks':      [-3-3/n_hist, -2, -1],
               'xticklabels': ['>-0.1', '-1', '-10'],
               'logy':        True,
               'ylim':        [1, 10000],
               'yticks':      [1, 10, 100, 1000, 10000],
               'ylabel':      'Counts',
               'hide':        ['top', 'right'] }
    
    x_neg = [[-3-2/n_hist]] + [[i] for i in x_neg]
    y_neg = [[y_neu[0]]] + [[i] for i in y_neg]
    
    mp.plot(type='bar', ax=ax_neg, x=x_neg, y=y_neg, plotprops=hist_props, **pprops)
    
    ### positive coefficient subplot
    
    ax_pos = plt.subplot(gs_hist[0, 1])
    
    pprops = { 'colors':      c_pos,
               'xlim':        [xmin-4.5/n_hist, -1],
               'xticks':      [-3-3/n_hist, -2, -1],
               'xticklabels': ['<0.1', '1', '10'],
               'logy':        True,
               'ylim':        [1, 10000],
               'yticks':      [],
               'hide':        ['top', 'right', 'left'] }
               
    x_pos = [[-3-2/n_hist]] + [[i] for i in x_pos]
    y_pos = [[y_neu[1]]] + [[i] for i in y_pos]
    
    mp.plot(type='bar', ax=ax_pos, x=x_pos, y=y_pos, plotprops=hist_props, **pprops)

    ## b -- fraction of nonsynonymous mutations by selection coefficient rank
        
    ### plot lines
    
    gs_sns = gridspec.GridSpec(1, 1, **box_sns)
    ax_sns = plt.subplot(gs_sns[0, 0])
    
    lineprops = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 1 }
    dashprops = {'lw': 2*SIZELINE, 'ls': ':', 'alpha': 0.5 }
    
    print(len(df_sel_all))
    
    pprops = { 'xlim':        [1, len(df_sel_all)],
               'ylim':        [0.5, 1.05],
               'xticks':      [1, 100, 10000],
               'xminorticks': [10, 1000],
               'xminorticklabels': ['', ''],
               'xticklabels': ['1', '100', '10000'],
               'logx':        True,
               'yticks':      [0.5, 1],
               'yticklabels': [50, 100],
               'yminorticks': [0.75],
               'xlabel':      'Number of largest\nselection coefficients',
               'ylabel':      'Fraction\nnonsynonymous (%)',
               'axoffset':    0.1,
               'theme':       'open' }

    xdat = [np.arange(len(df_sel_all)) + 1]
    ydat = [fraction_ns]
    mp.line(ax=ax_sns, x=xdat, y=ydat, colors=[COLOR_2], plotprops=lineprops, **pprops)

    xdat = [[0, len(df_sel_all) + 1]]
    ydat = [[fraction_ns[-1], fraction_ns[-1]]]
    mp.plot(type='line', ax=ax_sns, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
    
    ## c - fraction of nonsynonymous mutations by selection coefficient size
    
    ### plot fractions
    
    gs_frac = gridspec.GridSpec(1, 1, **box_frac)
    ax_frac = plt.subplot(gs_frac[0, 0])
    
    lineprops = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 1, 'drawstyle': 'steps-post'}
    dashprops = {'lw': 2*SIZELINE, 'ls': ':', 'alpha': 0.5 }
    histprops = dict(lw=SIZELINE/2, width=0.20, ls='solid', alpha=0.7, edgecolor=[BKCOLOR], orientation='vertical')
    
    pprops = { #'xlim':        [1e-4, 1e-0],
               #'xticks':      [1e-4, 1e-3, 1e-2, 1e-1, 1e0],
               'xlim':        [-4.2, -0.8],
               'xticks':      [-4, -3, -2, -1],
               'xticklabels': ['0.01', '0.1', '1', '10'],
               'logx':        False,
               'ylim':        [0, 1.05],
               'yticks':      [0, 0.5, 1],
               'yticklabels': [0, 50, 100],
               'yminorticks': [0.25, 0.75],
               'xlabel':      'Magnitude of selection\ncoefficients (%)',
               'ylabel':      'Fraction\nnonsynonymous (%)',
               'axoffset':    0.1,
               'theme':       'open' }

    xdat = [np.log10(intervals)]
    ydat = [percent_ns]
#    mp.plot(type='line', ax=ax_frac, x=xdat, y=ydat, colors=[COLOR_1], plotprops=lineprops, **pprops)
    mp.plot(type='bar', ax=ax_frac, x=xdat, y=ydat, colors=[COLOR_2], plotprops=histprops, **pprops)

    print('non-synonymous percentage:', percent_ns)
    print('intervals:', intervals)
    
    # labels and legends
    
    ax_sns.text(box_hist['left'] + (box_hist['right']-box_hist['left'])/2, box_hist['bottom']-0.08, 'Inferred selection coefficients, ' + '$\hat{s}$' + ' (%)',
                ha='center', va='top', transform=fig.transFigure, **DEF_LABELPROPS)
    
    ## circle plot

    gs_circ = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_circ)
    ax_circ = plt.subplot(gs_circ[0, 0])

    df_sel = pd.read_csv(selection_file_ns, comment='#', memory_map=True)
    
    sig_s, sig_site_real, sig_nuc_idx = plot_circle_formatter(ax_circ, df_sel, label2ddr)

    # MAKE LEGEND

    invt = ax_circ.transData.inverted()
    xy1  = invt.transform((0,0))
    xy2  = invt.transform((0,9))
    
    coef_legend_x  = 0.60
    coef_legend_dx = 0.05
    coef_legend_y  = -0.89
    coef_legend_dy = (xy1[1]-xy2[1])

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
    ex_s     = [-0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05]
    show_s   = [    1,     0,     0,     0,     0, 1,    0,    0,    0,    0,    1]
    c_s      = [C_DEL, C_BEN]
    for i in range(len(ex_s)):
        plotprops = dict(lw=0, marker='o', s=(np.fabs(ex_s[i]) - DS_NORM)*S_MULT if ex_s[i]!=0 else 0, clip_on=False)
        mp.scatter(ax=ax_circ, x=[[coef_legend_x + i*coef_legend_dx]], y=[[coef_legend_y]], colors=[c_s[ex_s[i]>0]], plotprops=plotprops)
        if show_s[i]:
            ax_circ.text(coef_legend_x + i*coef_legend_dx, coef_legend_y + 0.75*coef_legend_dy, '%d' % (100*ex_s[i]), **txtprops)
    ax_circ.text(coef_legend_x + len(ex_s)*coef_legend_dx/2, coef_legend_y + (2.25*coef_legend_dy), 'Inferred selection\ncoefficient, $\hat{s}$ (%)', **txtprops)
    
    ax_sns.text( box_hist['left']-0.05, box_hist['top']+0.01, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sns.text( box_sns['left'] -0.05, box_sns['top'] +0.01, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sns.text( box_frac['left']-0.05, box_frac['top']+0.01, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_circ.text(box_circ['left']-0.01, box_circ['top']+0.01, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    # SAVE FIGURE

    plt.savefig('%s/fig-2-circle%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_selection_statistics_old(infer_file, link_file, link_file_all, syn_file, index_file=None, full_tv_file=None, syn_lower_limit=55):
    """ Makes a plot containing the distribution of inferred coefficients and the enrichment of nonsynonymous mutations.
    Combines selection_dist_plot and syn_plot functions.
    correlation_plots determines whether or not the 3 plots containing the correlations are made."""
    
    # loading and processing the data
    data            = np.load(infer_file, allow_pickle=True)
    linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in EVERY region
    linked_anywhere = np.load(link_file_all, allow_pickle=True)    # sites that are fully linked in ANY region
    syn_data        = np.load(syn_file, allow_pickle=True)
    
    allele_number = data['allele_number']
    labels        = [GET_LABEL(i) for i in allele_number]
    types_temp    = syn_data['types']
    type_locs     = syn_data['locations']
    linked_all    = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    inferred      = data['selection']
    labels        = [GET_LABEL(i) for i in data['allele_number']]
    s_link        = np.zeros(len(linked_sites))
    digits        = len(str(len(inferred)))
    print('number of sites:', len(inferred))
    
    mutant_types  = []
    for i in allele_number:
        if i in type_locs:
            mutant_types.append(types_temp[list(type_locs).index(i)])
        else:
            mutant_types.append('unknown')
        
    inferred_new  = []    # the sum of the inferred coefficients for the linked sites
    for i in range(len(linked_sites)):
        for j in range(len(linked_sites[i])):
            if np.any(linked_sites[i][j]==np.array(labels)):
                s_link[i] += inferred[np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
    for i in range(len(labels)):
        if labels[i] not in linked_all:
            inferred_new.append(inferred[i])
    s_link = inferred_new + list(s_link)
    
    # null distribution
    if full_tv_file:
        inferred_null = FIND_NULL_DISTRIBUTION(full_tv_file, link_file)
        
    # find enrichment in nonsynonymous sites
    site_types = []
    for group in linked_anywhere:
        site_types_temp = []
        for site in group:
            if np.any(site == np.array(labels)):
                site_types_temp.append(mutant_types[list(labels).index(site)])
            else:
                site_types_temp.append('null')
        site_types.append(site_types_temp)

    nonsyn_nonlinked  = np.zeros(len(inferred) - 1)
    for i in range(len(inferred)-1):
        nonsyn_nonlinked[i] = NUMBER_NONSYNONYMOUS_NONLINKED(inferred, i+1, linked_anywhere, site_types, labels, mutant_types)
        
    # find the percentage nonsynonymous with selection coefficients between given intervals
    intervals  = np.logspace(-4, 0, num=12)
    percent_ns = []
    for i in range(len(intervals)):
        if i==0:
            idxs_temp  = np.nonzero(np.logical_and(np.fabs(inferred)>0, np.fabs(inferred)<=intervals[i]))
        else:
            idxs_temp  = np.nonzero(np.logical_and(np.fabs(inferred)>intervals[i-1], np.fabs(inferred)<=intervals[i]))
#        if i==0:
#            idxs_temp  = np.nonzero(np.logical_and(inferred>0, inferred<=intervals[i]))
#        else:
#            idxs_temp  = np.nonzero(np.logical_and(inferred>intervals[i-1], inferred<=intervals[i]))
        #idxs_temp  = np.nonzero(np.logical_and(inferred<intervals[i], inferred>=intervals[i+1]))
        sites_temp = np.array(labels)[idxs_temp]
        types_temp = np.array(mutant_types)[idxs_temp]
        num_nonsyn = NUMBER_NONSYNONYMOUS(linked_anywhere, site_types, sites_temp, types_temp)
        percent_ns.append(num_nonsyn)
        
    # change negative values to zeros
    for i in range(len(percent_ns)):
        if percent_ns[i]<0:
            percent_ns[i] = 0
        
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.1
    fig   = plt.figure(figsize=(w, goldh))

#    box_hist = dict(left=0.15, right=0.55, bottom=0.60, top=0.95)
#    box_sns  = dict(left=0.70, right=0.97, bottom=0.60, top=0.95)
#    box_frac = dict(left=0.15, right=0.72, bottom=0.16, top=0.45)

    box_hist = dict(left=0.15, right=0.95, bottom=0.64, top=0.95)
    box_sns  = dict(left=0.15, right=0.46, bottom=0.15, top=0.45)
    box_frac = dict(left=0.64, right=0.95, bottom=0.16, top=0.45)
    
    ## a -- selection coefficient histogram
    
    ### plot histogram
    
    s_link    = np.array(s_link)
    s_pos_log = np.log10(s_link[s_link>0])
    s_neg_log = np.log10(-s_link[s_link<0])
    
    n_hist = 15
    xmin   = -3
    xmax   = 0
    x_neg  = np.linspace(xmin, xmax, num=n_hist)
    x_pos  = np.linspace(xmin, xmax, num=n_hist)
    y_neg  = []
    y_pos  = []
    y_neu  = []
    for i in range(len(x_neg)-1):
        y_neg.append(np.sum((s_neg_log>x_neg[i]) * (s_neg_log<=x_neg[i+1])))
    y_neg.append(np.sum(s_neg_log>x_neg[-1]))
    y_neu.append(np.sum(s_neg_log<x_neg[0]))
    for i in range(len(x_pos)-1):
        y_pos.append(np.sum((s_pos_log>x_pos[i]) * (s_pos_log<=x_pos[i+1])))
    y_pos.append(np.sum(s_pos_log>x_pos[-1]))
    y_neu.append(np.sum(s_pos_log<x_pos[0]))
    
    norm  = 4
    c_neg = [hls_to_rgb(0.02, 1., 0.83)]
    c_pos = [hls_to_rgb(0.58, 1., 0.60)]
    for i in range(len(y_neg)):
        t = np.min([(x_neg[i] + norm)/norm, 1])
        c_neg.append(hls_to_rgb(0.58, 0.53 * t + 1. * (1 - t), 0.60))
        t = np.min([(x_pos[i] + norm)/norm, 1])
        c_pos.append(hls_to_rgb(0.02, 0.53 * t + 1. * (1 - t), 0.83))
        
    
#    wspace  = 0.20
#    gs_hist = gridspec.GridSpec(1, 3, width_ratios=[n_hist*2/3, 2, n_hist], wspace=wspace, **box_hist)
#
#    hist_props = dict(lw=AXWIDTH/2, width=2.4/n_hist, align='edge', edgecolor=[BKCOLOR], orientation='vertical')
#
#    ### negative coefficient subplot
#
#    ax_neg = plt.subplot(gs_hist[0, 0])
#
#    pprops = { 'colors':      [C_DEL_LT],
#               'xlim':        [xmin, -1][::-1],
#               'xticks':      [-3, -2, -1],
#               'xticklabels': [r'$-0.1$', r'$-1$', r'$-10$'],
#               'ylim':        [0, 750],
#               'yticks':      [0, 250, 500, 750],
#               'ylabel':      'Counts',
#               'hide':        ['top', 'right'] }
#
#    mp.plot(type='bar', ax=ax_neg, x=[x_neg], y=[y_neg], plotprops=hist_props, **pprops)
#
#    ### neutral coefficient subplot
#
#    ax_neu = plt.subplot(gs_hist[0, 1])
#
#    pprops = { 'colors':      [C_NEU_LT],
#               'xlim':        [xmin/n_hist, -xmin/n_hist],
#               'xticks':      [0],
#               'ylim':        [0, 750],
#               'yticks':      [],
#               'hide':        ['top', 'right', 'left'],
#               'xaxstart':    xmin/n_hist,
#               'xaxend':      -xmin/n_hist }
#
#    mp.plot(type='bar', ax=ax_neu, x=[[xmin/n_hist, 0]], y=[y_neu], plotprops=hist_props, **pprops)
#
#    ### positive coefficient subplot
#
#    ax_pos = plt.subplot(gs_hist[0, 2])
#
#    pprops = { 'colors':      [C_BEN_LT],
#               'xlim':        [xmin, xmax],
#               'xticks':      [-3, -2, -1, 0],
#               'xticklabels': [r'$0.1$', r'$1$', r'$10$', r'$100$'],
#               'ylim':        [0, 750],
#               'yticks':      [],
#               'hide':        ['top', 'right', 'left'] }
#
#    mp.plot(type='bar', ax=ax_pos, x=[x_pos], y=[y_pos], plotprops=hist_props, **pprops)


    wspace  = 0.20
    gs_hist = gridspec.GridSpec(1, 2, width_ratios=[(n_hist*2/3)+1, n_hist+1], wspace=wspace, **box_hist)
    
    hist_props = dict(lw=SIZELINE/2, width=2.4/n_hist, align='center', edgecolor=[BKCOLOR], orientation='vertical')
    
    ### negative coefficient subplot
    
    ax_neg = plt.subplot(gs_hist[0, 0])
    
    pprops = { 'colors':      c_neg,
               'xlim':        [xmin-4.5/n_hist, -1][::-1],
               'xticks':      [-3-3/n_hist, -2, -1],
               'xticklabels': ['>-0.1', '-1', '-10'],
               'ylim':        [0, 750],
               'yticks':      [0, 250, 500, 750],
               'ylabel':      'Counts',
               'hide':        ['top', 'right'] }
    
    x_neg = [[-3-3/n_hist]] + [[i] for i in x_neg]
    y_neg = [[y_neu[0]]] + [[i] for i in y_neg]
    
    #x_neg = np.array(x_neg) + 3/n_hist
    
    mp.plot(type='bar', ax=ax_neg, x=x_neg, y=y_neg, plotprops=hist_props, **pprops)
    
    ### positive coefficient subplot
    
    ax_pos = plt.subplot(gs_hist[0, 1])
    
    pprops = { 'colors':      c_pos,
               'xlim':        [xmin-4.5/n_hist, xmax],
               'xticks':      [-3-3/n_hist, -2, -1, 0],
               'xticklabels': ['<0.1', '1', '10', '100'],
               'ylim':        [0, 750],
               'yticks':      [],
               'hide':        ['top', 'right', 'left'] }
               
    x_pos = [[-3-3/n_hist]] + [[i] for i in x_pos]
    y_pos = [[y_neu[1]]] + [[i] for i in y_pos]
    
    #x_pos = np.array(x_pos) + 3/n_hist
    
    mp.plot(type='bar', ax=ax_pos, x=x_pos, y=y_pos, plotprops=hist_props, **pprops)

    ## b -- fraction of nonsynonymous mutations by selection coefficient rank
        
    ### plot lines
    
    gs_sns = gridspec.GridSpec(1, 1, **box_sns)
    ax_sns = plt.subplot(gs_sns[0, 0])
    
    lineprops = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 1 }
    dashprops = {'lw': 2*SIZELINE, 'ls': ':', 'alpha': 0.5 }
    
    pprops = { 'xlim':        [1, 5000],
               'ylim':        [0, 1.05],
               'xticks':      [1, 10, 100, 1000, 5000],
               'xticklabels': ['1', '10', '100', '1000', ''],
               'logx':        True,
               'yticks':      [0, 0.5, 1],
               'yticklabels': [0, 50, 100],
               'yminorticks': [0.25, 0.75],
               'xlabel':      'Number of largest\nselection coefficients',
               'ylabel':      'Fraction\nnonsynonymous (%)',
               'axoffset':    0.1,
               'theme':       'open' }

    xdat = [np.arange(len(nonsyn_nonlinked)) + 1]
    ydat = [nonsyn_nonlinked]
    mp.line(ax=ax_sns, x=xdat, y=ydat, colors=[COLOR_2], plotprops=lineprops, **pprops)

    xdat = [np.arange(len(nonsyn_nonlinked)) + 1]
    ydat = [[nonsyn_nonlinked[-1] for i in range(len(nonsyn_nonlinked))]]
    mp.plot(type='line', ax=ax_sns, x=xdat, y=ydat, colors=[BKCOLOR], plotprops=dashprops, **pprops)
    
    ## c - fraction of nonsynonymous mutations by selection coefficient size
    
    ### plot fractions
    
    gs_frac = gridspec.GridSpec(1, 1, **box_frac)
    ax_frac = plt.subplot(gs_frac[0, 0])
    
    lineprops = {'lw': 2*SIZELINE, 'ls': '-', 'alpha': 1, 'drawstyle': 'steps-post'}
    dashprops = {'lw': 2*SIZELINE, 'ls': ':', 'alpha': 0.5 }
    histprops = dict(lw=SIZELINE/2, width=0.28, ls='solid', alpha=0.7, edgecolor=[BKCOLOR], orientation='vertical')
    
    pprops = { #'xlim':        [1e-4, 1e-0],
               #'xticks':      [1e-4, 1e-3, 1e-2, 1e-1, 1e0],
               'xlim':        [-4, 0],
               'xticks':      [-4, -3, -2, -1, 0],
               'xticklabels': ['0.01', '0.1', '1', '10', '100'],
               'logx':        False,
               'ylim':        [0, 1.05],
               'yticks':      [0, 0.5, 1],
               'yticklabels': [0, 50, 100],
               'yminorticks': [0.25, 0.75],
               'xlabel':      'Magnitude of selection\ncoefficients (%)',
               'ylabel':      'Fraction\nnonsynonymous (%)',
               'axoffset':    0.1,
               'theme':       'open' }

    xdat = [np.log10(intervals)]
    ydat = [percent_ns]
#    mp.plot(type='line', ax=ax_frac, x=xdat, y=ydat, colors=[COLOR_1], plotprops=lineprops, **pprops)
    mp.plot(type='bar', ax=ax_frac, x=xdat, y=ydat, colors=[COLOR_2], plotprops=histprops, **pprops)

    print('non-synonymous percentage:', percent_ns)
    print('intervals:', intervals)
    
    # labels and legends
    
    ax_sns.text(box_hist['left']-0.11, box_hist['top']+0.01, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sns.text(box_sns['left'] -0.11, box_sns['top'] +0.01, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sns.text(box_frac['left']-0.11, box_frac['top']+0.01, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    ax_sns.text(box_hist['left'] + (box_hist['right']-box_hist['left'])/2, box_hist['bottom']-0.08, 'Inferred selection coefficients, ' + '$\hat{s}$' + ' (%)',
                ha='center', va='top', transform=fig.transFigure, **DEF_LABELPROPS)
#    ax_sns.text(box_sns['left']  + (box_sns['right'] - box_sns['left'])/2,  box_sns['bottom']-0.08, 'Number of largest\nselection coefficients',
#                ha='center', va='top', transform=fig.transFigure, **DEF_LABELPROPS)
                
#    ax_neg.text(xmin-1.5/n_hist, 740, '$-0.1<\hat{s}$', ha='center', va='bottom', **DEF_LABELPROPS)
#    ax_pos.text(xmin-1.5/n_hist, 710, '$\hat{s}<0.1$', ha='center', va='bottom', **DEF_LABELPROPS)

#    lineprops = { 'lw' : AXWIDTH/2., 'ls' : '-', 'alpha' : 1.0 }
#    pprops    = { 'xlim' : [0, 1], 'xticks' : [], 'ylim' : [0, 1], 'yticks' : [],
#        'hide' : ['top','bottom','left','right'], 'plotprops' : lineprops }
#    txtprops = {'ha' : 'right', 'va' : 'center', 'color' : BKCOLOR, 'family' : FONTFAMILY,
#        'size' : SIZELABEL, 'rotation' : 90, 'transform' : fig.transFigure}
#    ax[0][0].text(box_traj['left']-0.025, box_traj['top'] - (box_traj['top']-box_traj['bottom'])/2., 'Frequency', **txtprops)

#    # Plotting the enrichment of nonsynonymous mutations vs. number of coefficients
#    ax1  = fig.add_subplot(grid[0, 1])
#    ax1.set_ylim(syn_lower_limit, 101)
#    ax1.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW, length=2)
#    for line in ['right',  'top']:
#        ax1.spines[line].set_visible(False)
#    for line in ['left', 'bottom']:
#        ax1.spines[line].set_linewidth(SPINE_LW)
#    sns.lineplot(np.arange(len(nonsyn_nonlinked))+1, 100*nonsyn_nonlinked, ax=ax1, lw=1, color=MAIN_COLOR)
#    ax1.axhline(100*nonsyn_nonlinked[-1], lw=1, color=COMP_COLOR, alpha=0.75)
#    ax1.text(0.25, (100*nonsyn_nonlinked[-1]-syn_lower_limit)/(100-syn_lower_limit) - 0.05, 'Background', fontsize=6, transform=ax1.transAxes, ha='center', va='center')
#    ax1.set_xscale('log')
#    ax1.set_xlim(1, len(nonsyn_nonlinked)+1)
#    ax1.set_ylabel('Nonsynonymous (%)', fontsize=AXES_FONTSIZE)
#    ax1.set_xlabel("Number of largest coefficients", fontsize=AXES_FONTSIZE)
#    print('background:', 100*nonsyn_nonlinked[-1])
    
    # SAVE FIGURE

    plt.savefig('%s/fig-2%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_variant_selection_old(variant_file, trajectory_file, variant_list=[], s_cutoff=0.05):
    """ Map out mutations in the major variants, their aggregate inferred selection coefficients, and
        frequency trajectories across regions. """
        
    # Load data
    df_var = pd.read_csv(variant_file, comment='#', memory_map=True)
    df_sel = 0
    if len(variant_list)>0:
        df_sel = df_var[df_var.variant_names.isin(variant_list)]
    else:
        df_sel = df_var[np.fabs(df_var.selection_coefficient)>s_cutoff]
    
    mut_idxs   = []
    mut_labels = []
    mut_ns     = []
    s_sort = np.argsort(np.fabs(df_sel.selection_coefficient))[::-1]
    print(np.array(np.fabs(df_sel.selection_coefficient))[s_sort])
    for i in s_sort:
        
        # get mutation locations
        temp_idxs = []
        data_idxs = df_sel.iloc[i].nucleotides.split('/')
        for j in range(len(data_idxs)):
            if data_idxs[j]!='not_present':
                temp_idxs.append(int(data_idxs[j]))
            else:
                temp_idxs.append(-9999)
        
        # get mutation labels
        temp_labels = []
        data_sites  = df_sel.iloc[i].sites.split('/')
        data_labels = df_sel.iloc[i].aa_mutations.split('/')
        for j in range(len(data_labels)):
            if data_labels[j]!='not_present' and '>' in data_labels[j]:
                f, l = data_labels[j].split('>')[0], data_labels[j].split('>')[1]
                site = int(data_sites[j].split('-')[1])
                if l=='-' and f!='-':
                    temp_labels.append(r'$\Delta %d$' % site)
                else:
                    temp_labels.append('%s%d%s' % (f, site, l))
            else:
                temp_labels.append('')
        
        # get synonymous/NA (0) or nonsynonymous (1)
        temp_ns = [1 if j=='NS' else 0 for j in df_sel.iloc[i].synonymous.split('/')]
        
        mut_idxs.append(temp_idxs)
        mut_labels.append(temp_labels)
        mut_ns.append(temp_ns)
        
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.1
    fig   = plt.figure(figsize=(w, goldh))

    box_vars = dict(left=0.13, right=0.90, bottom=0.60, top=0.95)
    box_traj = dict(left=0.13, right=0.70, bottom=0.12, top=0.45)

    gs_vars = gridspec.GridSpec(1, 1, **box_vars)
    gs_traj = gridspec.GridSpec(1, 1, **box_traj)
    
    ## a -- map of mutations for important variants and linked groups

    ax_vars = plt.subplot(gs_vars[0, 0])
    
    var_colors = sns.husl_palette(len(mut_idxs))
    
    pprops = { 'xlim':   [0, 30000],
               'ylim':   [-2*len(mut_idxs)-1, 1],
               'xticks': [],
               'yticks': [],
               'theme':  'open',
               'hide':   ['left','right', 'top', 'bottom'] }
            
    for i in range(len(mut_idxs)):
        pprops['facecolor'] = ['None']
        pprops['edgecolor'] = [BKCOLOR]
        sprops = dict(lw=AXWIDTH, s=2., marker='o', alpha=1)
        mp.scatter(ax=ax_vars, x=[mut_idxs[i]], y=[[-2*i for j in range(len(mut_idxs[i]))]], plotprops=sprops, **pprops)
        
#        s_muts = [mut_idxs[i][j] for j in range(len(mut_idxs[i])) if mut_ns[i][j]==0]
#
#        pprops['facecolor'] = [C_NEU_LT]
#        sprops = dict(lw=0, s=2., marker='o', alpha=1)
#        if len(s_muts)>0:
#            mp.scatter(ax=ax_vars, x=[s_muts], y=[[-2*i for j in range(len(s_muts))]], plotprops=sprops, **pprops)
        
        ns_muts = [mut_idxs[i][j] for j in range(len(mut_idxs[i])) if mut_ns[i][j]==1]
        ns_labels = [mut_labels[i][j] for j in range(len(mut_idxs[i])) if mut_ns[i][j]==1]
        
        pprops['facecolor'] = [var_colors[i]]
        sprops = dict(lw=0, s=2., marker='o', alpha=1)
        if i==len(mut_idxs)-1:
            mp.plot(type='scatter', ax=ax_vars, x=[ns_muts], y=[[-2*i for j in range(len(ns_muts))]], plotprops=sprops, **pprops)
        else:
            mp.scatter(ax=ax_vars, x=[ns_muts], y=[[-2*i for j in range(len(ns_muts))]], plotprops=sprops, **pprops)
        
#        # plot data points
#
#        y_vals = np.array(y_vals)
#        y_vals[y_vals<0.5] = 0.5
#
#        pprops['facecolor'] = ['None' for _c1 in range(len(x_ben[k]))]
#        pprops['edgecolor'] = eclist
#        sprops = dict(lw=AXWIDTH, s=2., marker='o', alpha=1.0)
#        temp_x = [[x_vals[_c1] + np.random.normal(0, 0.08) for _c2 in range(len(y_vals[_c1]))] for _c1 in range(len(y_vals))]
#        mp.scatter(ax=ax, x=temp_x, y=y_vals, plotprops=sprops, **pprops)
#
#        pprops['facecolor'] = fclist
#        sprops = dict(lw=0, s=2., marker='o', alpha=1)
#        mp.scatter(ax=ax, x=temp_x, y=y_vals, plotprops=sprops, **pprops)
#
#    dashlineprops = { 'lw' : SIZELINE * 2.0, 'ls' : ':', 'alpha' : 0.5 }
#    sprops = dict(lw=0, s=9., marker='o')
#
#    pprops = { 'yticks': [],
#               'xticks': [],
#               'hide'  : ['top', 'bottom', 'left', 'right'],
#               'theme' : 'open' }
#
#    for i in range(n_exs):
#        pprops['xlim'] = [-0.04, 0.04]
#        pprops['ylim'] = [  0.5,  1.5]
#        ydat           = [1]
#        xdat           = [inferred[i]]
#        xerr           = error[i]
#        if (i==n_exs-1):
#            pprops['xticks']      = [-0.04,   -0.02,      0,   0.02,   0.04]
#            pprops['xticklabels'] = [  ' ', r'$-2$', r'$0$', r'$2$', r'$4$']
#            pprops['xlabel']      = 'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)'
#            pprops['hide']        = ['top', 'left', 'right']
#            pprops['axoffset']    = 0.3
#        mp.plot(type='error', ax=ax[i][1], x=[xdat], y=[ydat], xerr=[xerr], colors=[colors[i]], **pprops)
#        ax[i][1].axvline(x=actual_unique[(-(i//2) + 2)%3], color=BKCOLOR, **dashlineprops)

    # SAVE FIGURE

    plt.savefig('%s/fig-3%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)

