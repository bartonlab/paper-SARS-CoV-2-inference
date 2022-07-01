#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Check what the above means

import sys
import argparse
import numpy as np                          # numerical tools
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib
import matplotlib.cm as cm
import os
import palettable as palet
import datetime as dt
import colorsys
import pandas as pd
#matplotlib.rcdefaults()
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import mplot as mp
from scipy import stats as spstats
from colorsys import hls_to_rgb, hsv_to_rgb
from tabulate import tabulate
from tqdm import tqdm


def complementary(color_hex):
    """ Returns a color complementary to the given r,g,b. """
    #hsv = colorsys.rgb_to_hsv(r, g, b)
    rgb = mcolors.to_rgb(color_hex)
    hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    return hsv_to_rgb((hsv[0] + 0.5) % 1, hsv[1], hsv[2])

NUC = ['-', 'A', 'C', 'G', 'T']
NUMBERS  = list('0123456789')
ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+++++++++++++++++++++++++++'

PROTEINS = ['NSP1', 'NSP2', 'NSP3', 'NSP4', 'NSP5', 'NSP6', 'NSP7', 'NSP8','NSP9', 'NSP10', 'NSP12', 'NSP13', 'NSP14', 
            'NSP15', 'NSP16', 'S', 'ORF3a', 'E', 'M', 'ORF6', 'ORF7a', 'ORF7b', 'ORF8', 'N', 'ORF10']
PROTEIN_LENGTHS = {'ORF3a' : 828, 'E' : 228, 'ORF6' : 186, 'ORF7a' : 365, 'ORF7b' : 132, 'S' : 3822,
                   'N' : 1260, 'M' : 669, 'ORF8' : 336, 'ORF10' : 117, 'NSP1' : 539, 'NSP2' : 1914,
                   'NSP3' : 5834, 'NSP4' : 1500, 'NSP5' : 917, 'NSP6' : 870, 'NSP7' : 249, 'NSP8' : 594,
                   'NSP9' : 339, 'NSP10' : 417, 'NSP12' :2795, 'NSP13' : 1803, 'NSP14' : 1582, 
                   'NSP15' : 1038, 'NSP16' : 894}

cm_to_inch = lambda x: x/2.54   

SINGLE_COLUMN_WIDTH = cm_to_inch(9)
DOUBLE_COLUMN_WIDTH = cm_to_inch(19)

# Plot parameters
palette1            = sns.hls_palette(2, l=0.5, h=0.5)
LCOLOR              = '#969696'
BKCOLOR             = '#252525'
#MAIN_COLOR          = '#6699CC'                  #mcolors.to_rgb('cornflowerblue')
#COMP_COLOR          = complementary(MAIN_COLOR)  #complementary(MAIN_COLOR[0], MAIN_COLOR[1], MAIN_COLOR[2])
MAIN_COLOR          = palette1[0]
COMP_COLOR          = palette1[1]
SIZELINE            = 1
AXWIDTH             = 0.4
DPI                 = 1200
SMALLSIZEDOT        = 6.
SPINE_LW            = 0.75
FREQ_LW             = 1
AXES_FONTSIZE       = 8
TICK_FONTSIZE       = 6
TICK_LENGTH         = 4
LEGEND_FONTSIZE     = 6

# Directories
PROCESS_DIR  = '/Users/brianlee/SARS-CoV-2-Data/Processing-files'
fig_path   = 'figures'
FIG_DIR    = fig_path
image_path = 'images'

# load data_processing module
cwd = os.getcwd()
os.chdir(PROCESS_DIR)
import data_processing as dp
os.chdir(cwd)


def apply_standard_params(ax, **kwargs):
    """ Apply default settings to the given axes."""
    
    ax.tick_params(labelsize=TICK_FONTSIZE, length=TICK_LENGTH, width=SPINE_LW)
    for line in ['right', 'top']:
        ax.spines[line].set_visible(False)
    for line in ['left', 'bottom']:
        ax.spines[line].set_linewidth(SPINE_LW)


def calculate_linked_coefficients(inf_file, link_file, tv=False):
    """ Finds the coefficients and the labels for the linked sites"""
    
    # loading and processing the data
    data            = np.load(inf_file, allow_pickle=True)
    linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in every region
    inferred        = data['selection']
    if len(np.shape(inferred))==1 and 'error_bars' in inf_file:
        error       = data['error_bars']
    else:
        error       = np.zeros(np.shape(inferred))
    allele_number   = data['allele_number']
    #labels          = [get_label(i) for i in allele_number]    #VERSION FOR OLD 2-ALLELE DATA
    labels          = [dp.get_label2(i) for i in allele_number]
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
        #inferred_link = np.zeros((len(inferred), len(linked_sites)))
        #error_link    = np.zeros((len(inferred), len(linked_sites)))
        inferred_link = []
        error_link    = []
        labels_link   = []
        for i in range(len(linked_sites)):
            labels_temp = []
            inf_temp    = np.zeros(len(inferred))
            error_temp  = np.zeros(len(inferred))
            for j in range(len(linked_sites[i])):
                if np.any(linked_sites[i][j]==np.array(labels)):
                    #inferred_link[:, i] += inferred[:, np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
                    #error_link[:, i]    += error[:, np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
                    inf_temp   += inferred[:, np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
                    error_temp += error[:, np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]] ** 2
                    labels_temp.append(linked_sites[i][j])
            inferred_link.append(inf_temp)
            error_link.append(np.sqrt(error_temp))
            labels_link.append(labels_temp)
        
    return labels_link, inferred_link, error_link


def calculate_linked_coefficients_alt(inf_file, link_file, tv=False):
    """ Finds the coefficients and the labels for the linked sites"""
    
    # loading and processing the data
    data            = np.load(inf_file, allow_pickle=True)
    linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in every region
    if isinstance(linked_sites[0], str):
        linked_sites = [linked_sites]    # list of sites linked to given site.
    inferred        = data['selection']
    allele_number   = data['allele_number']
    labels          = [get_label2(i) for i in allele_number]
    linked_all      = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
        
    # finding total coefficients for linked sites and adding them together.
    if not tv:
        inferred_link   = np.zeros(len(linked_sites))
        labels_link     = []
        for i in range(len(linked_sites)):
            labels_temp = []
            for j in range(len(linked_sites[i])):
                if np.any(linked_sites[i][j]==np.array(labels)):
                    inferred_link[i] += inferred[np.nonzero(linked_sites[i][j]==np.array(labels))[0][0]]
                    labels_temp.append(linked_sites[i][j])
            labels_link.append(labels_temp)
    else:
        inferred_link = np.zeros((len(inferred), len(linked_sites)))
        labels_link   = []
        for i in range(len(linked_sites)):
            labels_temp = []
            for j in range(len(linked_sites[i])):
                if np.any(linked_sites[i][j]==np.array(labels)):
                    inferred_link[:, i] += inferred[:, np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
                    labels_temp.append(linked_sites[i][j])
            labels_link.append(labels_temp)
        
    return labels_link, inferred_link


def calculate_correlation(x,y):
    """ Calculates the Pearson correlation between x and y"""
    mx = np.mean(x)
    my = np.mean(y)
    stdx = np.std(x)
    stdy = np.std(y)
    cov  = np.sum((x-mx)*(y-my)) / (len(x) - 1)
    cor  = cov / (stdx * stdy)
    return cor


def mask1d(array, mask=None):
    """mask a single 1D array along the given axis"""
    return array[mask]


def mask_arrays(arrays, mask, axis=0):
    """mask all of the arrays using the mask"""
    new_arrays = []
    if isinstance(axis, int):
        return [np.apply_along_axis(mask1d, axis, i, mask=mask) for i in arrays]
    else:
        return [np.apply_along_axis(mask1d, axis[i], arrays[i], mask) for i in range(len(arrays))]
    #if axis == 0:
    #    return [np.array(i)[mask] for i in arrays]
    #elif axis == 1:
    #    return [np.array(i)[:, mask] for i in arrays]
    #elif axis ==2:
    #    return [np.array(i)[:, :, mask] for i in arrays]
    #else:
    #    print(f'axis {axis} not supported in function')


def linked_comparison_plot(inf_file, link_file, num_show=20, outfile='individual-selection-linked'):
    """Compares the inferred coefficients for each site in a linked group.
    FUTURE: add also a comparison of the trajectories"""
    
    linked_sites = np.load(link_file, allow_pickle=True)
    data         = np.load(inf_file,  allow_pickle=True)
    inferred     = data['selection']
    site_labels  = [get_label(i) for i in data['allele_number']]
    coefficients = []    # the inferred coefficients for the sites in the linked groups
    new_labels   = []
    for i in range(len(linked_sites)):
        link_s    = np.zeros(len(linked_sites[i]))    # the inferred coefficients
        labs_temp = []
        for j in range(len(linked_sites[i])):
            if np.any(linked_sites[i][j]==np.array(site_labels)):
                link_s[j] = inferred[np.where(np.array(linked_sites[i][j])==np.array(site_labels))[0][0]]
                if linked_sites[i][j][:2]!='NC': site_label = linked_sites[i][j][:-2]
                else: site_label = linked_sites[i][j]
                labs_temp.append(site_label)
            #else:
                #print(f'site {linked_sites[i][j]} in group {linked_sites[i]} not found in inference data')
        coefficients.append(link_s)
        new_labels.append(labs_temp)
    coefficients       = [i for i in coefficients if np.sum(i)!=0]
    new_labels         = [i for i in new_labels if len(i) > 0]
    total_coefficients = [np.sum(i) for i in coefficients]
    indices            = np.argsort(total_coefficients)[::-1]
    coefficients       = np.array(coefficients)[indices][:num_show]
    new_labels         = np.array(new_labels)[indices][:num_show]
    fig, axes = plt.subplots(num_show, 1, figsize=[DOUBLE_COLUMN_WIDTH*3, DOUBLE_COLUMN_WIDTH*10], gridspec_kw={'hspace' : 0.3, 'top' : 0.95, 'bottom' : 0.05})
    for i in range(num_show):
        total = np.sum(coefficients[i])
        ax = axes[i]
        ax.plot(np.arange(len(coefficients[i])), coefficients[i], lw=0, marker='o')
        ax.set_xticks(np.arange(len(coefficients[i])))
        ax.set_xticklabels(new_labels[i], rotation=45)
        ax.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
        ax.text(1.02, 0.5, 'total\n{:.4f}'.format(total), transform=ax.transAxes, ha='center', va='center')
    fig.savefig(os.path.join(fig_path, outfile + '.png'), dpi=1200)
    
    
def plot_early_detection_subdate(inf_file, link_file, null_file, t_end=None, t_start=None, out_file='early-detection-subdate'):
    """ Plots the trajectory of the inferred coefficient for the sites in question."""
    
    null_dist  = np.genfromtxt(null_file, delimiter=',')
    max_null   = np.amax(null_dist) * 100
    alpha      = np.load(link_file, allow_pickle=True)
    inf_data   = np.load(inf_file, allow_pickle=True)
    selection  = inf_data['selection']
    times      = inf_data['times']
    sites      = inf_data['sites']
    alleles    = [dp.get_label_new(i) for i in sites]
    
    if t_end==None:
        t_end_idx = -1
    else:
        t_end_idx = list(times).index(t_end)
    if t_start==None:
        t_start_idx = 0
    else:
        t_start_idx = list(times).index(t_start)
    times = times[t_start_idx:t_end_idx]
    selection = selection[t_start_idx:t_end_idx] * 100
    
    idxs_alpha = [alleles.index(i) for i in alpha]
    s_alpha    = np.sum(np.array(selection)[:, idxs_alpha], axis=1)
    x_detect   = np.array(times)[s_alpha>max_null][0]
    print(f'variant detected at time {x_detect}')
    print(f'selection coefficient at time of detection = {s_alpha[list(times).index(x_detect)]}')
        
    xticks      = np.arange(times[0], times[-1], 21)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
    xticklabels = [f'{i.month}/{i.day}' for i in xticklabels] 
    
    fig, ax = plt.subplots(1,1, figsize=[SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH])
    ax.plot(times, s_alpha, color='k', lw=SIZELINE)
    ax.set_xlim(times[0] - 5, times[-1] + 5)
    ax.tick_params(labelsize=TICK_FONTSIZE + 2, length=4, width=SPINE_LW)
    ax.set_ylabel('Inferred Selection (%)', fontsize=AXES_FONTSIZE + 2, labelpad=4)
    ax.set_xlabel('Date', fontsize=AXES_FONTSIZE + 2)
    ax.axvline(x=x_detect, color='k', ls=':')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    for line in ['top', 'right']:
        ax.spines[line].set_visible(False)
    for line in ['left', 'bottom']:
        ax.spines[line].set_linewidth(SPINE_LW)
        
    fig.savefig(os.path.join(FIG_DIR, out_file + '.png'), dpi=2500)
    fig.savefig(os.path.join(FIG_DIR, out_file + '.png'), dpi=2500)
    #print(dt.date(2020,1,1) + dt.timedelta(int(x_detect)))
    
    
def plot_early_detection(inf_file, link_file, null_file, t_end=None, t_start=None, out_file='early-detection-subdate'):
    """ Plots the trajectory of the inferred coefficient for the sites in question."""
    
    null_dist  = np.genfromtxt(null_file, delimiter=',')
    max_null   = np.amax(null_dist)
    alpha      = np.load(link_file, allow_pickle=True)
    #inf_data   = np.load(inf_file, allow_pickle=True)
    inf_data   = pd.read_csv(inf_file)
    selection  = list(inf_data['selection'])
    frequency  = list(inf_data['frequency'])
    times      = list(inf_data['time'])
    #selection  = inf_data['selection']
    #times      = inf_data['times']
    #sites      = inf_data['sites']
    #alleles    = [dp.get_label_new(i) for i in sites]
    
    if t_end==None:
        t_end_idx = -1
    else:
        t_end_idx = list(times).index(t_end)
    if t_start==None:
        t_start_idx = 0
    else:
        t_start_idx = list(times).index(t_start)
    times = times[t_start_idx:t_end_idx]
    selection = np.array(selection[t_start_idx:t_end_idx]) * 100
    frequency = np.array(frequency[t_start_idx:t_end_idx]) * 100
    
    #idxs_alpha = [alleles.index(i) for i in alpha]
    #s_alpha    = np.sum(np.array(selection)[:, idxs_alpha], axis=1)
    x_detect   = np.array(times)[selection>max_null*100][0]
    print(f'detection time is {x_detect}')
        
    xticks      = np.arange(times[0], times[-1], 14)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in xticks]
    xticklabels = [f'{i.month}/{i.day}' for i in xticklabels] 
    
    grid_kws  = {'hspace' : 0.2}
    fig, axes = plt.subplots(2,1, figsize=[SINGLE_COLUMN_WIDTH * 1.614, DOUBLE_COLUMN_WIDTH], gridspec_kw=grid_kws)
    
    # plot selection
    ax = axes[1]
    ax.plot(times, selection, color='k', lw=SIZELINE)
    ax.set_xlim(times[0] - 5, times[-1] + 5)
    ax.tick_params(labelsize=TICK_FONTSIZE + 2, length=4, width=SPINE_LW)
    ax.set_ylabel('Inferred Selection (%)', fontsize=AXES_FONTSIZE + 2, labelpad=4)
    ax.set_xlabel('Date', fontsize=AXES_FONTSIZE + 2)
    ax.axvline(x=x_detect, color='k', ls=':')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    for line in ['top', 'right']:
        ax.spines[line].set_visible(False)
    for line in ['left', 'bottom']:
        ax.spines[line].set_linewidth(SPINE_LW)
    
    # plot frequency
    ax = axes[0]
    ax.set_xlim(times[0] - 5, times[-1] + 5)
    ax.plot(times, frequency, color='k', lw=SIZELINE)
    ax.axvline(x=x_detect, color='k', ls=':')
    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    ax.tick_params(labelsize=TICK_FONTSIZE + 2, length=4, width=SPINE_LW)
    ax.set_ylabel('Frequency (%)', fontsize=AXES_FONTSIZE + 2, labelpad=4)
    #ax.set_xlabel('Date', fontsize=AXES_FONTSIZE + 2)
    for line in ['top', 'right']:
        ax.spines[line].set_visible(False)
    for line in ['left', 'bottom']:
        ax.spines[line].set_linewidth(SPINE_LW)
    ax.set_ylim(0, 100)
    
    fig.savefig(os.path.join(FIG_DIR, out_file + '.png'), dpi=1200)
    #print(dt.date(2020,1,1) + dt.timedelta(int(x_detect)))
    
    
def plot_variant_traj(file=None, variants=[], var_sites=[], add_vars=[], add_sites=[], out='variant-comparison', min_freq=0.01, min_length=0):
    """Plots the trajectories for the variants in the variants argument, averaging over the sites given for each variant.
    Only plots the trajectories in regions in which all variants in the variants argument are present.
    Variants given in add_vars will be plotted but are not required to appear"""
    
    month_dict = {1 : 'Jan', 2 : 'Feb', 3 : 'Mar', 4 : 'Apr', 5 : 'May', 6 : 'Jun',
                  7 : 'Jul', 8 : 'Aug', 9 : 'Sep', 10 : 'Oct', 11 : 'Nov', 12 : 'Dec'}
    start_time = 258    # first day to plot the trajectories from, in this case, the day Alpha appeared
    
    # load data
    data      = np.load(file, allow_pickle=True)
    alleles   = data['allele_number']
    times     = data['times']
    traj      = data['traj']
    locs      = data['locations']
    nucs_all  = data['mutant_sites']
    sites     = [get_label(int(i[:i.find('-')])) + i[-2:] for i in alleles]
    sites_all = []
    for group in nucs_all:
        sites_all.append([get_label(int(i[:i.find('-')])) + i[-2:] for i in group])
    
    # filter locations that don't have data for the variants
    mask  = []
    for i in sites_all:
        present = True
        for j in var_sites:
            if j[0] not in i:
                present = False
        if present:
            mask.append(True)
        else:
            mask.append(False)
    mask  = np.array(mask)
    traj  = traj[mask]
    locs  = locs[mask]
    times = times[mask]
    sites_all = np.array(sites_all)[mask]
    
    # Combine trajectories for the same regions
    temp_locs   = [i[:i.find('2')-1] for i in locs]
    unique_locs = np.unique(temp_locs)
    locs_repeat = []
    idxs_repeat = []
    for loc in unique_locs:
        if len(np.where(loc==np.array(temp_locs))[0])>1:
            locs_repeat.append(loc)
            idxs_repeat.append(np.where(loc==np.array(temp_locs))[0])
    new_locs  = []
    new_times = []
    new_traj  = []
    new_sites = []
    for i in range(len(locs_repeat)):
        temp_sites = []
        idxs_temp  = idxs_repeat[i]
        for j in range(len(idxs_temp)):
            temp_sites += list(sites_all[idxs_temp[j]])
        temp_sites = np.unique(temp_sites)
        t_min      = np.amin([np.amin(times[j]) for j in idxs_temp])
        t_max      = np.amax([np.amax(times[j]) for j in idxs_temp])
        temp_times = np.arange(t_min, t_max + 1)
        temp_traj  = np.zeros((len(temp_times), len(temp_sites)))
        #print(f'shape of traj for region {locs_repeat[i]}', np.shape(temp_traj))
        temp_traj.fill(np.nan)
        for j in range(len(temp_sites)):
            for k in range(len(idxs_temp)):
                if temp_sites[j] in sites_all[idxs_temp[k]]:
                    t_start = list(temp_times).index(times[idxs_temp[k]][0])
                    #print(t_start)
                    #print(np.shape(traj[idxs_temp[k]][:, list(sites_all[idxs_temp[k]]).index(temp_sites[j])]))
                    #print(len(times[idxs_temp[k]]))
                    temp_traj[t_start:t_start + len(times[idxs_temp[k]]), j] = traj[idxs_temp[k]][:, list(sites_all[idxs_temp[k]]).index(temp_sites[j])]
        
        new_sites.append(temp_sites)
        new_times.append(temp_times)
        new_locs.append(locs_repeat[i])
        new_traj.append(temp_traj)
        
    #print(np.array(new_locs))
        
    idxs_elim = np.array([idxs_repeat[i][j] for i in range(len(idxs_repeat)) for j in range(len(idxs_repeat[i]))])
    idxs_keep = np.array([i for i in range(len(locs)) if i not in idxs_elim])
    sites_all = list(np.array(sites_all)[idxs_keep]) + new_sites
    traj  = list(np.array(traj)[idxs_keep]) + new_traj
    times = list(np.array(times)[idxs_keep]) + new_times
    locs  = list(np.array(locs)[idxs_keep]) + new_locs
    
    # Eliminate regions where the maximumn frequencies of any of the variants never rises above the frequency cutoff
    idxs_elim = []
    for i in range(len(locs)):
        max_freq_min = 1
        for j in var_sites:
            temp_freq = np.amax(traj[i][:, list(sites_all[i]).index(j[0])])
            max_freq_min = min(max_freq_min, temp_freq)
        if max_freq_min < min_freq:
            idxs_elim.append(i)
    #idxs_elim = np.array(idxs_elim)
    idxs_keep = np.array([i for i in range(len(locs)) if i not in idxs_elim])
    sites_all = list(np.array(sites_all)[idxs_keep])
    traj      = list(np.array(traj)[idxs_keep])
    locs      = list(np.array(locs)[idxs_keep])
    times     = list(np.array(times)[idxs_keep])
    
    # Trim time intervals for plotting
    new_traj  = []
    new_times = []
    for i in range(len(traj)):
        if start_time in times[i]:
            idx = list(times[i]).index(start_time)
            new_traj.append(traj[i][idx:])
            new_times.append(times[i][idx:])
        else:
            new_traj.append(traj[i])
            new_times.append(times[i])
    times = new_times
    traj  = new_traj
    
    # Eliminate regions with time-series shorter than min_length
    idxs_keep = np.array([i for i in range(len(times)) if len(times[i])>=min_length])
    times     = np.array(times)[idxs_keep]
    traj      = np.array(traj)[idxs_keep]
    locs      = np.array(locs)[idxs_keep]
    sites_all = np.array(sites_all)[idxs_keep]
    
    # create figure and axes
    plot_columns = 5
    plot_rows    = int(len(locs) / plot_columns) + 1
    grid_kws     = {'hspace' : 0.35, 
                    'wspace' : 0.1,
                    'left'   : 0.07,
                    'right'  : 0.99, 
                    'top'    : 0.985, 
                    'bottom' : 0.05}
    
    fig = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 1.2])
    grid = matplotlib.gridspec.GridSpec(plot_rows, plot_columns, **grid_kws)
    #fig, axes = plt.subplots(plot_rows, plot_columns, gridspec_kw=grid_kws,
    #                         figsize=[DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 1.4])
    yticks    = [0, 0.5, 1]
    yticklabs = ['0', '0.5', '1']
    
    # find earliest and latest date so all regions can be plotted on the same time scale
    min_time = np.amin([np.amin(i) for i in times])
    max_time = np.amax([np.amax(i) for i in times])
    xlabels  = []
    xticks   = []
    for i in range(min_time, max_time + 1):
        time = dt.date(2020, 1, 1) + dt.timedelta(int(i))
        if time.day == 1:
            new_time = month_dict[int(time.month)] + ' ' + str(time.year)[2:] 
            xlabels.append(new_time)
            xticks.append(i)
    if len(xticks)>5:
        xticks  = xticks[::int(len(xticks) / 5)]
        xlabels = xlabels[::int(len(xlabels) / 5)]
    
    # make the y-axis label
    ax_tot = fig.add_subplot(grid[:, :])
    for line in ['top', 'right', 'left', 'bottom']:
        ax_tot.spines[line].set_visible(False)
    ax_tot.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax_tot.set_ylabel('Frequency', fontsize=AXES_FONTSIZE)
    #ax_tot.set_xticks([])
    #ax_tot.set_yticks([])
    
    # plot trajectories
    for j in range(plot_rows):
        for i in range(plot_columns):
            idx  = (i) + (plot_columns * j)
            #ax   = axes[j, i]
            ax   = fig.add_subplot(grid[j, i])
            if idx >= len(locs):
                ax.axis('off')
                continue
            loc  = locs[idx]
            tr   = traj[idx]
            muts = sites_all[idx]
            t    = times[idx]
            
            # set plot properties
            ax.set_ylim(0,1)
            ax.set_xlim(min_time, max_time)
            ax.set_yticks(yticks)
            if i == 0:
                ax.set_yticklabels(yticklabs)
                #ax.set_ylabel('Frequency', fontsize=AXES_FONTSIZE)
            else:
                ax.set_yticklabels([])
            apply_standard_params(ax)
            ax.set_xticks(xticks)
            if j==plot_rows-1 or (j==plot_rows-2 and idx+plot_columns>=len(locs)):
                ax.set_xticklabels(xlabels)
                ax.set_xlabel('Date', fontsize=AXES_FONTSIZE)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis='x', pad=0, rotation=45)
            ax.tick_params(labelsize=TICK_FONTSIZE, length=2, width=SPINE_LW)
            
            """
            # format xticks and xticklabels
            xlabels = []
            xticks  = []
            for i in range(len(t)):
                time = dt.date(2020, 1, 1) + dt.timedelta(int(t[i]))
                if time.day == 1:
                    new_time = month_dict[int(time.month)] + ' ' + str(time.year)[2:] 
                    xlabels.append(new_time)
                    xticks.append(t[i])
            if len(xticks)>4:
                xticks  = xticks[::int(len(xticks) / 4)]
                xlabels = xlabels[::int(len(xticks) / 4)]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels)
            ax.tick_params(axis='x', pad=-2, rotation=45)
            """
            
            # plot frequency trajectories
            for k in range(len(variants)):
                traj_var = []
                for l in range(len(var_sites[k])):
                    if var_sites[k][l] in muts:
                        traj_var.append(tr[:, list(muts).index(var_sites[k][l])])
                traj_var = np.mean(np.array(traj_var), axis=0)
                ax.plot(t, traj_var, label=variants[k], linewidth=SIZELINE)
            for k in range(len(add_vars)):
                if add_sites[k][0] in muts:
                    traj_var = []
                    for l in range(len(add_sites[k])):
                        if add_sites[k][l] in muts:
                            traj_var.append(tr[:, list(muts).index(add_sites[k][l])])
                    traj_var = np.mean(np.array(traj_var), axis=0)
                    ax.plot(t, traj_var, label=add_vars[k], linewidth=SIZELINE)
                else:
                    ax.plot([t[0]], [0], label=add_vars[k], linewidth=SIZELINE)
            
            # make title
            loc_split = loc.split('-')
            if loc_split[1]=='usa':
                title = '-'.join(loc.split('-')[2:])
            else:
                title = '-'.join(loc.split('-')[1:])
            if title.find('2')!=-1:
                title = title[:title.find('2')-1]
            if title.find('---')!=-1:
                title = title[:title.find('---')][:-1]
            if title.split('-')[-1] == 'england_wales_scotland':
                title = '-'.join(title.split('-')[:-1])
            if title.split('-')[-1] == 'northern ireland':
                title = 'Northern Ireland'
            while title[-1]=='-':
                title = title[:-1]
            title = title.capitalize()
            ax.set_title(title, fontsize=AXES_FONTSIZE, pad=-4)
            
            # make legend
            if i==0 and j==0:
                ax.legend(loc='upper left', fontsize=LEGEND_FONTSIZE)
    
    plt.savefig(os.path.join(FIG_DIR, out + '.png'), dpi=1200)
    
    
def plot_selection(data, out='selection-plot', title=None, cutoff=1, theoretical=False, small=False):
    """ Plot selection coefficients with error bars"""
    df   = pd.read_csv(data)
    s    = np.array(list(df['selection coefficient'])) * 100
    if theoretical:
        std = np.array(list(df['error'])) * 100
    else:
        std  = np.array(list(df['standard deviation'])) * 100
        
    ymax = np.amax(s) + np.amax(std) + 0.5
    ymin = np.amin(s) - np.amax(std) - 0.5
    if not small:
        mask = np.absolute(s)>=cutoff
    else:
        mask = np.absolute(s)<=cutoff
    s    = np.array(s)[mask]
    std  = np.array(std)[mask]
    sort = np.argsort(s)
    s    = s[sort]
    std  = std[sort]
    print(f'there are {len(s)} selection coefficients')
    
    grid_kws = {'left'   : 0.1,
                'right'  : 0.99, 
                'top'    : 0.925, 
                'bottom' : 0.15}    
    fig, ax = plt.subplots(1, 1, figsize=[DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH], gridspec_kw=grid_kws)
    ax.set_ylabel('Inferred selection (%)', fontsize=AXES_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=AXES_FONTSIZE)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    apply_standard_params(ax)
    ax.errorbar(np.arange(len(s)), s, yerr=std, lw=0, marker='.', elinewidth=0.5, ms=2)
    plt.savefig(os.path.join(FIG_DIR, out + '.png'), dpi=1500)
    
    
def selection_error(bs_data=None, ss_data=None, out='selection-plot', cutoff=1):
    """ Plot selection coefficients with error bars.
    bs_data : data file containing inferred coefficients as well as error due to bootstrapping sequence data
    ss_data : data file containing inferred coefficients as well as error due to subsampling regions"""
    bs_df   = pd.read_csv(bs_data)
    s       = np.array(list(bs_df['selection coefficient'])) * 100
    bs_std  = np.array(list(bs_df['standard deviation'])) * 100
    th_std  = np.array(list(bs_df['error'])) * 100    # the theoretical standard deviation
    ss_df   = pd.read_csv(ss_data)
    ss_std  = np.array(list(ss_df['standard deviation'])) * 100
    
    max_std = np.amax([np.amax(ss_std), np.amax(bs_std)])
    ymax = np.amax(s) + max_std + 0.5
    ymin = np.amin(s) - max_std - 0.5
    
    mask   = np.absolute(s)>=cutoff
    s      = np.array(s)[mask]
    bs_std = np.array(bs_std)[mask]
    ss_std = np.array(ss_std)[mask]
    th_std = np.array(th_std)[mask]
    sort   = np.argsort(s)
    s      = s[sort]
    bs_std = bs_std[sort]
    ss_std = ss_std[sort]
    th_std = th_std[sort]
    print(f'there are {len(s)} selection coefficients')
    
    grid_kws = {'left'   : 0.1,
                'right'  : 0.99, 
                'top'    : 0.925, 
                'bottom' : 0.05,
                'hspace' : 0.15}    
    fig  = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH])
    grid = matplotlib.gridspec.GridSpec(2, 1, **grid_kws)
    
    ax1  = fig.add_subplot(grid[0, 0])
    ax1.set_ylim(ymin, ymax)
    ax1.set_xticks([])
    apply_standard_params(ax1)
    ax1.errorbar(np.arange(len(s)), s, yerr=bs_std, lw=0, marker='.', elinewidth=0.5, ms=2)
    ax1.set_title('Uncertainty due to subsampling sequences', fontsize=AXES_FONTSIZE)
    ax1.set_ylabel('Inferred selection (%)', fontsize=AXES_FONTSIZE)
    
    ax2  = fig.add_subplot(grid[1, 0])
    ax2.set_ylim(ymin, ymax)
    ax2.set_xticks([])
    apply_standard_params(ax2)
    ax2.errorbar(np.arange(len(s)), s, yerr=ss_std, lw=0, marker='.', elinewidth=0.5, ms=2)
    ax2.set_title('Uncertainty due to subsampling regions', fontsize=AXES_FONTSIZE)
    ax2.set_ylabel('Inferred selection (%)', fontsize=AXES_FONTSIZE)
    ax2.set_xlabel('Single nucleotide variant', fontsize=AXES_FONTSIZE)
    
    fig.text(0.05, 0.935, 'a', fontweight='bold', transform=fig.transFigure, ha='center', va='center', fontsize=AXES_FONTSIZE+6)
    fig.text(0.05, 0.475, 'b', fontweight='bold', transform=fig.transFigure, ha='center', va='center', fontsize=AXES_FONTSIZE+6)
    
    plt.savefig(os.path.join(FIG_DIR, out + '.pdf'), dpi=1500)
    
    
def matrix_multiplot(covariance_mat=False, **kwargs):
    """Plots multiple matrices, normalized and unnormalized, for each of the variants.
    Expects a variants keyword containing a list of named variants, and a files keyword containing the files.
    If there are multiple files for each variant, 'files' keyword should be a list of lists.
    There must be an 'out' keyword as well.
    Optionally can contain 'titles' and 'figsize' keywords."""
    # read in parameters
    if 'variants' in kwargs:
        variants = kwargs['variants']
    else:
        print('no variants specified')
    if 'files' in kwargs:
        files = kwargs['files']
    else:
        print('no files given')
    if 'figsize' in kwargs:
        figsize = kwargs['figsize']
    else:
        figsize = [DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 1.614]
    if isinstance(files[0], type([])):    # multiple plots for each variant, 1 variant per row
        plot_columns = len(files[0])
        plot_rows    = len(variants)
    elif len(variants)==len(files):    # one plot for each variant
        plot_columns = len(variants)
        plot_rows    = 1
    if 'titles' in kwargs:
        titles = kwargs['titles']
    else:
        titles = None 
    if 'out' in kwargs:
        out = kwargs['out']
    else: 
        out = 'matrix-plot.csv'
    if 'symmetric' in kwargs:
        symmetric = kwargs['symmetric']
    else:
        symmetric = False
    if 'plot_diag' in kwargs:
        plot_diag = kwargs['plot_diag']
    else:
        plot_diag = False
    if 'zeroed_norm' in kwargs:
        zeroed_norm = kwargs['zeroed_norm']
    else:
        zeroed_norm = True
    if 'multi_cmap' in kwargs:
        multi_cmap = kwargs['multi_cmap']
    else:
        multi_cmap = False
    if 'common_norm' in kwargs:
        common_norm = True
        max_value = 0
        for file in files:
            df = pd.read_csv(file)
            data = np.array(df.to_numpy()[:, 1:], dtype=np.float32)
            np.fill_diagonal(data, 0.0)
            max_value = max(max_value, np.amax(np.absolute(data)))
    else:
        common_norm = False
    
    #width_ratios = [15] * (plot_columns - 1) + [1]
    grid_kws     = {'hspace' : 0.5, 
                    'wspace' : 0.1,
                    'left'   : 0.15,
                    'right'  : 0.9, 
                    'top'    : 0.985, 
                    'bottom' : 0.15,
                    'width_ratios' : [30, 1]}
    if multi_cmap:
        grid_kws['width_ratios'] = [1, 1]
        grid_kws['wspace'] = 0.4
    fig = plt.figure(figsize=figsize)
    grid = matplotlib.gridspec.GridSpec(plot_rows, plot_columns, **grid_kws)
    if covariance_mat:
        ax1 = fig.add_subplot(grid[0, 0])
        plot_props1 = {
            'file'        : files[0],
            'title'       : titles[0],
            'symmetrical' : True,
            'ax'          : ax1,
            'fig'         : fig
        }
        matrix_color_plot(**plot_props1)
        ax2 = fig.add_subplot(grid[0, 1])
        plot_props2 = {
            'file'        : files[1],
            'title'       : titles[1],
            'symmetrical' : True,
            'ax'          : ax2,
            'fig'         : fig
        }
        matrix_color_plot(**plot_props2)
    else:
        #subgrid_kws = {'wspace' : 0.4}
        for i in range(plot_rows):
            if not multi_cmap:
                sub_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(1, plot_columns, subplot_spec=grid[i, 0], wspace=0.4)
            else:
                sub_grid = grid
            for j in range(plot_columns):
                ax = fig.add_subplot(sub_grid[i, j])
                if plot_rows==1:
                    file  = files[j]
                    title = titles[j]
                else:
                    file  = files[i][j]
                    title = titles[i][j]
                plotprops = {
                    'file'        : file,
                    'title'       : title,
                    'ax'          : ax,
                    'fig'         : fig,
                    'zeroed_norm' : True
                }
                if 'normed' in file:
                    plotprops['one_diag'] = True
                else:
                    plotprops['zero_diag']   = True
                    plotprops['clip']        = False
                if common_norm:
                    plotprops['max_value'] = max_value
                if symmetric:
                    plotprops['symmetrical'] = True
                if plot_diag:
                    plotprops['zero_diag'] = False
                    plotprops['one_diag']  = False
                #if j!=plot_columns - 2:
                #    plotprops['return_colorbar'] = True
                cmap, norm = matrix_color_plot(**plotprops)
                if multi_cmap:
                    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.65)
                    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
            if not multi_cmap:
                ax_color = fig.add_subplot(grid[i, -1])
                cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_color, shrink=0.6)
    
    fig.savefig(os.path.join(FIG_DIR, out + '.png'), dpi=1500)
    
    
def matrix_multiplot_new(**kwargs):
    """Plots multiple matrices, normalized and unnormalized, for each of the variants.
    Expects a variants keyword containing a list of named variants, and a files keyword containing the files.
    If there are multiple files for each variant, 'files' keyword should be a list of lists.
    There must be an 'out' keyword as well.
    Optionally can contain 'titles' and 'figsize' keywords."""
    # read in parameters
    if 'variants' in kwargs:
        variants = kwargs['variants']
    else:
        print('no variants specified')
    if 'files' in kwargs:
        files = kwargs['files']
    else:
        print('no files given')
    if 'figsize' in kwargs:
        figsize = kwargs['figsize']
    else:
        figsize = [DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 1.4]
    if isinstance(files[0], type([])):    # multiple plots for each variant, 1 variant per row
        plot_columns = len(files[0])
        plot_rows    = len(variants)
    elif len(variants)==len(files):    # one plot for each variant
        plot_columns = len(variants)
        plot_rows    = 1
    else:
        print('files and variants input are of incompatible shape')
    if 'titles' in kwargs:
        titles = kwargs['titles']
    else:
        titles = None 
    if 'out' in kwargs:
        out = kwargs['out']
    else: 
        out = 'matrix-plot.csv'
    
    if 'plot_diag' in kwargs:
        plot_diag = kwargs['plot_diag']
    else:
        plot_diag = False
    if 'zeroed_norm' in kwargs:
        zeroed_norm = kwargs['zeroed_norm']
    else:
        zeroed_norm = True
    
    #width_ratios = [15] * (plot_columns - 1) + [1]
    grid_kws     = {'hspace' : 0.25, 
                    'wspace' : 0.15,
                    'left'   : 0.1,
                    'right'  : 0.999, 
                    'top'    : 0.96, 
                    'bottom' : 0.06}
    
    # for the correlation plots use the same scale
    max_value = 0
    for i in range(len(files)):
        df    = pd.read_csv(files[i][0])
        data  = np.array(df.to_numpy()[:, 1:], dtype=np.float32)
        np.fill_diagonal(data, 0.0)
        max_value = max(max_value, np.amax(np.absolute(data)))
        

    fig = plt.figure(figsize=figsize)
    grid = matplotlib.gridspec.GridSpec(plot_rows, plot_columns, **grid_kws)
    for i in range(plot_rows):
        #sub_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(1, plot_columns, subplot_spec=grid[i, 0], wspace=0.4)
        for j in range(plot_columns):
            ax = fig.add_subplot(grid[i, j])
            file  = files[i][j]
            #title = titles[i][j]
            plotprops = {
                'file'          : file,
                'ax'            : ax,
                'fig'           : fig,
                'zeroed_norm'   : zeroed_norm,
                'plot_colorbar' : True
                }
            if 'normed' in file:
                plotprops['one_diag'] = True
            else:
                plotprops['zero_diag']   = True
                plotprops['clip']        = False
                plotprops['negative']    = True
            if i==0:
                #plotprops['title'] = titles[j]
                title = titles[j]
                ax.set_title(title, fontsize=AXES_FONTSIZE+2, pad=15)
            if j==0:
                plotprops['max_value'] = max_value

            cmap, norm = matrix_color_plot(**plotprops)
    
    # add variant labels
    alpha_coords   = [0.01, 0.825]
    delta_coords   = [0.01, 0.525]
    omicron_coords = [0.01, 0.2]
    fig.text(alpha_coords[0],   alpha_coords[1],   'Alpha',   rotation=90, transform=fig.transFigure, ha='left', va='center', fontsize=AXES_FONTSIZE+2)
    fig.text(delta_coords[0],   delta_coords[1],   'Delta',   rotation=90, transform=fig.transFigure, ha='left', va='center', fontsize=AXES_FONTSIZE+2)
    fig.text(omicron_coords[0], omicron_coords[1], 'Omicron', rotation=90, transform=fig.transFigure, ha='left', va='center', fontsize=AXES_FONTSIZE+2)
    
    # add a,b,c,.. labels
    fig.text(0.025, 0.955, 'a', fontweight='bold', transform=fig.transFigure, ha='center', va='center', fontsize=AXES_FONTSIZE+6)
    fig.text(0.525, 0.955, 'b', fontweight='bold', transform=fig.transFigure, ha='center', va='center', fontsize=AXES_FONTSIZE+6)
    fig.text(0.025, 0.655, 'c', fontweight='bold', transform=fig.transFigure, ha='center', va='center', fontsize=AXES_FONTSIZE+6)
    fig.text(0.525, 0.655, 'd', fontweight='bold', transform=fig.transFigure, ha='center', va='center', fontsize=AXES_FONTSIZE+6)
    fig.text(0.025, 0.35,  'e', fontweight='bold', transform=fig.transFigure, ha='center', va='center', fontsize=AXES_FONTSIZE+6)
    fig.text(0.525, 0.35,  'f', fontweight='bold', transform=fig.transFigure, ha='center', va='center', fontsize=AXES_FONTSIZE+6)

        #ax_color = fig.add_subplot(grid[i, -1])
        #cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_color, shrink=0.6)
    
    fig.savefig(os.path.join(FIG_DIR, out + '.png'), dpi=1500)
    
    
def matrix_color_plot(
      file=None, 
      max_value=None, 
      zero_diag=False,
      one_diag=True,
      out=None, 
      nuc_labels=False, 
      ch=False, 
      title=None, 
      clip=True, 
      symmetrical=False, 
      ax=None, 
      fig=None, 
      plot_colorbar=False, 
      zeroed_norm=True,
      negative=False
):
    """ Plot the magnitude of the entries of a matrix as colors on a 2D plot"""
    df = pd.read_csv(file)
    x  = np.arange(len(df) + 1)
    labels = list(df.columns[1:])
    #labels = [i[:-2] for i in labels]
    if not nuc_labels:
        labs_split = [i.split('-') for i in labels]
        prots      = [labs_split[i][0] for i in range(len(labs_split))]
        prot_index = []
        for i in range(len(labs_split)):
            prot_index.append(labs_split[i][1])

    data = np.array(df.to_numpy()[:, 1:], dtype=np.float32)
    if zero_diag:
        for i in range(len(data)):
            data[i, i] = 0
    elif one_diag:
        for i in range(len(data)):
            data[i, i] = 1
        
    # sorting mutations
    if not nuc_labels:
        sort1  = np.argsort(prots)[::-1]
        data   = data[sort1][:, sort1]
        labels = np.array(labels)[sort1]
        prots  = np.array(prots)[sort1]
        prot_index    = np.array(prot_index)[sort1]
        prots_unique  = np.unique(prots)
        sorted_labels = []
        for prot in prots_unique:
            idxs = np.where(prots==prot)[0]
            labs_temp = labels[idxs]
            prot_idx_temp = prot_index[idxs]
            sort2 = np.argsort(prot_idx_temp)[::-1]
            labs_temp = labs_temp[sort2]
            for label in labs_temp:
                sorted_labels.append(label)
        sorter = []
        for i in range(len(sorted_labels)):
            sorter.append(list(labels).index(sorted_labels[i]))
        sorter = np.array(sorter)
        data   = data[sorter][:, sorter]
        labels = sorted_labels
        
        labels = [i.split('-') for i in labels]
        new_labels = []
        for label in labels:
            if label[0]=='NC':
                new_labels.append(label)
            else:
                label[2] = str(int(label[2]) + 1)
                new_labels.append(label)
        labels = ['-'.join(i) for i in new_labels]
    print(labels)
        
    
    # making figure
    grid_kws = {'left'   : 0.07,
                'right'  : 0.99, 
                'top'    : 0.925, 
                'bottom' : 0.15}    
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=[DOUBLE_COLUMN_WIDTH * 1.2, DOUBLE_COLUMN_WIDTH], gridspec_kw=grid_kws)
    
    # plotting matrix
    max_v = np.amax(np.abs(data))
    if max_value:
        max_v = max_value
    #norm = matplotlib.colors.CenteredNorm(halfrange=max_value)
    #cmap = 'seismic'
    
    if ch:
        cmap = sns.cubehelix_palette(start=.5, rot=-1, as_cmap=True, dark=0, light=1)
    elif symmetrical:
        cmap = sns.color_palette("vlag", as_cmap=True)
    elif negative:
        cmap = sns.color_palette('mako', as_cmap=True)
    else:
        cmap = sns.color_palette('rocket_r', as_cmap=True)
        
    if clip:
        if not negative:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
        else:
            norm = matplotlib.colors.Normalize(vmin=-1, vmax=0, clip=True)
    elif symmetrical:
        norm = matplotlib.colors.Normalize(vmin=-max_v, vmax=max_v)
    elif zeroed_norm:
        if not negative:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=max_v)
        else:
            norm = matplotlib.colors.Normalize(vmax=0, vmin=-max_v)
    else:
        norm = matplotlib.colors.Normalize()
    
    #cmap = sns.color_palette("mako", as_cmap=True)
    #cmap = sns.color_palette("coolwarm", as_cmap=True)
    #cmap = sns.color_palette("vlag", as_cmap=True)
    #cmap = 'seismic'
    #cmap = 'bwr'
    #cmap = 'PiYG'
    #norm = matplotlib.colors.Normalize(vmin=-max_v, vmax=max_v)
    #norm = matplotlib.colors.LogNorm(vmin=-max_v, vmax=max_v)
    #norm = matplotlib.colors.SymLogNorm(linthresh=0.1, vmin=-max_v, vmax=max_v)
    im = ax.imshow(data, cmap=cmap, norm=norm)
    #im = ax.pcolormesh(x, x, data, cmap=cmap)
    if plot_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.99)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE-2)
    
    # adjusting figure propterties
    ticks = np.arange(len(labels))
    ax.tick_params(labelsize=TICK_FONTSIZE-2, length=TICK_LENGTH, width=SPINE_LW)
    for line in ['left', 'right', 'top', 'bottom']:
        ax.spines[line].set_linewidth(SPINE_LW)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    if title:
        #ax.set_title(title, fontsize=AXES_FONTSIZE, pad=15)
        ax.set_title(title, fontsize=AXES_FONTSIZE, pad=-10)
    
    if out is not None:
        plt.savefig(os.path.join(FIG_DIR, out + '.png'), dpi=1500)
        
    return cmap, norm


def travel_plot(inf_dir, link_file, sim1, sim2, outfile='travel-plot'):
    """Plot travel related figures"""
    
    fig  = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH])
    grid = matplotlib.gridspec.GridSpec(1, 2, wspace=0.2, left=0.05, right=0.95, bottom=0.2, top=0.95)
    right_grid = grid[0, 1].subgridspec(2, 1, wspace=0.2, hspace=0.3)
    
    migration_grid = right_grid[1, :].subgridspec(1, 2, wspace=0.2)
    migration_plot(inf_dir, link_file, grid=migration_grid, fig=fig)
    
    ax_hist = fig.add_subplot(right_grid[0, :])
    hist_plot_inflow(sim1, sim2, ax=ax_hist)
    fig.savefig(os.path.join(FIG_DIR, outfile + '.png'), dpi=1500)


def migration_plot(inf_dir, link_file, grid=None, fig=None, days_migrating=100, out_file='migration-plot'):
    """ Given inference files with different numbers of imported sequences per time, makes a plot of how the inference changes
    depending on the number of total imported sequences.
    --generations_migrating specifies the number of days for which sequences are migrating into the region"""
    
    # matplotlib functions
    offset_box     = matplotlib.offsetbox
    annotation_box = offset_box.AnnotationBbox
    offset_img     = offset_box.OffsetImage
    
    uk_file        = os.path.join(image_path, 'United-kingdom-map-blank.png')
    spain_file     = os.path.join(image_path, 'Spain-map-blank.jpeg')
    
    # find index of the desired site in the groups of linked sites
    site         = 'S-222-1-T'
    linked_sites = np.load(link_file, allow_pickle=True)
    if isinstance(linked_sites[0], str):
        linked_sites = [linked_sites]
    site_idx     = 0
    for i in range(len(linked_sites)):
        if site in linked_sites[i]:
            site_idx = i
            break
    
    # iterate through the various inference files and determine the inferred selection coefficients and the number of inflowing sequences for each
    num_inflow = []
    inferred   = []
    for file in os.listdir(inf_dir):
        start_idx  = file.index('inflow') + 6
        num_inflow.append(int(file[start_idx:-4]) * days_migrating)
        linked_inf = calculate_linked_coefficients_alt(os.path.join(inf_dir, file), link_file)[1][site_idx]
        inferred.append(linked_inf)
    indices    = np.argsort(num_inflow)
    num_inflow = np.array(num_inflow)[indices]
    inferred   = np.array(inferred)[indices]
    num_inflow = num_inflow[inferred>0]
    inferred   = inferred[inferred>0]
    slope      = (inferred[1] - inferred[0]) / (num_inflow[1] - num_inflow[0])
    xintercept = - inferred[0] / slope
    inferred   = list(inferred)
    num_inflow = list(num_inflow)
    inferred.append(0)
    num_inflow.append(xintercept)
    inferred   = np.array(inferred)
    num_inflow = np.array(num_inflow)
    
    # make figure and plot the inferred coefficient vs. the number of inflowing sequences 
    #fig = plt.figure(figsize=[SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH/1.618])
    #grid = matplotlib.gridspec.GridSpec(1, 2, wspace=0.2, left=0.05, right=0.95, bottom=0.2, top=0.95)
    
    ax_pic     = fig.add_subplot(grid[0, 0])
    patches    = []
    
    uk_xy      = [0.55, 0.75]
    uk_pic     = matplotlib.image.imread(uk_file)
    uk_box     = offset_img(uk_pic, zoom=0.02)
    uk_ann     = annotation_box(uk_box, uk_xy, frameon=False)
    uk_ann.set_zorder(100)
    ax_pic.add_artist(uk_ann)
    ax_pic.text(uk_xy[0] - 0.25, uk_xy[1] + 0.15, 'UK', fontsize=AXES_FONTSIZE, ha='center', va='center', transform=ax_pic.transAxes, zorder=101)
    spain_xy   = [0.175, 0.175]
    spain_pic  = matplotlib.image.imread(spain_file)
    spain_box  = offset_img(spain_pic, zoom=0.075)
    spain_ann  = annotation_box(spain_box, spain_xy, frameon=False)
    spain_ann.set_zorder(102)
    ax_pic.add_artist(spain_ann)
    ax_pic.text(spain_xy[0] + 0.2, spain_xy[1] - 0.1, 'Spain', fontsize=AXES_FONTSIZE, ha='center', va='center', transform=ax_pic.transAxes, zorder=103)
    
    start_xy   = np.array(spain_xy) + np.array([0, 0.05])
    end_xy     = np.array(uk_xy) - np.array([-0.1, 0.05])
    arrowstyle = matplotlib.patches.ArrowStyle.CurveB(head_length=5, head_width=2.5)
    arrow      = matplotlib.patches.FancyArrowPatch(posA=start_xy, posB=end_xy, arrowstyle=arrowstyle, connectionstyle='arc3,rad=0.3', zorder=104)
    patches.append(arrow)
    ax_pic.text(((start_xy[0] + end_xy[0]) / 2) + 0.15,  ((start_xy[1] + end_xy[1]) / 2) - 0.15, 'm', fontsize=8, va='center', ha='right', transform=ax_pic.transAxes)
    
    for patch in patches:
        ax_pic.add_patch(patch)
    ax_pic.axis('off')
    
    ax = fig.add_subplot(grid[0, 1])
    ax.plot(num_inflow, 100*np.array(inferred), color=MAIN_COLOR)
    ax.set_ylabel('Inferred coefficient (%)', fontsize=AXES_FONTSIZE)
    ax.set_xlabel('Importations (m)',         fontsize=AXES_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
    for line in ['right', 'top']:
        ax.spines[line].set_visible(False)
    
    #df = pd.DataFrame.from_dict({'importations' : num_inflow, 'importations per day' : np.array(num_inflow)/days_migrating, 'inferred coefficient' : inferred})
    
    fig.savefig(os.path.join(fig_path, out_file + '.png'), dpi=1200)
    
    
def hist_plot_inflow(simulation1, simulation2, ax=None, title=None, x_lims=None):
    inf1  = np.load(simulation1, allow_pickle=True)['inferred']
    inf2 = np.load(simulation2, allow_pickle=True)['inferred']
    inflow1 = [inf1[i][6] for i in range(len(inf1)) if 6 in range(len(inf1[i]))]
    inflow2 = [inf2[i][6] for i in range(len(inf2)) if 6 in range(len(inf2[i]))]
    inflow_weights1 = np.ones(len(inflow1)) / len(inflow1)
    inflow_weights2 = np.ones(len(inflow2)) / len(inflow2)
    hist_dict1 = {"weights" : inflow_weights1, "range" : [-0.01, 0.03]}
    hist_dict2 = {"weights" : inflow_weights2, "range" : [-0.02, 0.02 ]}
    #fig = plt.figure(figsize=[SINGLE_COLUMN_WIDTH,2])
    #ax  = fig.add_subplot(1,1,1)
    ax.set_ylim(0, 0.1)
    plt.tick_params(axis='y', labelsize=TICK_FONTSIZE, width=SPINE_LW)
    sns.distplot(inflow1, kde=False, bins=50, color="lightcoral", hist_kws=hist_dict1, label='naive')
    sns.distplot(inflow2, kde=False, bins=50, color="cornflowerblue", hist_kws=hist_dict2, label='corrected')
    #ax.axvline(x=0, color='k')
    ax.set_xlabel('Inferred selection\ncoefficients (%)', fontsize=AXES_FONTSIZE)
    ax.set_ylabel('Frequency', fontsize=AXES_FONTSIZE)
    ax.text(0.35, 0.8, 'corrected', fontsize=6, transform=ax.transAxes, ha='center', va='center', color='cornflowerblue')
    ax.text(0.85, 0.9, 'naive', fontsize=6, transform=ax.transAxes, ha='center', va='center', color='lightcoral')
    s_max = 0.03
    ax.set_xlim(-s_max, s_max)
    ax.set_xticks(np.linspace(-s_max, s_max, 5))
    ax.set_xticklabels([f'{i}'.replace('-', '\N{MINUS SIGN}') for i in np.linspace(-int(s_max*100), int(s_max*100), 5, dtype=int)], fontsize=6)
    ax.tick_params(labelsize=TICK_FONTSIZE, width=0.5)
    
    for line in ['bottom', 'left']:
        ax.spines[line].set_linewidth(SPINE_LW)
    for line in ['top', 'right']:
        ax.spines[line].set_visible(False)
    #plt.gcf().subplots_adjust(bottom=0.3)
    #plt.gcf().subplots_adjust(left=0.3)
    #plt.savefig(os.path.join(fig_path, 'migration_correction.png'), dpi=1200)


#def plot_average_s_protein(inf_file, index_file, ax, median=True, n_elim=0):
def plot_average_s_protein(inf_file, locations, counts, counts_normed, ax, median=True, n_elim=0):
    """ Given an inference result and the full-index.csv file, plots the average selection coefficient per protein vs. the density of mutations in that protein."""
    
    data     = np.load(inf_file, allow_pickle=True)
    inf_all  = data['selection']
    labs_all = [get_label2(i) for i in data['allele_number']]

    # Load data and sort proteins by which proteins has the highest density of mutations
    #locations, counts, counts_normed = dp.mutations_per_protein(index_file)
    proteins, labels, inf            = dp.separate_by_protein(inf_file)
    #for i in range(len(locations)):
    #    print(locations[i], counts_normed[i])
    #indices       = np.argsort(counts_normed)
    #locations     = np.array(locations)[indices]
    #counts_normed = np.array(counts_normed)[indices]
    
    if n_elim>0 and not median:
        idxs     = np.argsort(inf_all)[::-1][:-n_elim]
        labs_all = np.array(labs_all)[idxs]
        new_labels, new_inf = [], []
        for i in range(len(inf)):
            temp_labels, temp_inf = [], []
            for j in range(len(inf[i])):
                if labels[i][j] in labs_all:
                    temp_labels.append(labels[i][j])
                    temp_inf.append(inf[i][j])
            new_labels.append(temp_labels)
            new_inf.append(temp_inf)
        labels  = new_labels
        inf     = new_inf
    
    if median:
        inf_average = [np.median(i) for i in inf]
    else:
        inf_average = [np.mean(i) for i in inf]
    
    #counts_new = []
    #for i in range(len(locations)):
    #    counts_new.append(counts_normed[list(proteins).index(locations[i])])
    counts_new = counts_normed
        
    # Sort inferred coefficients into the correct order
    inf_new = []
    for i in locations:
        inf_new.append(inf_average[list(proteins).index(i)])       
    inf_new     = np.array(inf_new)
    inf_average = inf_new
    
    locations_new = []
    for i in locations:
        if i not in ['S', 'E', 'N', 'M']:
            locations_new.append(i.lower())
        else:
            locations_new.append(i)
    locations = locations_new
    
    #print(counts_normed)
    
    # Make and plot figure
    #fig = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH])
    #ax  = fig.add_subplot(1,1,1)
    ax.set_xlim(np.amin(counts_normed)-0.05, np.amax(counts_normed)+0.05)
    ax.set_xlabel('Mutation density', fontsize=AXES_FONTSIZE)
    ax.set_ylabel('Average selection (%)', fontsize=AXES_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
    ax.plot(counts_new, 100*np.array(inf_average), '.-', linewidth=0)
    
    correlation = calculate_correlation(counts_normed, inf_average)
    print('correlation between the average inferred selection coefficient and mutation density:', correlation)
    
    # Change the locations of the labels so that they don't overlap
    """
    for i in range(len(locations)-1):
        if abs(counts_normed[i+1]-counts_normed[i])<0.01 and abs(inf_new[i+1]-inf_new[i])<0.00005:
            if counts_normed[i+1]-counts_normed[i]>=0:
                counts_normed[i+1] += 0.01
            else:
                counts_normed[i]   += 0.01
            if inf_new[i+1]-inf_new[i]>=0:
                inf_new[i+1] += 0.00001
            else:
                inf_new[i]   += 0.00001
    """
            
    # Label the points with the protein names
    for i, txt in enumerate(locations):
        ax.annotate(txt, (counts_new[i]+0.01, (100*inf_average[i])+0.0005), xycoords='data', fontsize=6)
    
    for line in ['right', 'top']:
        ax.spines[line].set_visible(False)
        
    ax.text(0.8, 0.15, 'correlation: %.2f' % correlation, transform=ax.transAxes, fontsize=AXES_FONTSIZE)
    #for line in ['left', 'bottom']:
    #    ax.spines[line].set_linewidth(0.5)
    
    #plt.savefig(os.path.join(fig_path, out_file+'.png'), dpi=1200)
                    
"""
def plot_sequence_number(file, ax, k):
    
    data  = np.load(file, allow_pickle=True)
    title = file.split('/')[-1]
    title = title[:title.find('.')]
    title = title.split('-')[-1]
    #title = title.replace('-', ' ')
    title = ' '.join([i.capitalize() for i in title.split(' ')])
    nVec  = data['nVec']
    times = data['times']
    #print(times)
    counts = [np.sum(nVec[t]) for t in range(len(nVec))]
    new_times = []
    for i in range(len(times)):
        time = dt.date(2020, 1, 1) + dt.timedelta(int(times[i]))
        new_time = str(time.month) + "/" + str(time.day)
        new_times.append(new_time)
    #plt.figure(figsize=[20, 10])
    new_times = np.array(new_times)
    ax.bar(times, counts)
    ax.set_title(title, fontsize=6, pad=5)
    ax.set_xlabel('Date', fontsize=6)
    ax.set_xticks(times[::int(len(times)/5)])
    ax.set_xticklabels(new_times[::int(len(times)/5)])
    ax.tick_params(labelsize=4, length=2, pad=1, width=0.5)
    #ax.grid(b=True, axis='y', linewidth=0.5)
    if k%4==0:
        ax.set_ylabel('Number of samples', fontsize=6)
    for line in ['right', 'top']:
        ax.spines[line].set_visible(False)
    for line in ['left', 'bottom']:
        ax.spines[line].set_linewidth(0.5)
"""
        
def plot_sequence_number(file, ax, fixed_axis=False):
    """ Given the output file of count_sequences, plot the number of sequences as a function of time. """
    
    #data  = np.load(file, allow_pickle=True)
    title = file.split('/')[-1]
    title = title[:title.find('.')]
    #print(title)
    if title.split('-')[-2]=='california' and title.split('-')[-1]=='south':
        title = 'southern california'
    if title.split('-')[-1]=='england_wales_scotland':
        title = 'Great Britain'
    title = title.split('-')[-1]
    #title = title.replace('-', ' ')
    title = ' '.join([i.capitalize() for i in title.split(' ')])
    if 'Bosnia' in title:
        title = 'Bosnia-Herzegovina'
    
    data = pd.read_csv(file)
    times = list(data['date'])
    times, counts = np.unique(times, return_counts=True)
    
    ax.bar(times, counts, width=0.8, lw=0)
    ax.set_title(title, fontsize=AXES_FONTSIZE - 2, y=0.8)
    #ax.set_xlabel('Date', fontsize=6)
    #ax.set_xticks(times[::int(len(times)/num_ticks)])
    #ax.set_xticklabels(new_times[::int(len(times)/num_ticks)], rotation=40)
    #ax.tick_params(labelsize=4, length=2, pad=1, width=SPINE_LW)
    #ax.grid(b=True, axis='y', linewidth=0.5)
    #if k%4==0:
    #    ax.set_ylabel('# of samples', fontsize=AXES_FONTSIZE)
    for line in ['right', 'top']:
        ax.spines[line].set_visible(False)
    for line in ['left', 'bottom']:
        ax.spines[line].set_linewidth(SPINE_LW)
        

def location_from_filename(filename):
    """Find the location corresponding to a file name"""
    file_loc = filename.split('-')
    if file_loc[1]=='united kingdom' and file_loc[2]=='england_wales_scotland':
        filename = '-'.join(file_loc[:3]) + '---'
    filename  = filename[:filename.find('---')]
    namesplit = filename.split('-')
    while any([j in namesplit[-1] for j in NUMBERS]):
        namesplit = namesplit[:-1]
    location = '-'.join(namesplit)
    return location
        
        
def sampling_plots(folder, out_file=None, log=False):
    """ Given the folder containing the data files, plots the number of sampled genomes for each region."""
    
    N = len(os.listdir(folder))
    gen_dir = '/Users/brianlee/Desktop/sampling-plots'
    if not os.path.exists(gen_dir):
        os.mkdir(gen_dir)
    else:
        for file in os.listdir(gen_dir):
            path = os.path.join(gen_dir, file)
            os.remove(path)
        
    locs_all = []
    for i in range(N):
        filename = os.listdir(folder)[i]
        if 'sites' in filename:
            continue
        #print(filename)
        """
        file_loc = filename.split('-')
        if file_loc[1]=='united kingdom' and file_loc[2]=='england_wales_scotland':
            #filename = '-'.join(file_loc[:3]) + '---' + '-'.join(file_loc[-6:])
            filename = '-'.join(file_loc[:3]) + '---'
        #if file_loc[2]=='california' and file_loc[3]=='south':
        #    filename = 'southern california---'
        filename  = filename[:filename.find('---')]
        namesplit = filename.split('-')
        while any([j in namesplit[-1] for j in NUMBERS]):
            namesplit = namesplit[:-1]
        location = '-'.join(namesplit)
        """
        location = location_from_filename(filename)
        locs_all.append(location)
    locs_unique = np.unique(locs_all)
    #print(locs_unique)
    print(f'number of regions is {len(locs_unique)}')
    
    new_files = []
    num_seqs  = []
    all_times = []
    max_seqs  = []    # the maximum number of sequences that appear on any given day
    for k in range(len(locs_unique)):
        #print(locs_unique[k])
        df_temp = pd.DataFrame()
        for j in range(N):
            filename   = os.listdir(folder)[j]
            if 'sites' in filename:
                continue
            filepath = os.path.join(folder, filename)
            loc_temp = location_from_filename(filename)

            if loc_temp==locs_unique[k]:
                data = pd.read_csv(filepath)
                df_temp = pd.concat([df_temp, data])
            
        new_file = os.path.join(gen_dir, locs_unique[k]+'.npz')
        df_temp.to_csv(new_file, index=False)
        dates = list(df_temp['date'])
        times, counts = np.unique(dates, return_counts=True)
        new_files.append(new_file)
        num_seqs.append(np.sum(counts))
        all_times.append(times)
        max_seqs.append(np.amax(counts))
    
    # order regions by total number of samples
    sorter    = np.argsort(num_seqs)[::-1]
    new_files = np.array(new_files)[sorter]
    all_times = np.array(all_times)[sorter]
    max_seqs  = np.array(max_seqs)[sorter]
    
    # Make discrete possibilities for the y-axis scale
    ylim_intervals = [100, 500, 1000, 2500, 8000, 15000]
    ylims          = []
    for i in range(len(max_seqs)):
        limit = 0
        for j in ylim_intervals:
            if max_seqs[i] > j:
                continue
            limit = j
            break
        if limit == 0:
            print(f'maximum sequences in region {new_files[i]} is {max_seqs[i]}')
        ylims.append(limit)
    
    # order regions by their ylimits
    if not log:
        yticks    = [np.arange(0, ylims[i] + 1, int(ylims[i] / 5)) for i in range(len(ylims))]
        sorter    = np.argsort(ylims)[::-1]
        new_files = new_files[sorter]
        all_times = all_times[sorter]
        max_seqs  = max_seqs[sorter]
        ylims     = np.array(ylims)[sorter]
        yticks    = np.array(yticks)[sorter]
        ylims_unique  = pd.unique(ylims)
        group_lengths = np.array([len(np.nonzero(ylims==i)[0]) for i in ylims_unique])
    
    # Find the times to use for the x-axis tick labels
    times_full = np.unique([all_times[i][j] for i in range(len(all_times)) for j in range(len(all_times[i]))])
    month_dict = {1 : 'Jan', 2 : 'Feb', 3 : 'Mar', 4 : 'Apr', 5 : 'May', 6 : 'Jun',
                  7 : 'Jul', 8 : 'Aug', 9 : 'Sep', 10 : 'Oct', 11 : 'Nov', 12 : 'Dec'}
    
    xlabels = []
    xticks  = []
    for i in range(len(times_full)):
        time     = dt.date(2020, 1, 1) + dt.timedelta(int(times_full[i]))
        if time.day == 1:
            new_time = month_dict[int(time.month)] + ' ' + str(time.year)[2:] 
            xlabels.append(new_time)
            xticks.append(times_full[i])
    num_ticks = 5
    spacing   = int(len(xlabels) / num_ticks)
    xlabels   = np.array(xlabels)[::spacing]
    xticks    = np.array(xticks)[::spacing]
    
    # make figure
    plots_per_row = 7
    plots_per_col = int(len(locs_unique) / plots_per_row) + 1
    fig    = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 1.3])
    #grid   = matplotlib.gridspec.GridSpec(int(len(locs_unique)/plots_per_row)+1, plots_per_row, hspace=0.65, wspace=0.15, bottom=0.05, top=0.98, left=0.05, right=0.99)
    if log:
        grid   = matplotlib.gridspec.GridSpec(int(len(locs_unique)/plots_per_row)+1, plots_per_row, hspace=0.3, wspace=0.05, bottom=0.015, top=0.98, left=0.05, right=0.99)
    else:
        n_rows = np.sum([int(i / plots_per_row) + 1 for i in group_lengths])
        grid   = matplotlib.gridspec.GridSpec(n_rows, plots_per_row, hspace=0.2, wspace=0.025, bottom=0.025, top=0.98, left=0.05, right=0.99)
    ax_tot = fig.add_subplot(grid[:, :])
    for line in ['right', 'left', 'top', 'bottom']:
        ax_tot.spines[line].set_visible(False)
    ax_tot.set_ylabel('Number of sequences per day', fontsize=AXES_FONTSIZE, labelpad=15)
    ax_tot.set_xticks([])
    ax_tot.set_yticks([])
    
    # plot different groups on their own rows
    if not log:
        row_idx  = 0
        plot_idx = 0
        for i in range(len(group_lengths)):
            for j in range(group_lengths[i]):
                ax = fig.add_subplot(grid[int(j / plots_per_row) + row_idx, j % plots_per_row])
                ax.set_xlim(times_full[0], times_full[-1])
                ax.set_ylim(0, ylims_unique[i])
                ax.set_xticks(xticks)
                ax.set_yticks(yticks[plot_idx])
                if j % plots_per_row == 0:
                    ax.set_yticklabels(yticks[plot_idx])
                    if row_idx == n_rows -1:
                        ax.set_axisbelow(True)
                else:
                    ax.set_yticklabels([])
                if (j > group_lengths[i] - plots_per_row - 1) and (i == len(group_lengths) - 1):
                    ax.set_xticklabels(xlabels, rotation = 45)
                else:
                    ax.set_xticklabels([])
                ax.tick_params(labelsize=4, length=2, pad=1, width=SPINE_LW)
                plot_sequence_number(new_files[plot_idx], ax)
                plot_idx += 1
            #plot_idx += group_lengths[i]
            row_idx  += int(group_lengths[i] / plots_per_row) + 1          
    else:
        for k in range(len(new_files)):
            ax = fig.add_subplot(grid[int(k/plots_per_row), k%plots_per_row])
            ax.set_xlim(times_full[0], times_full[-1])
            #ax.set_ylim(0, ylims[k])
            ax.set_xticks(xticks)
            #if k > plots_per_row * (plots_per_col - 1) - 1:
            if k > len(new_files) - plots_per_row - 1:
                ax.set_xticklabels(xlabels, rotation = 45)
            else:
                ax.set_xticklabels([])
            #ax.set_yticks(yticks[k])
            ax.set_yscale('log')
            ax.tick_params(labelsize=4, length=2, pad=1, width=SPINE_LW)
            plot_sequence_number(new_files[k], ax)
            ax.set_ylim(1, 10000)
            ax.set_yticks([1, 10, 100, 1000, 10000, 100000])
            if k % plots_per_row != 0:
                ax.set_yticklabels([])
            #else:
            #    ax.set_yticks([1, 10, 100, 1000, 10000, 100000])
        
    if out_file:
        plt.savefig(os.path.join(fig_path, out_file+'.pdf'), dpi=800)
    else:
        plt.savefig(os.path.join(fig_path, 'sampling-dists.pdf'), dpi=800)
        
        
def selection_table(file, out=None, old=False):
    """ Makes a table of the largest selection coefficients"""
    df_sel = pd.read_csv(file, memory_map=True)

    s_sort       = np.argsort(df_sel['selection coefficient'])[::-1]
    table_data   = []
    table_labels = ['Rank', 'Protein', 'Mutation (nt)', 'Mutation (aa)', 'Selection (%)', 'Location', 'Phenotypic effect']
    errors       = []
    sel          = []
    for i in range(50):
        row   = []
        entry = df_sel.iloc[list(s_sort)[i]]
        loc   = str(entry['amino acid number in protein'])
        anc   = entry['amino acid mutation'][0]
        mut   = entry['amino acid mutation'][-1]
        nuc_n = entry['nucleotide number']
        nuc_m = entry['nucleotide']
        nuc_r = entry['reference nucleotide']
        if not old:
            stdev = entry['standard deviation']
        name  = anc + loc + mut
        nuc_name = nuc_r + str(nuc_n) + nuc_m
        row.append('%s' % (i + 1))
        row.append('%s' % entry['protein'])
        row.append('%s' % nuc_name)
        row.append('%s' % name)
        if not old:
            row.append('%.1f $\pm$ %.1f' % (100 * entry['selection coefficient'], 100 * stdev))
        else:
            row.append('%.1f' % (100 * entry['selection coefficient'])) 
        #row.append('%.1f' % (100 * stdev))                     
        row.append('')
        row.append('')
        if not old:
            errors.append(100 * stdev)
        sel.append(100 * entry['selection coefficient'])
        #print('%.3f\t%s\t%s' % (entry['selection coefficient'], entry['protein'], name))
        table_data.append(row)
    
    data = {'Rank' : np.array(table_data)[:, 0], 
            'Protein' : np.array(table_data)[:, 1],
            'Mutation' : np.array(table_data)[:, 3], 
            'Selection' : np.array(table_data)[:, 4]}
    df = pd.DataFrame(data=data)
    if out:
        df.to_csv(out + '.csv')
    else:
        df.to_csv(os.path.join(fig_path, 'selection-table.csv'))
    
    print('\\setlength{\\tabcolsep}{12pt}')
    print('\\begin{table}')
    print('\\centering')
    table_str = tabulate(table_data, headers=table_labels, tablefmt='latex_booktabs', numalign='left')
    table_str = table_str.replace('\\textbackslash{}', '\\')
    table_str = table_str.replace('\\$', '$')
    print(table_str)                              
    print('\\caption{Table of most highly selected mutations across the SARS-CoV-2 genome.}')
    print('\\label{table:selection}')
    print('\\end{table}')
    print(sel)
    print(errors)
    
    
def variant_selection_table(file, ind_file):
    """ Makes a table of the selection coefficients for the major variants. 
    file is a .npz file containing information for the variants.
    ind_file is a .csv file containing information for individual mutations."""
    
    def sort_by_contribution(sites, array, ind_file):
        """ Sorts the sites and array in a group in decending order of their selection coefficient."""
        #df      = pd.read_csv(ind_file)
        #idxs    = df['nucleotides']
        #aa_idxs = df['amino acid number in protein']
        #prot    = df['protein']
        data    = np.load(ind_file, allow_pickle=True)
        s_ind   = data['selection']
        alleles = data['allele_number']
        labels  = [dp.get_label_new(i) for i in alleles]
        s_sites = [s_ind[labels.index(i)] for i in sites]
        mask    = np.argsort(s_sites)[::-1]
        sites   = np.array(sites)[mask]
        array   = np.array(array)[mask]
        return sites, array
        
    
    df = pd.read_csv(file)
    
    name_dict = {'Alpha' : 'B.1.1.7', 'Beta' : 'B.1.351', 'Gamma' : 'P.1', 'Delta' : 'B.1.617.2', 
                 'Lambda' : 'C.37', 'Epsilon' : 'B.1.427//B.1.429', '20e_eu1' : 'B.1.177', 'B.1' : 'B.1',
                 'Ba.1' : 'BA.1', 'Ba.2.12.1' : 'BA.2.12.1', 'Ba.2' : 'Ba.2', 'Ba.4' : 'Ba.4', 'Ba.5' : 'Ba.5'}
    
    variant_names = list(df['variant_names'])
    group         = [i.split(' ')[0] for i in variant_names]
    mask          = np.array(group)!='Group'
    df_var        = df[mask]
    var_names     = [i.capitalize() for i in df_var['variant_names']]
    synonymous    = [i.split('/')   for i in df_var['synonymous']]
    aa_muts       = [i.split('/')   for i in df_var['aa_mutations']]
    sites         = [i.split('/')   for i in df_var['sites']]
    mut_nucs      = [i.split('/')   for i in df_var['mutant_nuc']]
    pango_name    = [name_dict[i]   for i in var_names]
    selection     = list(df_var['selection_coefficient'])
    muts_full     = []
    for i in range(len(sites)):
        new_sites   = [j.split('-') for j in sites[i]]
        prots_temp  = [new_sites[j][0]  for j in range(len(new_sites))  if synonymous[i][j]!='S']
        codons_temp = [new_sites[j][1]  for j in range(len(new_sites))  if synonymous[i][j]!='S']
        aa_temp     = [aa_muts[i][j]    for j in range(len(aa_muts[i])) if synonymous[i][j]!='S']
        aa_init     = [j[0]  for j in aa_temp]
        aa_final    = [j[-1] for j in aa_temp]
        mutations   = [prots_temp[j] + '-' + aa_init[j] + codons_temp[j] + aa_final[j] for j in range(len(prots_temp))]
        site_labs   = [sites[i][j] + '-' + mut_nucs[i][j] for j in range(len(mut_nucs[i])) if synonymous[i][j]!='S']
        
        sites_var, muts_var = sort_by_contribution(site_labs, mutations, ind_file)
        muts_var = pd.unique(muts_var)
        muts_full.append(', '.join(muts_var))
        #muts_full.append(', '.join(mutations))
    
    table_labels = ['Variant', 'Pango Lineage', 'Selection Coefficient (%)',  'Mutations']
    table_data   = []
    sorter       = np.argsort(selection)[::-1]
    for i in range(len(df_var)):
        idx = sorter[i]
        row = []
        if var_names[idx]=='20e_eu1':
            var_names[idx]='20E-EU1'
        row.append(var_names[idx])
        row.append(pango_name[idx])
        row.append('%.1f' % (100 * selection[idx]))
        row.append(muts_full[idx])
        table_data.append(row)
    
    print('\\setlength{\\tabcolsep}{12pt}')
    print('\\begin{table}')
    print('\\centering') 
    print(tabulate(table_data, headers=table_labels, tablefmt='latex_booktabs', numalign='left'))
    print('\\caption{Table of selection coefficients for groups of mutations. Mutations that contribute most strongly to selection are listed first.}')
    print('\\label{table:variant_selection}')
    print('\\end{table}')
    
    
def early_detection_table(file):
    """ Make a table of the variants that are detected early and the number of times they are detected early"""
    
    df     = pd.read_csv(file)
    index  = np.arange(1, len(df)+1)
    var    = list(df['Variants'])
    num    = list(df['Number of detections'])
    muts   = [i.split(' ') for i in list(df['Mutations'])]
    muts   = [', '.join(i) for i in muts]
    labels = ['Variants', 'Number of detections', 'Mutations']
    table_data = []
    for i in range(len(index)):
        row = []
        #row.append(index[i])
        row.append(var[i].replace(' ', ', '))
        row.append(num[i])
        row.append(muts[i])
        table_data.append(row)
        
    print('\\setlength{\\tabcolsep}{12pt}')
    print('\\begin{table*}')
    print('\\centering')
    print('\\footnotesize')
    table_str = tabulate(table_data, headers=labels, tablefmt='latex_booktabs', numalign='left')
    print(table_str)                              
    print('\\caption{Table of variants that are detected as increasing transmission and the number of regions they are detected.}')
    print('\\label{table:early_detection}')
    print('\\end{table*}')
    

def linkage_plot(linked_sites, link_labels=None):
    """ Make Hive plots for the groups of linked coefficients given in linked_sites"""
    
    if type(linked_sites)==str:
        linked = np.load(linked_sites, allow_pickle=True)
    else:
        linked = linked_sites
        
    working_dir = os.getcwd()
    os.chdir('/Users/brianlee/SARS-CoV-2-Data/Processing-files')
    import data_processing as dp
    os.chdir(working_dir)
    if not link_labels:
        link_labels = [dp.get_label_old(linked[i][0]) for i in range(len(linked))]
        
    unmasked   = s_unmasked = np.load('/Users/brianlee/Python/MPL/6-29-20-epidemiological/infer-11-30-g1-30.npz', allow_pickle=True)
    s_unmasked = unmasked['selection']
    alleles    = unmasked['allele_number']
    labels     = [dp.get_label_old(i) for i in alleles]
    s_masked   = np.zeros((len(s_unmasked), len(s_unmasked)))
    counter    = 0
    for file in os.listdir(os.fsencode(LINK_DIR)):
        filename    = os.fsdecode(file)
        if filename != 'linkage-coefficients.npz':
            filepath    = os.path.join(LINK_DIR, filename)
            #print(np.load(filepath, allow_pickle=True)['selection'])
            s_masked[list(labels).index(filename[:-4])] = np.array(np.load(filepath, allow_pickle=True)['selection'])
            counter += 1
    effect = s_unmasked - s_masked
    
    # make index file
    ### RIGHT NOW THIS DOESN'T ALLOW FOR INSERTIONS 
    ref_seq = list(pd.read_csv('/Users/brianlee/SARS-CoV-2-Data/11-30/africa-south africa-None-None-None-None-index.csv').reference_nucleotide)
    if not os.path.exists(os.path.join(INFERENCE_DIR, 'index-full-11-30.csv')):
        g = open(os.path.join(INFERENCE_DIR, 'index-full-11-30.csv'), mode='w')
        g.write('alignment,polymorphic,reference_index,reference_nucleotide\n')
        for i in range(29902):
            if i in list(alleles):
                poly_idx = str(list(alleles).index(i))
            else:
                poly_idx = 'NA'
            g.write('%d,%s,%d,%s\n' % (i, poly_idx, i+1, ref_seq[i]))
        g.close()

    # make change in selection files
    for i in range(len(linked)):
        file_name   = '%s-delta-s.csv' % link_labels[i]
        f = open(os.path.join(INFERENCE_DIR, file_name), mode='w')
        f.write('mask_polymorphic_index,mask_nucleotide,target_polymorphic_index,target_nucleotide,effect,distance\n')
        for j in range(len(s_unmasked)):
            for k in range(len(s_unmasked)):
                if alleles[k] in list(linked[i]):
                    f.write('%d,%s,%d,%s,' % (j,str(1),k,str(1)))
                    f.write('%1e,' % (effect[j][k]))
                    f.write('%d\n' % (np.fabs(alleles[j] - alleles[k])))
        f.close()
        
    # create figure and subplots
    num_rows  = int((len(linked)+1)/3)+1
    fig       = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, (DOUBLE_COLUMN_WIDTH / 3) * num_rows])
    grid      = matplotlib.gridspec.GridSpec(num_rows, 3, figure=fig, wspace=0.05, hspace=0.05)
    for i in range(3):
        for j in range(num_rows):
            if 3*(i+1) + j <= len(linked):
                print(link_labels[3*i+j])
                ax        = fig.add_subplot(grid[j, i])
                file_name = '%s-delta-s.csv' % link_labels[3*i + j]
                ax.set_title(labels[3*i+j])
                plot_hive(ax, file_name, index_file=os.path.join(INFERENCE_DIR, 'index-full-11-30.csv'), sites=linked[3*i+j], link_site=link_labels[3*i+j])
    plt.savefig(os.path.join(fig_path, 'hive-plots.png'), dpi=1200)
        

def plot_hive(ax_hive, tag, index_file=None, sites=None, link_site=None, **pdata):
    """
    Hive plot of \Delta s effects. sites is a list of sites to mark. 
    link_site is the name of the prefix for the delta_s file that contains the effect of masking sites on specific other sites.
    if link_file is none then the plot shows the effect of all sites on all other sites.
    """

    # import stored data
    if not link_site:
        df_ds    = pd.read_csv('%s/delta-s.csv' %    (INFERENCE_DIR),            comment='#', memory_map=True)
        out_path = os.path.join(fig_path, 'hive-plot.png')
    else:
        df_ds    = pd.read_csv('%s/%s-delta-s.csv' % (INFERENCE_DIR, link_site), comment='#', memory_map=True)
        out_path = os.path.join(fig_path, 'hive-plot-'+link_site+'.png')

    if not index_file:
        df_index = pd.read_csv('%s/index-full.csv' % (INFERENCE_DIR),   comment='#', memory_map=True)
    else:
        df_index = pd.read_csv(os.path.join(INFERENCE_DIR, index_file), comment='#', memory_map=True)
    #linked   = np.load('%s/linked-sites.npy' % (DATA_DIR), allow_pickle=True)
    #inferred = np.load(INFER_FILE, allow_pickle=True)
    
    L        = len(df_index)
    #nuc_idxs = inferred['allele_number']   # the nucleotide indices of the inferred sites, used to label specific sites
    
    # set epitope labels
    """
    epitope_start = []
    epitope_end   = []
    epitope_label = list(set(list(df_index[pd.notnull(df_index.epitope)].epitope)))
    print(tag, epitope_label)
    
    for e in epitope_label:
        epitope_start.append(np.min(df_index[df_index.epitope==e].alignment))
        epitope_end.append(np.max(df_index[df_index.epitope==e].alignment))
   """
    # PLOT FIGURE

    # populate links

    idx_mask_pos  = []
    idx_delta_pos = []
    ds_pos        = []
    idx_mask_neg  = []
    idx_delta_neg = []
    ds_neg        = []

    c_cutoff  = 0.0004
    ds_cutoff = 0.0005
    ds_max    = 0.008

    df_ds = df_ds[np.fabs(df_ds.effect)>ds_cutoff]
    for it, entry in df_ds.iterrows():
        if (entry.mask_polymorphic_index==entry.target_polymorphic_index#):
#            continue
            and entry.mask_nucleotide==entry.target_nucleotide):
            continue
        if entry.effect>0:
            mask_alignment_index   = df_index[df_index.polymorphic==  entry.mask_polymorphic_index].iloc[0].alignment
            target_alignment_index = df_index[df_index.polymorphic==entry.target_polymorphic_index].iloc[0].alignment
            idx_mask_pos.append(mask_alignment_index)
            idx_delta_pos.append(target_alignment_index)
            ds_pos.append(entry.effect)
        elif entry.effect<0:
            mask_alignment_index   = df_index[df_index.polymorphic==  entry.mask_polymorphic_index].iloc[0].alignment
            target_alignment_index = df_index[df_index.polymorphic==entry.target_polymorphic_index].iloc[0].alignment
            idx_mask_neg.append(mask_alignment_index)
            idx_delta_neg.append(target_alignment_index)
            ds_neg.append(entry.effect)
    
    ds_pos           = (np.array(ds_pos) - c_cutoff) / ds_max
    ds_pos[ds_pos>1] = 1
    pos_sort         = np.argsort(ds_pos)[::-1]
    idx_mask_pos     = np.array(idx_mask_pos)[pos_sort]
    idx_delta_pos    = np.array(idx_delta_pos)[pos_sort]
    ds_pos           = ds_pos[pos_sort]

    ds_neg           = (np.fabs(np.array(ds_neg)) - c_cutoff) / ds_max
    ds_neg[ds_neg>1] = 1
    neg_sort         = np.argsort(ds_neg)[::-1]
    idx_mask_neg     = np.array(idx_mask_neg)[neg_sort]
    idx_delta_neg    = np.array(idx_delta_neg)[neg_sort]
    ds_neg           = ds_neg[neg_sort]

    # arc plot for large values of Delta s

    c_pos  = LCOLOR
    c_neg  = LCOLOR
    c_circ = { True : c_neg, False : c_pos }

    r_min     = 0.05
    r_max     = 1.00
    r_norm    = float(L) / (r_max - r_min)
    arc_mult  = SIZELINE * ds_max / 1.5
    arc_alpha = 1

    mask_angle_p =  -np.pi/2
    mask_angle_n = 3*np.pi/2
    ds_pos_angle =   np.pi/6
    ds_neg_angle = 5*np.pi/6
    da           = 0.005

    ### CHANGE THE COLORS HERE ###
    circ_color  = [hls_to_rgb(0.02, 0.53 * ds + 1. * (1 - ds), 0.83) for ds in ds_pos]
    circ_color += [hls_to_rgb(0.58, 0.53 * ds + 1. * (1 - ds), 0.60) for ds in ds_neg]
    circ_rad    = [[r_min + idx_mask_pos[i]/r_norm, r_min + idx_delta_pos[i]/r_norm] for i in range(len(ds_pos))]
    circ_rad   += [[r_min + idx_mask_neg[i]/r_norm, r_min + idx_delta_neg[i]/r_norm] for i in range(len(ds_neg))]
    circ_arc    = [dict(lw=arc_mult * np.fabs(0.1) / ds_cutoff, alpha=arc_alpha) for ds in ds_pos]
    circ_arc   += [dict(lw=arc_mult * np.fabs(0.1) / ds_cutoff, alpha=arc_alpha) for ds in ds_neg]
    circ_angle  = [[mask_angle_p+da/r[0], ds_pos_angle-da/r[1]] for r in circ_rad[:len(ds_pos)]]
    circ_angle += [[mask_angle_n-da/r[0], ds_neg_angle+da/r[1]] for r in circ_rad[len(ds_pos):]]
    circ_bezrad = [(r[0]+r[1])/2 for r in circ_rad]
    circ_x      = [i for i in ds_pos] + [i for i in ds_neg]
    circ_y      = [i for i in ds_pos] + [i for i in ds_neg]
    
    # mark sites
    
    for i in range(len(sites)):
        rad = r_min + (sites[i] / r_norm)
        scatter_x = [rad * np.cos(mask_angle_p),
                     rad * np.cos(ds_pos_angle),
                     rad * np.cos(ds_neg_angle) ]
        scatter_y = [rad * np.sin(mask_angle_p),
                     rad * np.sin(ds_pos_angle),
                     rad * np.sin(ds_neg_angle) ]
        smallprops = dict(lw=0, marker='o', s=1.0*SMALLSIZEDOT, zorder=9999, clip_on=False)
        bigprops   = dict(lw=0, marker='o', s=1.5*SMALLSIZEDOT, zorder=9998, clip_on=False)
        mp.scatter(ax=ax_hive, x=[scatter_x], y=[scatter_y], colors=['#FFFFFF'], plotprops=smallprops)
        mp.scatter(ax=ax_hive, x=[scatter_x], y=[scatter_y], colors=[BKCOLOR],   plotprops=  bigprops)
    
    # mark epitopes
    """
    for i in range(len(epitope_label)):
        r_mid = r_min + (epitope_start[i] + epitope_end[i]) / (2 * r_norm)
        scatter_x = [r_mid * np.cos(mask_angle_p),
                     r_mid * np.cos(ds_pos_angle),
                     r_mid * np.cos(ds_neg_angle) ]
        scatter_y = [r_mid * np.sin(mask_angle_p),
                     r_mid * np.sin(ds_pos_angle),
                     r_mid * np.sin(ds_neg_angle) ]
        smallprops = dict(lw=0, marker='o', s=1.0*SMALLSIZEDOT, zorder=9999, clip_on=False)
        bigprops   = dict(lw=0, marker='o', s=1.7*SMALLSIZEDOT, zorder=9998, clip_on=False)
        mp.scatter(ax=ax_hive, x=[scatter_x], y=[scatter_y], colors=['#FFFFFF'], plotprops=smallprops)
        mp.scatter(ax=ax_hive, x=[scatter_x], y=[scatter_y], colors=[BKCOLOR],   plotprops=  bigprops)
    """

    # Make plot

    pprops = { 'colors':   circ_color,
               'xlim':     [-1.0, 1.0],
               'ylim':     [-1.0, 1.0],
               'size':     L,
               'rad':      circ_rad,
               'arcprops': circ_arc,
               'angle':    circ_angle,
               'bezrad':   circ_bezrad,
               'noaxes':   True }

    plotprops = dict(lw=AXWIDTH, ls='-', zorder=999)
    line_x = [[r_min * np.cos(mask_angle_p), r_max * np.cos(mask_angle_p)],
              [r_min * np.cos(ds_pos_angle), r_max * np.cos(ds_pos_angle)],
              [r_min * np.cos(ds_neg_angle), r_max * np.cos(ds_neg_angle)] ]
    line_y = [[r_min * np.sin(mask_angle_p), r_max * np.sin(mask_angle_p)],
              [r_min * np.sin(ds_pos_angle), r_max * np.sin(ds_pos_angle)],
              [r_min * np.sin(ds_neg_angle), r_max * np.sin(ds_neg_angle)] ]
    mp.line(ax=ax_hive, x=line_x, y=line_y, colors=[BKCOLOR for i in range(len(line_x))], plotprops=plotprops)

    mp.plot(type='circos', ax=ax_hive, x=circ_x, y=circ_y, **pprops)
    
    if not link_site:
        plt.savefig(out_path, dpi=DPI)
        
        
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
    

def get_label_old(i):
    """ For a SARS-CoV-2 reference sequence index i, return the label in the form 
    'coding region - protein number-nucleotide in codon number'. 
    For example, 'ORF1b-204-1'."""
    i = int(i)
    frame_shift = str(i - get_codon_start_index(i))
    if   (25393<=i<=26220):
        return "ORF3a-" + str(int((i - 25393 + 1) / 3) + 1) + '-' + frame_shift
    elif (26245<=i<=26472):
        return "E-"     + str(int((i - 26245 + 1) / 3) + 1) + '-' + frame_shift
    elif (27202<=i<=27387):
        return "ORF6-"  + str(int((i - 27202 + 1) / 3) + 1) + '-' + frame_shift
    elif (27394<=i<=27759):
        return "ORF7a-" + str(int((i - 27394 + 1) / 3) + 1) + '-' + frame_shift
    elif (27756<=i<=27887):
        return "ORF7b-" + str(int((i - 27756 + 1) / 3) + 1) + '-' + frame_shift
    elif (  266<=i<=805):
        return "NSP1-"  + str(int((i - 266   + 1) / 3) + 1) + '-' + frame_shift
    elif (  806<=i<=2719):
        return "NSP2-"  + str(int((i - 806   + 1) / 3) + 1) + '-' + frame_shift
    elif ( 2720<=i<=8554):
        return "NSP3-"  + str(int((i - 2720  + 1) / 3) + 1) + '-' + frame_shift
            # Compound protein containing a proteinase, a phosphoesterase transmembrane domain 1, etc.
    elif ( 8555<=i<=10054):
        return "NSP4-"  + str(int((i - 8555  + 1) / 3) + 1) + '-' + frame_shift
            # Transmembrane domain 2
    elif (10055<=i<=10972):
        return "NSP5-"  + str(int((i - 10055 + 1) / 3) + 1) + '-' + frame_shift
            # Main proteinase
    elif (10973<=i<=11842):
        return "NSP6-"  + str(int((i - 10973 + 1) / 3) + 1) + '-' + frame_shift
            # Putative transmembrane domain
    elif (11843<=i<=12091):
        return "NSP7-"  + str(int((i - 11843 + 1) / 3) + 1) + '-' + frame_shift
    elif (12092<=i<=12685):
        return "NSP8-"  + str(int((i - 12092 + 1) / 3) + 1) + '-' + frame_shift
    elif (12686<=i<=13024):
        return "NSP9-"  + str(int((i - 12686 + 1) / 3) + 1) + '-' + frame_shift
            # ssRNA-binding protein
    elif (13025<=i<=13441):
        return "NSP10-" + str(int((i - 13025 + 1) / 3) + 1) + '-' + frame_shift
            # CysHis, formerly growth-factor-like protein
    # Check that aa indexing is correct for NSP12, becuase there is a -1 ribosomal frameshift at 13468. It is not, because the amino acids in the first frame
    # need to be added to the counter in the second frame.
    elif (13442<=i<=13467):
        return "NSP12-" + str(int((i - 13442 + 1) / 3) + 1) + '-' + frame_shift
    elif (13468<=i<=16236):
        return "NSP12-" + str(int((i - 13468 + 1) / 3) + 1) + '-' + frame_shift
            # RNA-dependent RNA polymerase
    elif (16237<=i<=18039):
        return "NSP13-" + str(int((i - 16237 + 1) / 3) + 1) + '-' + frame_shift
            # Helicase
    elif (18040<=i<=19620):
        return "NSP14-" + str(int((i - 18040 + 1) / 3) + 1) + '-' + frame_shift
            # 3' - 5' exonuclease
    elif (19621<=i<=20658):
        return "NSP15-" + str(int((i - 19621 + 1) / 3) + 1) + '-' + frame_shift
            # endoRNAse
    elif (20659<=i<=21552):
        return "NSP16-" + str(int((i - 20659 + 1) / 3) + 1) + '-' + frame_shift
            # 2'-O-ribose methyltransferase
    elif (21563<=i<=25384):
        return "S-"     + str(int((i - 21563 + 1) / 3) + 1) + '-' + frame_shift
    elif (28274<=i<=29533):
        return "N-"     + str(int((i - 28274 + 1) / 3) + 1) + '-' + frame_shift
    elif (29558<=i<=29674):
        return "ORF10-" + str(int((i - 29558 + 1) / 3) + 1) + '-' + frame_shift
    elif (26523<=i<=27191):
        return "M-"     + str(int((i - 26523 + 1) / 3) + 1) + '-' + frame_shift
    elif (27894<=i<=28259):
        return "ORF8-"  + str(int((i - 27894 + 1) / 3) + 1) + '-' + frame_shift
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
    
"""    
def correlation_plots(file, syn_file, index_file, out_file='correlation-plots', median=True):
    # Plots the correlations of nonsynonymous enrichment, average inferred selection, and mutation density in each protein.
    
    fig = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH])
    grid = matplotlib.gridspec.GridSpec(3, 1, hspace=0.3, figure=fig, bottom=0.05, left=0.075, right=0.95, top=0.95)
    fig.text(0.05,    0.64,   'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.05,    0.31,   'c', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.05,    0.97,   'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    
    # making the scatterplots comparing nonsynonymous enrichment, average inferred selection, and mutation density in each protein
    ax_syn_dens = fig.add_subplot(grid[0, :])
    ax_syn_sel  = fig.add_subplot(grid[1, :])
    ax_sel_dens = fig.add_subplot(grid[2, :])
    axes = [ax_syn_dens, ax_syn_sel, ax_sel_dens]
    syn_enrichment_scatter(file, syn_file, index_file, ax_syn_dens)
    nonsyn_selection_plot(file, syn_file, index_file, ax_syn_sel, median=median)
    plot_average_s_protein(file, index_file, ax_sel_dens, median=median)
    for axis in axes:
        axis.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
        for line in ['top', 'right']:
            axis.spines[line].set_visible(False)
        for line in ['bottom', 'left']:
            axis.spines[line].set_linewidth(SPINE_LW)
    fig.savefig(os.path.join(fig_path, out_file + '.png'), dpi=1200)
"""

def correlation_plots(file, syn_file, data_dir, ref_file=None, out_file='correlation-plots', median=True):
    """ Plots the correlations of nonsynonymous enrichment, average inferred selection, and mutation density in each protein."""
    
    # finding the mutations per protein
    #locations, counts, counts_normed = dp.mutations_per_protein(data_dir, ref_tag=ref_file, index=False)
    locations, counts, counts_normed = dp.mutation_percentage(data_dir)    # it is assumed here that data_dir is actually the nucleotide counts file
    #locations, counts, counts_normed = dp.muts_per_protein_alt(data_dir)
    
    fig = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH])
    grid = matplotlib.gridspec.GridSpec(3, 1, hspace=0.3, figure=fig, bottom=0.05, left=0.075, right=0.95, top=0.95)
    fig.text(0.05,    0.64,   'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.05,    0.31,   'c', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.05,    0.97,   'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    
    # making the scatterplots comparing nonsynonymous enrichment, average inferred selection, and mutation density in each protein
    ax_syn_dens = fig.add_subplot(grid[0, :])
    ax_syn_sel  = fig.add_subplot(grid[1, :])
    ax_sel_dens = fig.add_subplot(grid[2, :])
    axes = [ax_syn_dens, ax_syn_sel, ax_sel_dens]
    
    #syn_enrichment_scatter(file, syn_file, index_file, ax_syn_dens)
    #nonsyn_selection_plot(file, syn_file, index_file, ax_syn_sel, median=median)
    #plot_average_s_protein(file, index_file, ax_sel_dens, median=median)
    syn_enrichment_scatter(file, syn_file, locations, counts, counts_normed, ax_syn_dens)
    nonsyn_selection_plot(file, syn_file, locations, counts, counts_normed, ax_syn_sel, median=median)
    plot_average_s_protein(file, locations, counts, counts_normed, ax_sel_dens, median=median)
    for axis in axes:
        axis.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
        for line in ['top', 'right']:
            axis.spines[line].set_visible(False)
        for line in ['bottom', 'left']:
            axis.spines[line].set_linewidth(SPINE_LW)
    fig.savefig(os.path.join(fig_path, out_file + '.png'), dpi=1200)
    
    
    
def correlation_plots_new(file, syn_file, count_file, ref_file=None, out_file='correlation-plots', median=True):
    """ Plots the correlations of nonsynonymous enrichment, average inferred selection, and mutation density in each protein."""
    
    # finding the mutations per protein
    #locations, counts, counts_normed = dp.mutations_per_protein(data_dir, ref_tag=ref_file, index=False)
    #locations, counts, counts_normed = dp.mutation_percentage(data_dir)    # it is assumed here that data_dir is actually the nucleotide counts file
    #locations, counts, counts_normed = dp.muts_per_protein_alt(data_dir)
    prots1, mutation_percentage = dp.mutation_percentage(count_file)
    prots2, nonsyn_percentage   = dp.nonsyn_percentage(count_file, syn_file)
    assert(np.all(np.array(prots1)==np.array(prots2)))
    
    # finding the average selection coefficient in each protein
    data     = np.load(file, allow_pickle=True)
    inferred = data['selection']
    labels   = [dp.get_label2(i) for i in data['allele_number']]
    
    proteins, site_labels, inf = dp.separate_by_protein(file)
    idxs     = np.argsort(inferred)[::-1]
    labs_all = np.array(labels)[idxs]
    new_labels, new_inf = [], []
    for i in range(len(inf)):
        temp_labels, temp_inf = [], []
        for j in range(len(inf[i])):
            if site_labels[i][j] in labs_all:
                temp_labels.append(site_labels[i][j])
                temp_inf.append(inf[i][j])
        new_labels.append(temp_labels)
        new_inf.append(temp_inf)
    labels  = new_labels
    inf     = new_inf
    
    if median:
        inf_average = [np.median(i) for i in inf]
    else:
        inf_average = [np.mean(i) for i in inf]
    
    fig = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH])
    grid = matplotlib.gridspec.GridSpec(3, 1, hspace=0.3, figure=fig, bottom=0.05, left=0.075, right=0.95, top=0.95)
    fig.text(0.05,    0.64,   'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.05,    0.31,   'c', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.05,    0.97,   'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    
    # making the scatterplots comparing nonsynonymous enrichment, average inferred selection, and mutation density in each protein
    ax1 = fig.add_subplot(grid[0, :])
    ax2 = fig.add_subplot(grid[1, :])
    ax3 = fig.add_subplot(grid[2, :])
    axes = [ax1, ax2, ax3]
    
    locations_new = []
    for i in prots1:
        if i not in ['S', 'E', 'N', 'M']:
            locations_new.append(i.lower())
        else:
            locations_new.append(i)
    proteins = locations_new
    
    print(np.shape(proteins), proteins)
    print(np.shape(mutation_percentage), np.shape(nonsyn_percentage), np.shape(inf_average))
    ax1.set_ylabel('Nonsynonymous (%)', fontsize=AXES_FONTSIZE)
    ax1.set_xlabel('Mutation density', fontsize=AXES_FONTSIZE)
    ax1.plot(mutation_percentage, nonsyn_percentage, '.-', linewidth=0)
    ax1.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
    for i, txt in enumerate(proteins):
        ax1.annotate(txt, (mutation_percentage[i] + 0.002, (nonsyn_percentage[i] + 0.01)), fontsize=6)
        
    correlation = calculate_correlation(mutation_percentage, nonsyn_percentage)
    print("correlation between nonsynonymous and mutation density:", correlation)
    ax1.text(0.8, 0.15, 'correlation: %.2f' % correlation, transform=ax1.transAxes, fontsize=8)
    
    spearcorr1 = spstats.spearmanr(mutation_percentage, nonsyn_percentage)
    pearcorr1  = spstats.pearsonr(mutation_percentage, nonsyn_percentage)
    print('mutation percentage and nonsynonymous percentage')
    print('spearman correlation:\t', spearcorr1)
    print('pearson  correlation:\t', pearcorr1)

    
    ax2.set_ylabel('Nonsynonymous (%)', fontsize=AXES_FONTSIZE)
    ax2.set_xlabel('Average inferred selection (%)', fontsize=AXES_FONTSIZE)
    ax2.plot(100*np.array(inf_average), nonsyn_percentage, '.-', linewidth=0)
    ax2.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
    for i, txt in enumerate(proteins):
        ax2.annotate(txt, (np.array(inf_average[i])*100 + 0.0002, (nonsyn_percentage[i] + 0.01)), fontsize=6)
    
    correlation = calculate_correlation(inf_average, nonsyn_percentage)
    print("correlation between nonsynonymous and average inferred coefficient:", correlation)
    ax2.text(0.8, 0.15, 'correlation: %.2f' % correlation, transform=ax2.transAxes, fontsize=8)
    
    spearcorr2 = spstats.spearmanr(inf_average, nonsyn_percentage)
    pearcorr2  = spstats.pearsonr(inf_average, nonsyn_percentage)
    print('average selection and nonsynonymous percentage')
    print('spearman correlation:\t', spearcorr2)
    print('pearson  correlation:\t', pearcorr2)
    
    
    ax3.set_ylabel('Mutation density (%)', fontsize=AXES_FONTSIZE)
    ax3.set_xlabel('Average inferred selection (%)', fontsize=AXES_FONTSIZE)
    ax3.plot(100*np.array(inf_average), mutation_percentage, '.-', linewidth=0)
    ax3.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
    for i, txt in enumerate(proteins):
        ax3.annotate(txt, (np.array(inf_average[i])*100 + 0.0002, (mutation_percentage[i] + 0.01)), fontsize=6)
    
    correlation = calculate_correlation(inf_average, mutation_percentage)
    print("correlation between mutation density and average inferred coefficient:", correlation)
    ax3.text(0.8, 0.15, 'correlation: %.2f' % correlation, transform=ax3.transAxes, fontsize=8)
    
    spearcorr3 = spstats.spearmanr(inf_average, mutation_percentage)
    pearcorr3  = spstats.pearsonr(inf_average, mutation_percentage)
    print('average selection and mutation percentage')
    print('spearman correlation:\t', spearcorr3)
    print('pearson  correlation:\t', pearcorr3)
    
    
    for axis in axes:
        axis.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
        for line in ['top', 'right']:
            axis.spines[line].set_visible(False)
        for line in ['bottom', 'left']:
            axis.spines[line].set_linewidth(SPINE_LW)
            
    #fig.show()
    fig.savefig(os.path.join(fig_path, out_file + '.png'), dpi=1200)


    
def dist_syn_plots(file, link_file, link_file_all, syn_file, index_file=None, full_tv_file=None, syn_lower_limit=55, correlation_plots=False, out_file='syn_dist_plot'):
    """ Makes a plot containing the distribution of inferred coefficients and the enrichment of nonsynonymous mutations.
    Combines selection_dist_plot and syn_plot functions.
    correlation_plots determines whether or not the 3 plots containing the correlations are made."""
    
    # loading and processing the data
    data            = np.load(file, allow_pickle=True)
    linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in EVERY region
    linked_anywhere = np.load(link_file_all, allow_pickle=True)    # sites that are fully linked in ANY region
    syn_data        = np.load(syn_file, allow_pickle=True)
    allele_number   = data['allele_number']
    labels          = [get_label(i) for i in allele_number]
    types_temp      = syn_data['types']
    type_locs       = syn_data['locations']
    linked_all      = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    inferred        = data['selection']
    labels          = [get_label(i) for i in data['allele_number']]
    inferred_link   = np.zeros(len(linked_sites))
    digits          = len(str(len(inferred)))
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
                inferred_link[i] += inferred[np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
    for i in range(len(labels)):
        if labels[i] not in linked_all:
            inferred_new.append(inferred[i])
    inferred_new = inferred_new + list(inferred_link)
    
    # Null distribution
    if full_tv_file:
        inferred_null = find_null_distribution(full_tv_file, link_file)
    
    # creating figure and subfigures
    if correlation_plots:
        fig  = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH])
        grid = matplotlib.gridspec.GridSpec(4, 2, hspace=0.3, wspace=0.2, figure=fig, bottom=0.05, left=0.075, right=0.95, top=0.95)
        lower_grid = grid[1:, :].subgridspec(3, 1, hspace=0.3, wspace=0.2)
    else:
        fig  = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH / 1.618])
        grid = matplotlib.gridspec.GridSpec(1, 2, hspace=0.3, wspace=0.2, figure=fig, bottom=0.175, left=0.05, right=0.975, top=0.875)
        #grid = matplotlib.gridspec.GridSpec(1, 3, hspace=0.3, wspace=0.2, figure=fig, bottom=0.175, left=0.05, right=0.975, top=0.875)
    #left_grid  = grid[0, 0].subgridspec(1, 1)
    #right_grid = grid[0, 1].subgridspec(1, 1)
    
    if correlation_plots:
        fig.text(0.05,    0.72,   'c', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
        fig.text(0.05,    0.48,   'd', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
        fig.text(0.05,    0.24,   'e', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
        fig.text(0.05,    0.975,  'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
        fig.text(0.525,   0.975,  'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    else:
        fig.text(0.125,   0.965,   'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
        fig.text(0.525,   0.965,   'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    
    #fig.text(0.15,    0.925, 'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    #fig.text(0.55,    0.925, 'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    #fig.text(0.15,    0.575, 'c', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    #fig.text(0.55,    0.575, 'd', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    
    # plotting the distribution
    weights    = np.ones(len(inferred)) / len(inferred) 
    weights2   = np.ones(len(inferred_new)) / len(inferred_new) 
    x_min      = np.amin(inferred_new) - 0.025
    x_max      = np.amax(inferred_new) + 0.025
    tick_space = (x_max - x_min) / 6
    neg_ticks  = [-i for i in np.arange(0, -x_min, tick_space)]
    pos_ticks  = [i for i in np.arange(0, x_max, tick_space)]
    tick_locs  = neg_ticks[::-1] + pos_ticks
        
    ax2 = fig.add_subplot(grid[0, 0])
    ax2.hist(inferred_new, bins=50, range=[x_min+0.025, x_max-0.025], log=True, color=MAIN_COLOR) 
    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels([f'{int(round(i))}'.replace('-', '\N{MINUS SIGN}') for i in (100 * np.array(tick_locs))])
    ax2.set_ylim(0.75, 10**(digits-0.5))
    
    axes = [ax2]
    ax2.set_ylabel('Counts (Log)', fontsize=AXES_FONTSIZE)
    ax2.set_xlabel("Inferred selection coefficients (%)", fontsize=AXES_FONTSIZE)
    ax2.tick_params(width=SPINE_LW)
    for axis in axes:
        axis.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
        for line in ['right', 'top']:
            axis.spines[line].set_visible(False)
        for line in ['left', 'bottom']:
            axis.spines[line].set_linewidth(SPINE_LW)

    # Finding enrichment of nonsynonymous sites
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
        nonsyn_nonlinked[i] = number_nonsynonymous_nonlinked(inferred, i+1, linked_anywhere, site_types, labels, mutant_types)
    
    # Plotting the enrichment of nonsynonymous mutations vs. number of coefficients
    ax1  = fig.add_subplot(grid[0, 1])
    ax1.set_ylim(syn_lower_limit, 101)
    ax1.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW, length=2)
    for line in ['right',  'top']:
        ax1.spines[line].set_visible(False)
    for line in ['left', 'bottom']:
        ax1.spines[line].set_linewidth(SPINE_LW)
    sns.lineplot(np.arange(len(nonsyn_nonlinked))+1, 100*nonsyn_nonlinked, ax=ax1, lw=1, color=MAIN_COLOR)
    ax1.axhline(100*nonsyn_nonlinked[-1], lw=1, color=COMP_COLOR, alpha=0.75)
    ax1.text(0.25, (100*nonsyn_nonlinked[-1]-syn_lower_limit)/(100-syn_lower_limit) - 0.05, 'Background', fontsize=6, transform=ax1.transAxes, ha='center', va='center')
    ax1.set_xscale('log')
    ax1.set_xlim(1, len(nonsyn_nonlinked)+1)
    ax1.set_ylabel('Nonsynonymous (%)', fontsize=AXES_FONTSIZE)
    ax1.set_xlabel("Number of largest coefficients", fontsize=AXES_FONTSIZE)
    print('background:', 100*nonsyn_nonlinked[-1])
    
    # Plotting the null distribution
    # The null distribution
    """
    ax3 = fig.add_subplot(grid[0, 2])
    ax3.tick_params(labelsize=6, width=SPINE_LW, length=2)
    ax3.hist(inferred_null, bins=50, color=MAIN_COLOR, log=True)
    for line in ['top', 'right']:
        ax3.spines[line].set_visible(False)
    ax3.set_xticks([1, 100, 10000])
    ax3.text(0.15, 0.8, 'e', fontweight='bold', transform=ax2.transAxes, fontsize=8)
    """
    
    if correlation_plots:
        # making the scatterplots comparing nonsynonymous enrichment, average inferred selection, and mutation density in each protein
        ax_syn_dens = fig.add_subplot(lower_grid[0, :])
        ax_syn_sel  = fig.add_subplot(lower_grid[1, :])
        ax_sel_dens = fig.add_subplot(lower_grid[2, :])
        axes = [ax_syn_dens, ax_syn_sel, ax_sel_dens]
        syn_enrichment_scatter(file, syn_file, index_file, ax_syn_dens)
        nonsyn_selection_plot(file, syn_file, index_file, ax_syn_sel)
        plot_average_s_protein(file, index_file, ax_sel_dens)
        for axis in axes:
            axis.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
            
    
    plt.gcf().subplots_adjust(bottom=0.3, left=0.1, right=0.9)
    fig.savefig(os.path.join(fig_path, out_file + '.png'), dpi=1200)
    
"""    
def dist_syn_plots_alt(file, link_file, link_file_all, syn_file, detailed=False, syn_lower_limit=60):
    
    # loading and processing the data
    data            = np.load(file, allow_pickle=True)
    linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in EVERY region
    linked_anywhere = np.load(link_file_all, allow_pickle=True)    # sites that are fully linked in ANY region
    syn_data        = np.load(syn_file, allow_pickle=True)
    allele_number   = data['allele_number']
    labels          = [get_label(i) for i in allele_number]
    types_temp      = syn_data['types']
    type_locs       = syn_data['locations']
    linked_all      = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    inferred        = data['selection']
    labels          = [get_label(i) for i in data['allele_number']]
    inferred_link   = np.zeros(len(linked_sites))
    digits          = len(str(len(inferred)))
    
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
                inferred_link[i] += inferred[np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
    for i in range(len(labels)):
        if labels[i] not in linked_all:
            inferred_new.append(inferred[i])
    inferred_new = inferred_new + list(inferred_link)
    
    # creating figure and subfigures
    fig  = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH / 1.618])
    grid = matplotlib.gridspec.GridSpec(1, 2, hspace=0.05, figure=fig)
    fig.text(0.1,     0.95, 'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.525,   0.95, 'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    
    #fig.text(0.15,    0.925, 'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    #fig.text(0.55,    0.925, 'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    #fig.text(0.15,    0.575, 'c', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    #fig.text(0.55,    0.575, 'd', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    
    # plotting the distribution
    
    weights    = np.ones(len(inferred)) / len(inferred) 
    weights2   = np.ones(len(inferred_new)) / len(inferred_new) 
    x_min      = np.amin(inferred_new) - 0.025
    x_max      = np.amax(inferred_new) + 0.025
    tick_space = (x_max - x_min) / 6
    neg_ticks  = [-i for i in np.arange(0, -x_min, tick_space)]
    pos_ticks  = [i for i in np.arange(0, x_max, tick_space)]
    tick_locs  = neg_ticks[::-1] + pos_ticks
        
    ax2 = fig.add_subplot(grid[0, 0])
    #xticks = []
    #ax2.grid(b=True, axis='x', linewidth=0.5)
    #ax2.hist(inferred_new, bins=60, range=[x_min, x_max], weights=weights2, log=True) 
    ax2.hist(inferred_new, bins=50, range=[x_min+0.025, x_max-0.025], log=True, color=MAIN_COLOR) 
    ax2.set_xticks(tick_locs)
    #ax2.set_xticks(np.linspace(x_min, x_max, 5))
    #ax2.set_xticklabels([f'{i}'.replace('-', '\N{MINUS SIGN}') for i in np.linspace(int(x_min*100), int(x_max*100), 5, dtype=int)])
    ax2.set_xticklabels([f'{int(round(i))}'.replace('-', '\N{MINUS SIGN}') for i in (100 * np.array(tick_locs))])
    #ax2.set_ylim(0.75, 1000)
    ax2.set_ylim(0.75, 10**(digits-0.5))
    #ax2.set_xlim(tick_locs[0] - 0.01, tick_locs[-1] + 0.01)
    ax2.set_ylabel('Counts (Log)', fontsize=8)
    ax2.set_xlabel("Inferred selection\ncoefficients (%)", fontsize=8)
    ax2.tick_params(labelsize=6)
    for line in ['right', 'top']:
        ax2.spines[line].set_visible(False)
            
    #y_ticks = [str(i) for i in ax2.get_yticks()]
    #y_ticklabels = ['$\\mathdefault{10^{' + f'{float(i)}' + '}}$' for i in y_ticks]
    #print(y_ticklabels)
    #ax2.set_yticklabels(ax2.get_yticklabels()[:-2] + [matplotlib.text.Text(0, 1000.0, '')])
    ax2.set_yticklabels(ax2.get_yticklabels()[:-2] + [matplotlib.text.Text(0, float(10**digits), '')])
    
    plt.draw()

    # Finding enrichment of nonsynonymous sites
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
        nonsyn_nonlinked[i] = number_nonsynonymous_nonlinked(inferred, i+1, linked_anywhere, site_types, labels, mutant_types)
    
    # Plotting the enrichment of nonsynonymous mutations vs. number of coefficients
    
    ax1  = fig.add_subplot(grid[0, 1])
    #ax.grid(b=True, axis='y', linewidth=0.2)
    ax1.set_ylim(syn_lower_limit, 100)
    #plt.yticks(np.arange(11)/10)
    #ax1.set_title("Non-synonymous enrichment", fontsize='8')
    ax1.tick_params(labelsize=6, width=0.3, length=2)
    for line in ['right',  'top']:
        ax1.spines[line].set_visible(False)
    #ax.set_xlabel("Number of largest\ncoefficients", fontsize='6')
    #ax1.set_ylabel("Nonsynonymous (%)", fontsize='8')
    sns.lineplot(np.arange(len(nonsyn_nonlinked))+1, 100*nonsyn_nonlinked, ax=ax1, lw=1, color=MAIN_COLOR)
    ax1.axhline(100*nonsyn_nonlinked[-1], lw=1, color=COMP_COLOR, alpha=0.75)
    #ax1.text(0.25, (100*nonsyn_nonlinked-syn_lower_limit)/(100-syn_lower_limit) - 0.05, 'background', color=COMP_COLOR, fontsize=6, transform=ax1.transAxes, ha='center', va='center')
    ax1.text(0.25, (100*nonsyn_nonlinked[-1]-syn_lower_limit)/(100-syn_lower_limit) - 0.05, 'background', color=COMP_COLOR, fontsize=6, transform=ax1.transAxes, ha='center', va='center')
    #plt.gcf().subplots_adjust(bottom=0.3)
    #plt.gcf().subplots_adjust(left=0.2)
    #plt.savefig(os.path.join(fig_path, 'nonsynonymous.png'), dpi=1200)
    ax1.set_ylabel('Nonsynonymous (%)', fontsize=8)
    ax1.set_xlabel("Number of largest\ncoefficients", fontsize='8')
    plt.gcf().subplots_adjust(bottom=0.3, left=0.1, right=0.9)
    #plt.gcf().subplots_adjust(left=0.2)
    #plt.gcf().subplots_adjust(right=0.8)
    fig.savefig(os.path.join(fig_path, 'syn-dist-plot.png'), dpi=1200)
"""    

def selection_dist_plot(file, link_file):
    """ Given a file containing the inferred coefficients, plots 2 histograms comparing 
    1. the distribution of selection coefficients
    2. the distribution of selection coefficients when coefficients for linked sites are summed"""
    
    # loading and processing the data
    data          = np.load(file, allow_pickle=True)
    linked_sites  = np.load(link_file, allow_pickle=True)
    linked_all    = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    inferred      = data['selection']
    labels        = [get_label(i) for i in data['allele_number']]
    inferred_link = np.zeros(len(linked_sites))
    digits        = len(str(len(inferred)))
    inferred_new  = []    # the sum of the inferred coefficients for the linked sites
    for i in range(len(linked_sites)):
        for j in range(len(linked_sites[i])):
            if np.any(linked_sites[i][j]==np.array(labels)):
                inferred_link[i] += inferred[np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
    for i in range(len(labels)):
        if labels[i] not in linked_all:
            inferred_new.append(inferred[i])
    inferred_new = inferred_new + list(inferred_link)
    
    # plotting the data
    weights    = np.ones(len(inferred)) / len(inferred) 
    weights2   = np.ones(len(inferred_new)) / len(inferred_new) 
    x_min      = np.amin(inferred_new) - 0.025
    x_max      = np.amax(inferred_new) + 0.025
    tick_space = (x_max - x_min) / 6
    neg_ticks  = [-i for i in np.arange(0, -x_min, tick_space)]
    pos_ticks  = [i for i in np.arange(0, x_max, tick_space)]
    tick_locs  = neg_ticks[::-1] + pos_ticks
    fig  = plt.figure(figsize=[SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 1.618])
    grid = matplotlib.gridspec.GridSpec(2, 1, hspace=0.02, figure=fig)
    ax   = fig.add_subplot(grid[:, :])
    for line in ['right', 'left', 'top', 'bottom']:
        ax.spines[line].set_visible(False)
    ax.set_ylabel('Counts (Log)', fontsize=AXES_FONTSIZE, labelpad=20)
    ax.set_xlabel("Inferred selection\ncoefficients (%)", fontsize=AXES_FONTSIZE, labelpad=15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax1 = fig.add_subplot(grid[0, 0])
    #ax1.grid(b=True, axis='x', linewidth=0.5)
    #ax1.set_ylim(0.75, 1000)
    ax1.set_ylim(0.75, 10**digits)
    #ax1.set_xlim(tick_locs[0] - 0.01, tick_locs[-1] + 0.01)
    #ax1.hist(inferred, bins=60, range=[x_min, x_max], weights=weights, log=True)  
    ax1.hist(inferred, bins=50, range=[x_min+0.025, x_max-0.025], log=True, color='cornflowerblue')  
    ax1.text(0.75, 0.75, "Linked sites\nseparate", fontsize=8, transform=ax1.transAxes, ha='center', va='center')
    ax2 = fig.add_subplot(grid[1, 0])
    ax1.set_xticks(tick_locs)
    ax1.set_xticklabels('' for i in range(len(tick_locs)))
    #xticks = []
    #ax2.grid(b=True, axis='x', linewidth=0.5)
    #ax2.hist(inferred_new, bins=60, range=[x_min, x_max], weights=weights2, log=True) 
    ax2.hist(inferred_new, bins=50, range=[x_min+0.025, x_max-0.025], log=True, color='cornflowerblue') 
    ax2.set_xticks(tick_locs)
    #ax2.set_xticks(np.linspace(x_min, x_max, 5))
    #ax2.set_xticklabels([f'{i}'.replace('-', '\N{MINUS SIGN}') for i in np.linspace(int(x_min*100), int(x_max*100), 5, dtype=int)])
    ax2.set_xticklabels([f'{int(round(i))}'.replace('-', '\N{MINUS SIGN}') for i in (100 * np.array(tick_locs))])
    #ax2.set_ylim(0.75, 1000)
    ax2.set_ylim(0.75, 10**digits)
    #ax2.set_xlim(tick_locs[0] - 0.01, tick_locs[-1] + 0.01)
    ax2.text(0.75,0.75, "Linked sites\ncombined", fontsize=8, transform=ax2.transAxes, ha='center', va='center')
    for axis in [ax1, ax2]:
        axis.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
        for line in ['right', 'top']:
            axis.spines[line].set_visible(False)
    #y_ticks = [str(i) for i in ax2.get_yticks()]
    #y_ticklabels = ['$\\mathdefault{10^{' + f'{float(i)}' + '}}$' for i in y_ticks]
    #print(y_ticklabels)
    #ax2.set_yticklabels(ax2.get_yticklabels()[:-2] + [matplotlib.text.Text(0, 1000.0, '')])
    ax2.set_yticklabels(ax2.get_yticklabels()[:-2] + [matplotlib.text.Text(0, float(10**digits), '')])
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.20)
    plt.savefig(os.path.join(fig_path, 'inferred_distribution.png'), dpi=1200)
    
    
def selection_dist_protein_plot(file, protein='S', out_file='selection-dist-protein'):
    """ Given a file containing the inferred coefficients, plots histogram of the distribution of selection coefficients"""
    
    # loading and processing the data
    data          = np.load(file, allow_pickle=True)
    inferred      = data['selection']
    labels        = [get_label(i) for i in data['allele_number']]
    protein_labs  = [i for i in labels if i[:len(protein)]==protein]
    #print(protein)
    #print(protein_labs)
    protein_inf   = [inferred[i] for i in range(len(inferred)) if labels[i][:len(protein)]==protein]
    
    # plotting the data
    #weights    = np.ones(len(inferred)) / len(inferred) 
    #weights2   = np.ones(len(inferred_new)) / len(inferred_new) 
    x_min      = np.amin(inferred) - 0.025
    x_max      = np.amax(inferred) + 0.025
    tick_space = (x_max - x_min) / 5
    neg_ticks  = [-i for i in np.arange(0, -x_min, tick_space)]
    pos_ticks  = [i for i in np.arange(0, x_max, tick_space)]
    tick_locs  = neg_ticks[::-1] + pos_ticks[1:]
    xtick_labs = ['%.1f' % i for i in np.array(tick_locs)*100]
    fig  = plt.figure(figsize=[SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH])
    grid = matplotlib.gridspec.GridSpec(1, 1, figure=fig)
    ax   = fig.add_subplot(grid[:, :])
    for line in ['right', 'top']:
        ax.spines[line].set_visible(False)
    ax.set_title(protein, fontsize=AXES_FONTSIZE)
    ax.set_ylabel('Counts (Log)', fontsize=AXES_FONTSIZE, labelpad=20)
    #ax.set_ylim(bottom=0.75, top=None)
    ax.set_xlabel("Inferred selection\ncoefficients (%)", fontsize=AXES_FONTSIZE, labelpad=15)
    ax.hist(protein_inf, bins=50, range=[x_min+0.025, x_max-0.025], log=True, color=MAIN_COLOR)  
    #ax1.text(0.75, 0.75, "Linked sites\nseparate", fontsize=8, transform=ax1.transAxes, ha='center', va='center')
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(xtick_labs)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.20)
    plt.savefig(os.path.join(fig_path, out_file+'.png'), dpi=1200)
    

def find_null_distribution(file, link_file, neutral_tol=0.01):
    """ Given the inferred coefficients over all time, find those that are ultimately inferred to be nearly neutral,
    and then collect the inferred coefficients for these sites at every time."""
    
    # loading and processing the data
    data            = np.load(file, allow_pickle=True)
    linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in every region
    allele_number   = data['allele_number']
    labels          = [get_label(i) for i in allele_number]
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


def find_null_distribution_alt(file, link_file, neutral_tol=0.01):
    """ Given the inferred coefficients over all time, find those that are ultimately inferred to be nearly neutral,
    and then collect the inferred coefficients for these sites at every time."""
    
    # loading and processing the data
    data            = np.load(file,      allow_pickle=True)
    linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in every region
    traj            = data['traj']
    mutant_sites    = data['mutant_sites']
    times           = data['times']
    allele_number   = data['allele_number']
    labels          = [get_label2(i) for i in allele_number]
    linked_all      = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    inferred        = data['selection']
    #times           = data['times']
    #all_times       = data['all_times']
    #traj            = data['traj']
    digits          = len(str(len(inferred)*len(inferred[0])))
    
    # finding first time a mutation was observed
    t_observed     = np.zeros(len(allele_number))
    alleles_sorted = np.sort(allele_number)
    pos_all        = [np.searchsorted(alleles_sorted, mutant_sites[i]) for i in range(len(mutant_sites))]
    for i in range(len(traj)):
        positions   = pos_all[i]
        first_times = []
        for j in range(len(traj[i][0])):
            nonzero = np.nonzero(traj[i][:, j])[0]
            if len(nonzero) > 0: first_times.append(times[i][nonzero[0]])
            else:                first_times.append(9999)
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

    # If a site is inferred to have a selection coefficient of exactly zero, this is because it is the reference nucleotide.
    inferred_neut = np.array(inferred_neut)[np.array(inferred_neut)!=0]
    
    return inferred_neut


def find_null_distribution_ind_old(file, link_file, inf_dir, traj_file=None, neutral_tol=0.01, old=False):
    """ Given the inferred coefficients over all time, find those that are ultimately inferred to be nearly neutral,
    and then make a list of the inferred coefficients locally for those sites over all time and."""
    
    # loading and processing the data
    data            = np.load(file,      allow_pickle=True)
    linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in every region
    traj_full       = data['traj']
    mutant_sites    = data['mutant_sites']
    times           = data['times']
    allele_number   = data['allele_number']
    linked_all      = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    inferred        = data['selection']
    locations       = data['locations']
    if old:
        labels      = [get_label(i) for i in allele_number]
        mutant_sites= [[get_label(mutant_sites[i][j]) for j in range(len(mutant_sites[i]))] for i in range(len(mutant_sites))]
        times_full  = data['times']
    else:
        labels      = [get_label2(i) for i in allele_number]
        mutant_sites= [[get_label2(mutant_sites[i][j]) for j in range(len(mutant_sites[i]))] for i in range(len(mutant_sites))]
        times_full  = data['times_full']
    inferred = inferred[-1]
    
    
    """
    # finding first time a mutation was observed
    t_observed     = np.zeros(len(allele_number))
    alleles_sorted = np.sort(allele_number)
    pos_all        = [np.searchsorted(alleles_sorted, mutant_sites[i]) for i in range(len(mutant_sites))]
    for i in range(len(traj)):
        positions   = pos_all[i]
        first_times = []
        for j in range(len(traj[i][0])):
            nonzero = np.nonzero(traj[i][:, j])[0]
            if len(nonzero) > 0: first_times.append(times[i][nonzero[0]])
            else:                first_times.append(9999)
        for j in range(len(mutant_sites[i])):
            t_observed[positions[j]] = min(t_observed[positions[j]], first_times[j])
            
    t_init_link   = np.zeros(len(linked_sites))
    for i in range(len(labels)):
        if labels[i] in linked_all:
            for j in range(len(linked_sites)):
                if labels[i] in list(linked_sites[j]):
                    t_init_link[j] = min(t_init_link[j], t_observed[i])
    """
        
    # finding total coefficients for linked sites and adding them together.
    #inferred_link   = np.zeros((len(inferred), len(linked_sites)))
    inferred_link   = np.zeros(len(linked_sites))
    inferred_nolink = []
    labels_nolink   = []
    linked_idxs     = []
    for i in range(len(linked_sites)):
        for j in range(len(linked_sites[i])):
            if np.any(linked_sites[i][j]==np.array(labels)):
                #inferred_link[:,i] += inferred[:, np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
                #inferred_link[i] += inferred[np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
                inferred_link[i] += inferred[labels.index(linked_sites[i][j])]
                if i not in linked_idxs:
                    linked_idxs.append(i)
    for i in range(len(labels)):
        if labels[i] not in linked_all:
            #inferred_nolink.append(inferred[:,i])
            inferred_nolink.append(inferred[i])
            labels_nolink.append(labels[i])
    #nolink_new = np.zeros((len(inferred_nolink[0]), len(inferred_nolink)))
    nolink_new = np.zeros(len(inferred_nolink))
    for i in range(len(inferred_nolink)):
        #nolink_new[:,i] = inferred_nolink[i]
        nolink_new[i] = inferred_nolink[i]
    inferred_nolink = nolink_new
    #print(inferred_link)
    
    #for i in range(len(linked_sites)):
    #    print(inferred_link[i], linked_sites[i])
    #print('')
    
    # determing sites that are ultimately inferred to be nearly neutral
    #link_neut    = linked_sites[np.array(inferred_link[-1]) < neutral_tol]
    #nolink_neut  = np.array(labels_nolink)[np.array(inferred_nolink[-1]) < neutral_tol]
    link_neut    = linked_sites[np.absolute(inferred_link) < neutral_tol]
    nolink_neut  = np.array(labels_nolink)[np.array(inferred_nolink) < neutral_tol]
    link_all     = [link_neut[i][j] for i in range(len(link_neut)) for j in range(len(link_neut[i]))]
    
    """
    inf_link_temp = np.array(inferred_link)[np.array(inferred_link) < neutral_tol]
    for i in range(len(link_neut)):
        print(inf_link_temp[i], link_neut[i])
    print('')
    """
    
    # Finding inferred coefficients at every time in the regional inference data
    inferred_neut = []
    for file in os.listdir(inf_dir):
        if os.path.isdir(os.path.join(inf_dir, file)):
            continue
        data_temp = np.load(os.path.join(inf_dir, file), allow_pickle=True)
        s         = data_temp['selection']
        #print(np.shape(s))
        locs_temp = data_temp['locations']
        #traj      = data_temp['traj']
        #print([i for i in data_temp])
        times     = data_temp['times']
        if old:
            all_times = data_temp['times_all']
            muts  = [get_label(i) for i in data_temp['allele_number']]
        else:
            all_times = data_temp['time_infer']
            muts      = [get_label2(i) for i in data_temp['allele_number']]
            muts_loc  = data_temp['mutant_sites']
            muts_loc  = [[get_label2(muts_loc[i][j]) for j in range(len(muts_loc[i]))] for i in range(len(muts_loc))]
            times_loc = data['times_full']
       
        # find linked coefficients
        mask     = [np.any(np.isin(i, muts)) for i in link_neut]
        link_reg = link_neut[mask]
        #print(link_reg)
        linked_s = np.zeros((len(s), len(link_reg)))
        for i in range(len(link_reg)):
            for j in range(len(link_reg[i])):
                if link_reg[i][j] in muts:
                    linked_s[:, i] += s[:, muts.index(link_reg[i][j])]
            
            if np.amax(np.absolute(linked_s[:, i])) > 0.05:
                if np.abs(np.amax(linked_s[:, i])) > np.abs(np.amin(linked_s[:, i])):
                    print(file, np.amax(linked_s[:, i]), link_reg[i])
                else:
                    print(file, np.amin(linked_s[:, i]), link_reg[i])
        
        # adding all linked coefficients to inferred_neut
        #min_time  = np.amin([np.amin(i) for i in times_tot])
        #max_time  = np.amax([np.amax(i) for i in times_tot])
        #all_times = np.arange(min_time, max_time + 1)
        assert(len(all_times)==len(s))
        for i in range(len(link_reg)):
            for j in range(len(linked_s)):
                inferred_neut.append(linked_s[j, i])
                
        # adding nonlinked coefficients
        nolink_s      = s[:, np.isin(muts, nolink_neut)]
        if np.amax(np.absolute(nolink_s)) > 0.05:
            print(file, np.amax(nolink_s), np.amin(nolink_s), muts[np.argmin(np.amin(s, axis=0))])
        #t_init_nolink = np.array(t_initial)[np.isin(muts, nolink_neut)]
        #print(np.shape(nolink_s))
        #print(np.shape(t_init_nolink))
        #print(np.shape(times))
        #print(np.shape(t_init_nolink))
        #print(np.shape(times))
        #print(t_initial)
        for i in range(len(nolink_s)):
            for j in range(len(nolink_s[i])):
                inferred_neut.append(nolink_s[i][j])
        
    """
    inf_new       = np.concatenate((inferred_link, np.array(inferred_nolink)), axis=1)
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
    """
    inferred_neut = np.array(inferred_neut)[np.array(inferred_neut)!=0]
    
    return inferred_neut


def find_null_distribution_ind(file, link_file, inf_dir, traj_dir=None, neutral_tol=0.01, old=False, lower_cutoff=False):
    """ Given the inferred coefficients over all time, find those that are ultimately inferred to be nearly neutral,
    and then make a list of the inferred coefficients locally for those sites over all time and."""
    
    # loading and processing the data
    data            = np.load(file,      allow_pickle=True)
    linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in every region
    traj_full       = data['traj']
    mutant_sites    = data['mutant_sites']
    times           = data['times']
    allele_number   = data['allele_number']
    linked_all      = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    inferred        = data['selection']
    locations       = data['locations']
    labels          = [get_label_new(i) for i in allele_number]
    mutant_sites    = [[dp.get_label_new(mutant_sites[i][j]) for j in range(len(mutant_sites[i]))] for i in range(len(mutant_sites))]
    times_full      = data['times_full']
    #print([i for i in data])
    #times           = data['times']
    #all_times       = data['all_times']
    #traj            = data['traj']
    #digits          = len(str(len(inferred)*len(inferred[0])))
    #print(times_full)
    if traj_dir:
        traj_full = []
        traj_locs = []
        for file in os.listdir(traj_dir):
            traj_full.append(pd.read_csv(os.path.join(traj_dir, file)))
            traj_locs.append(file[:-4])
    
    
    """
    # finding first time a mutation was observed
    t_observed     = np.zeros(len(allele_number))
    alleles_sorted = np.sort(allele_number)
    pos_all        = [np.searchsorted(alleles_sorted, mutant_sites[i]) for i in range(len(mutant_sites))]
    for i in range(len(traj)):
        positions   = pos_all[i]
        first_times = []
        for j in range(len(traj[i][0])):
            nonzero = np.nonzero(traj[i][:, j])[0]
            if len(nonzero) > 0: first_times.append(times[i][nonzero[0]])
            else:                first_times.append(9999)
        for j in range(len(mutant_sites[i])):
            t_observed[positions[j]] = min(t_observed[positions[j]], first_times[j])
            
    t_init_link   = np.zeros(len(linked_sites))
    for i in range(len(labels)):
        if labels[i] in linked_all:
            for j in range(len(linked_sites)):
                if labels[i] in list(linked_sites[j]):
                    t_init_link[j] = min(t_init_link[j], t_observed[i])
    """
        
    # finding total coefficients for linked sites and adding them together.
    #inferred_link   = np.zeros((len(inferred), len(linked_sites)))
    inferred_link   = np.zeros(len(linked_sites))
    inferred_nolink = []
    labels_nolink   = []
    linked_idxs     = []
    for i in range(len(linked_sites)):
        for j in range(len(linked_sites[i])):
            if np.any(linked_sites[i][j]==np.array(labels)):
                #inferred_link[:,i] += inferred[:, np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
                #inferred_link[i] += inferred[np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
                inferred_link[i] += inferred[labels.index(linked_sites[i][j])]
                if i not in linked_idxs:
                    linked_idxs.append(i)
    for i in range(len(labels)):
        if labels[i] not in linked_all:
            #inferred_nolink.append(inferred[:,i])
            inferred_nolink.append(inferred[i])
            labels_nolink.append(labels[i])
    #nolink_new = np.zeros((len(inferred_nolink[0]), len(inferred_nolink)))
    nolink_new = np.zeros(len(inferred_nolink))
    for i in range(len(inferred_nolink)):
        #nolink_new[:,i] = inferred_nolink[i]
        nolink_new[i] = inferred_nolink[i]
    inferred_nolink = nolink_new

    #for i in range(len(inferred_link)):
    #    print(linked_sites[i], inferred_link[i])
    
    #for i in range(len(linked_sites)):
    #    print(inferred_link[i], linked_sites[i])
    #print('')
    
    # determing sites that are ultimately inferred to be nearly neutral
    #link_neut    = linked_sites[np.array(inferred_link[-1]) < neutral_tol]
    #nolink_neut  = np.array(labels_nolink)[np.array(inferred_nolink[-1]) < neutral_tol]
    if lower_cutoff:
        link_mask   = np.logical_and(np.absolute(inferred_link) < neutral_tol, np.absolute(inferred_link) > lower_cutoff)
        nolink_mask = np.logical_and(np.absolute(inferred_nolink) < neutral_tol, np.absolute(inferred_nolink) > lower_cutoff)
    else:
        link_mask   = np.absolute(inferred_link) < neutral_tol
        nolink_mask = np.absolute(inferred_nolink) < neutral_tol
    #link_neut    = linked_sites[np.absolute(inferred_link) < neutral_tol]
    #nolink_neut  = np.array(labels_nolink)[np.absolute(inferred_nolink) < neutral_tol]
    link_neut    = linked_sites[link_mask]
    nolink_neut  = np.array(labels_nolink)[nolink_mask]
    link_all     = [link_neut[i][j] for i in range(len(link_neut)) for j in range(len(link_neut[i]))]
    
    """
    inf_link_temp = np.array(inferred_link)[np.array(inferred_link) < neutral_tol]
    for i in range(len(link_neut)):
        print(inf_link_temp[i], link_neut[i])
    print('')
    """
    
    # Finding inferred coefficients at every time in the regional inference data
    inferred_neut = []
    for file in os.listdir(inf_dir):
        if os.path.isdir(os.path.join(inf_dir, file)):
            continue
        data_temp = np.load(os.path.join(inf_dir, file), allow_pickle=True)
        s         = data_temp['selection']
        #print(np.shape(s))
        locs_temp = data_temp['locations']
        #traj      = data_temp['traj']
        #print([i for i in data_temp])
        times     = data_temp['times']
        if 'time_infer' in data_temp:
            all_times = data_temp['time_infer']
        else:
            print(f'time_infer not in file {file}')
            all_times = data_temp['times_inf']
        muts      = [dp.get_label_new(i) for i in data_temp['allele_number']]
        
        traj_file_idxs = [traj_locs.index(i) for i in locs_temp]
        traj_temp      = np.array(traj_full)[np.array(traj_file_idxs)]
        muts_temp      = [list(i.columns.values)[1:] for i in traj_temp]
        times_tot      = [list(i['time']) for i in traj_temp]
        traj           = [i.drop(columns=['time']).to_numpy() for i in traj_temp]
        #times_tot = [times_full[list(locations).index(i)]   for i in locs_temp]
        #traj      = [traj_full[list(locations).index(i)]    for i in locs_temp]
        #muts_temp = [mutant_sites[list(locations).index(i)] for i in locs_temp]
        #times_tot = data_temp['times_all']
        
        # finding the first times that mutations arise
        t_initial = []
        for i in range(len(muts)):
            #nonzero = np.nonzero(traj[:, i])[0]
            nonzero = []
            for j in range(len(muts_temp)):
                if muts[i] in list(muts_temp[j]):
                    nonzero_temp = np.nonzero(traj[j][:, list(muts_temp[j]).index(muts[i])])[0]
                    if len(nonzero_temp) > 0:
                        #print(times_tot[j], nonzero_temp)
                        nonzero.append(times_tot[j][nonzero_temp[0]])
                    #nonzero.append(np.nonzero(traj[j][:, list(muts_temp[j]).index(muts[i])])[0])
            #nonzero = [np.nonzero(traj[j][:, i])[0] for j in range(len(traj))]
            #nonzero = [j[0] for j in nonzero if len(j)>0]
            #if len(nonzero) > 0: t_initial.append(times[np.amin(nonzero)])
            if len(nonzero) > 0: t_initial.append(np.amin(nonzero))
            else:                t_initial.append(9999)
            #if len(nonzero) > 0: t_initial.append(times[nonzero[0]])
            #else:                t_initial.append(9999)
        t_init_link   = np.zeros(len(link_neut))
        for i in range(len(muts)):
            if muts[i] in link_all:
                for j in range(len(link_neut)):
                    if muts[i] in list(link_neut[j]):
                        t_init_link[j] = min(t_init_link[j], t_initial[i])
        
        # find linked coefficients
        mask     = [np.any(np.isin(i, muts)) for i in link_neut]
        link_reg = link_neut[mask]
        linked_s = np.zeros((len(s), len(link_reg)))
        for i in range(len(link_reg)):
            for j in range(len(link_reg[i])):
                if link_reg[i][j] in muts:
                    linked_s[:, i] += s[:, muts.index(link_reg[i][j])]
            
            if np.amax(np.absolute(linked_s[:, i])) > 0.2:
                if np.abs(np.amax(linked_s[:, i])) > np.abs(np.amin(linked_s[:, i])):
                    print(np.amax(linked_s[:, i]), file, link_reg[i])
                else:
                    print(np.amin(linked_s[:, i]), file, link_reg[i])
        
        # adding all linked coefficients to inferred_neut
        #min_time  = np.amin([np.amin(i) for i in times_tot])
        #max_time  = np.amax([np.amax(i) for i in times_tot])
        #all_times = np.arange(min_time, max_time + 1)
        assert(len(all_times)==len(s))
        for i in range(len(link_reg)):
            for j in range(len(linked_s)):
                if t_init_link[i] <= all_times[j]:
                    inferred_neut.append(linked_s[j, i])
                
        # adding nonlinked coefficients
        nolink_s      = s[:, np.isin(muts, nolink_neut)]
        t_init_nolink = np.array(t_initial)[np.isin(muts, nolink_neut)]
        #print(np.shape(nolink_s))
        #print(np.shape(t_init_nolink))
        #print(np.shape(times))
        #print(np.shape(t_init_nolink))
        #print(np.shape(times))
        #print(t_initial)
        for i in range(len(nolink_s)):
            for j in range(len(nolink_s[i])):
                if t_init_nolink[j] <= all_times[i]:
                    inferred_neut.append(nolink_s[i][j])
        
    """
    inf_new       = np.concatenate((inferred_link, np.array(inferred_nolink)), axis=1)
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
    """
    print('')

    # If a site is inferred to have a selection coefficient of exactly zero, this is because it is the reference nucleotide.
    inferred_neut = np.array(inferred_neut)[np.array(inferred_neut)!=0]
    
    return inferred_neut


def b117_plot(inf_file, tv_file, link_file, full_tv_file=None, start=355, end_time=370, 
              out_file='b117-detection', nosmooth=False, window_size=10, matt_fig=False, group='b.1.1.7'):
    """ Plots the trajectory and the inferred coefficient over time for the B.1.1.7 group. Also compares to the null distribution."""
    if   group=='b.1.1.7' or group=='alpha':
        #site = 'S-570-1-A'
        site = 'S-570-1'
    elif group=='delta':
        site = 'S-80-1-C'
    inf_data  = np.load(inf_file, allow_pickle=True)
    inferred  = inf_data['selection']
    alleles   = inf_data['allele_number']
    labels    = [get_label(i) for i in alleles]
    traj      = inf_data['traj']
    muts      = inf_data['mutant_sites']
    times     = inf_data['times']
    locs      = inf_data['locations']
    tv_data   = np.load(tv_file, allow_pickle=True)
    print(len(times[0]))
    print(len(tv_data['times_all']))
    print(np.shape(traj[0]))
    print(times[0][0])
    """
    if len(np.shape(inferred))==1:
        inf_new = np.zeros((len(inferred), len(inferred[0])))
        for i in range(len(inferred)):
            inf_new[i] = inferred[]
    """
    if nosmooth:
        traj  = inf_data['traj_nosmooth']
        times = inf_data['times_full']

    # Finding the B.1.1.7 trajectory
    site_traj  = []
    site_times = []
    for i in range(len(traj)):
        muts_temp = [get_label(muts[i][j]) for j in range(len(muts[i]))]
        if site in muts_temp:
            site_traj.append(traj[i][:, muts_temp.index(site)])
            site_times.append(times[i])

    # Finding the inferred coefficient for the linked group over all time
    labels_link, inf_link, err_link = calculate_linked_coefficients(tv_file, link_file, tv=True)
    inf_link = np.swapaxes(inf_link, 0, 1)
    err_link = np.swapaxes(err_link, 0, 1)
    print(len(labels_link), np.shape(inf_link))
    #print(inf_link)
    site_index = []
    for i in range(len(labels_link)):
        labels_temp = [labels_link[i][j] for j in range(len(labels_link[i]))]
        if site in labels_temp:
            site_index.append(i)
    print(site_index)
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
        inferred_null = find_null_distribution(tv_file, link_file)
    else:
        inferred_null = find_null_distribution(full_tv_file, link_file)
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
    if not nosmooth:
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
    #print('frequency at detected time:', traj_detected)
    
    #site_inf = 100*site_inf
    
    ## Find first day where the frequency was greater than 0. 
    #t_arose = uk_time[uk_traj>0.01]
    #print('frequency above 1% at times:', t_arose)
    
    # colors
    palette2 = sns.hls_palette(3)
    color1 = palette2[0]
    color2 = palette2[1]
    color3 = palette2[2]
    
    # create the figure parameters
    top_plots      = 0.4    # percentage of figure taken up by top plots
    left           = 0.2
    l_bottom       = 0.1    # The bottom for the lower plots
    u_bottom       = 1 - top_plots  # the bottom for the upper plots
    l_width        = 0.6    # the width of the lower plots
    u_width        = 0.75    # the width of the upper plots
    u_height       = 0.15   # the height of each upper plot
    l_height       = 0.2   # the height of each lower plot
    spacing        = 0.01
    rect_u_s       = [left, u_bottom, u_width, u_height]    # The box for the upper selection plot 
    rect_u_traj    = [left, u_bottom + u_height + spacing, u_width, u_height]    # the box for the upper frequency plot
    rect_l_s       = [left, l_bottom, l_width, l_height]    # the box for the lower selection plot
    rect_l_traj    = [left, l_bottom + l_height + spacing, l_width, l_height]    # the box for the lower frequency plot
    rect_hist      = [left + l_width, l_bottom, 0.2, l_height]    # the box for the histogram plot
    
    ymin1, ymax1 = 0, 0.2    # the y-limits for the lower selection plot
    ymin3, ymax3 = 0, 0.6    # the y-limits for the lower frequency plot

    # The figure
    fig = plt.figure(figsize=[SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 1.618])
    
    ### the lower plots, focusing on the region in the time series that is especially of interest
    # The inferred coefficient over time
    ymax1 = site_inf[-1] * 1.1
    ax1 = fig.add_axes(rect_l_s)
    ax1.set_ylabel('Selection coefficient', color='k', fontsize=AXES_FONTSIZE)
    ax1.set_xlabel('Date', fontsize=AXES_FONTSIZE)
    ax1.set_ylim(ymin1, ymax1)
    ax1.set_xlim(start_time, times_all[-1])
    ax1.axhline(y=inf_detected, xmin=((t_detected-start_time)/(times_all[-1]-start_time)), xmax=1, color='k', linestyle='dashed')
    #ax1.axvline(x=t_detected, ymin=0, ymax=1, color='k', linestyle='dashed')
    #ax1.text(0.05+((t_detected-start_time)/(times_all[-1]-start_time)), 0.025, f"{t_detected}", fontsize=8, transform=ax1.transAxes, ha='center', va='center', color='k')
    #ax1.text(0.075+((t_detected-start_time)/(times_all[-1]-start_time)), ((inf_detected - ymin1)/(ymax1 - ymin1))+0.05, '%.3f' % inf_detected, fontsize=8,
    #         transform=ax1.transAxes, ha='center', va='center', color='k')
    ax1.text(0.9, ((inf_detected - ymin1)/(ymax1 - ymin1))+0.05, '%.3f' % inf_detected, fontsize=AXES_FONTSIZE,
             transform=ax1.transAxes, ha='center', va='center', color='k')
    ax1.tick_params(axis='y', labelcolor='k', labelsize=TICK_FONTSIZE, width=SPINE_LW)
    ax1.tick_params(axis='x', labelsize=TICK_FONTSIZE, width=SPINE_LW)
    #print(ax1.get_yticks())
    ax1.set_yticks(ax1.get_yticks()[:-2])
    #print(ax1.get_yticks())
    sns.lineplot(np.arange(start_time, start_time + len(site_inf)), site_inf, ax=ax1, color=color1)
    # convert xticks to dates
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in ax1.get_xticks()]
    if matt_fig:
        xticklabels     = [f'{i.day}/{i.month}' for i in xticklabels]
    else:
        xticklabels     = [f'{i.month}/{i.day}' for i in xticklabels]
    voc_detect_date = (dt.date(2021,12,1) - dt.date(2021,1,1)).days + 1
    detect_xcoord   = (voc_detect_date - start_time) 
    ax1.text((detect_xcoord / (end_time - start_time)) + 0.055, 0.075, 'reported', color=color2, transform=ax1.transAxes, ha='left', va='center', fontsize=8)
    ax1.axvline(x=voc_detect_date, ymin=0, ymax=1, color=color2, linestyle='dashed')
    ax1.set_xticklabels(xticklabels)
    ax1.text(-0.3, 1, 'd', fontweight='bold', transform=ax1.transAxes, fontsize=AXES_FONTSIZE)
    for line in ['top', 'right']:
        ax1.spines[line].set_visible(False)
    
    # The null distribution
    ax2 = fig.add_axes(rect_hist, sharey=ax1)
    ax2.tick_params(axis='y', labelleft=False, width=SPINE_LW)
    ax2.text(0.5, 0.5, 'Null\nDistribution', fontsize=8, transform=ax2.transAxes, ha='center', va='center')
    ax2.hist(inferred_null, bins=50, color=color1, orientation='horizontal', log=True)
    for line in ['top', 'right']:
        ax2.spines[line].set_visible(False)
    ax2.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
    ax2.set_xticks([1, 100, 10000])
    ax2.text(0.15, 0.8, 'e', fontweight='bold', transform=ax2.transAxes, fontsize=8)
    
    # The frequency trajectory
    ymax3 = uk_traj[-len(site_inf):][-1] * 1.1
    ax3 = fig.add_axes(rect_l_traj, sharex=ax1)
    ax3.tick_params(axis='x', labelbottom=False, width=SPINE_LW)
    ax3.tick_params(axis='y', labelcolor='k', labelsize=TICK_FONTSIZE, width=SPINE_LW)
    ax3.set_ylabel('Frequency (%)', color='k', fontsize=AXES_FONTSIZE)
    ax3.set_yticklabels(['%.1f' % (100*float(i)) for i in ax3.get_yticks()])
    sns.lineplot(np.arange(start_time, start_time + len(site_inf)), uk_traj[-len(site_inf):], color=color3, ax=ax3)
    #ax3.axvline(x=t_detected, ymin=ymin3, ymax=(traj_detected / ymax3), color='k', linestyle='dashed')
    ax3.axhline(y=traj_detected, xmin=((t_detected-start_time)/(times_all[-1]-start_time)), xmax=1, color='k', linestyle='dashed')
    ax3.text(1.085, traj_detected, '%.3f' % traj_detected, fontsize=8, transform=ax3.transAxes, ha='center', va='center')
    ax3.set_ylim(bottom=ymin3, top=ymax3)
    ax3.set_yticks([0, ymax3/2, ymax3])
    ax3.set_yticklabels([0, int(ymax3*50), int(ymax3*100)])
    ax3.text(-0.3, 0.9, 'c', fontweight='bold', transform=ax3.transAxes, fontsize=8)
    print(uk_traj[-len(site_inf):])
    
    ### the upper plots, plotting the inferred selection coefficient and the frequencies for all time
    # the inferred coefficient over time
    ax4 = fig.add_axes(rect_u_s)
    ax4.set_ylabel('Selection\ncoefficient', color='k', fontsize=AXES_FONTSIZE)
    ax4.set_xlabel('Date', fontsize=AXES_FONTSIZE)
    #ax4.set_ylim(ymin1, ymax1)
    ax4.set_ylim(bottom=0, top=full_inf[-1]+0.05)
    ax4.set_xlim(full_times[0], full_times[-1])
    ax4.tick_params(axis='y', labelcolor='k', labelsize=TICK_FONTSIZE, width=SPINE_LW)
    ax4.tick_params(axis='x', labelsize=TICK_FONTSIZE, width=SPINE_LW)
    sns.lineplot(full_times, full_inf, ax=ax4, color=color1)
    for line in ['top', 'right']:
        ax4.spines[line].set_visible(False)
    # converting the times to dates
    xticklabs = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in ax4.get_xticks()]
    xticklabs = [f'{i.month}/{i.day}' for i in xticklabs]
    ax4.set_xticklabels(xticklabs)
    ax4.text(-0.25, 0.90, 'b', fontweight='bold', transform=ax4.transAxes, fontsize=8)
    
    # the frequency over time
    ymax5 = uk_traj_full[-1]
    ax5 = fig.add_axes(rect_u_traj, sharex=ax4)
    ax5.tick_params(axis='x', labelbottom=False, width=SPINE_LW)
    ax5.tick_params(axis='y', labelcolor='k', labelsize=TICK_FONTSIZE, width=SPINE_LW)
    ax5.set_ylabel('Frequency (%)', color='k', fontsize=AXES_FONTSIZE)
    #ax5.set_ylim(bottom=ymin3, top=ymax3)
    ax5.set_ylim(bottom=0)
    #ax5.set_yticks([0, ymax5/2, ymax5])
    #ax5.set_yticklabels(['%.1f' % (100*float(i)) for i in ax5.get_yticks()])
    ax5.set_yticks([0, 0.5, 1])
    ax5.set_yticklabels(['0', '50', '100'])
    if nosmooth:
        sns.lineplot(full_times[-len(uk_traj_full):], uk_traj_full, color=color3, ax=ax5)
    else:
        sns.lineplot(full_times[-len(uk_traj_full):] - int(window_size/2), uk_traj_full, color=color3, ax=ax5)
    for line in ['top', 'right']:
        ax5.spines[line].set_visible(False)
    ax5.text(-0.25, 1, 'a', fontweight='bold', transform=ax5.transAxes, fontsize=8)
    
    # adding the vertical lines in the upper plots
    start_coords1 = np.array([(start_time - full_times[0]) / (full_times[-1] - full_times[0]), 0])
    start_coords2 = np.array([(start_time - full_times[0]) / (full_times[-1] - full_times[0]), 1])
    end_coords1   = np.array([(end_time -   full_times[0]) / (full_times[-1] - full_times[0]), 0])
    end_coords2   = np.array([(end_time -   full_times[0]) / (full_times[-1] - full_times[0]), 1])
    plt.annotate('', xy=start_coords1 + np.array([0, -0.025]), xytext=start_coords2, xycoords=ax4.transAxes, textcoords=ax5.transAxes, arrowprops={'arrowstyle' : '-'}, clip_on=False)
    plt.annotate('', xy=end_coords1   + np.array([0, -0.025]), xytext=end_coords2,   xycoords=ax4.transAxes, textcoords=ax5.transAxes, arrowprops={'arrowstyle' : '-'}, clip_on=False)
    plt.annotate('', xy=start_coords2 + np.array([-0.01, -0.025]), xytext=end_coords2 + np.array([+0.01, -0.025]), 
                 xycoords=ax5.transAxes, textcoords=ax5.transAxes, arrowprops={'arrowstyle' : '-'})
    
    # adding the lines connected the upper plots to the zoomed in lower plots
    plt.annotate('', xy=start_coords1 + np.array([0.005, 0.02]), xytext=[-0.005, 0.98], 
                 xycoords=ax4.transAxes, textcoords=ax3.transAxes, arrowprops={'arrowstyle' : '-'}, clip_on=False)
    plt.annotate('', xy=end_coords1   + np.array([-0.01, 0.01]), xytext=[1.005,  0.99], 
                 xycoords=ax4.transAxes, textcoords=ax3.transAxes, arrowprops={'arrowstyle' : '-'}, clip_on=False)
    
    # adding a dotted line at the time of detection on both lower plots
    detect_coords1 = [t_detected, 0]
    detect_coords2 = [t_detected, traj_detected]
    kw = dict(linestyle="--", color="k", linewidth=1.5)
    cp = matplotlib.patches.ConnectionPatch(xyA=detect_coords1, xyB=detect_coords2, coordsA="data", coordsB="data", axesA=ax1, axesB=ax3, clip_on=False, **kw)
    ax3.add_artist(cp)
    #plt.annotate('', xy=detect_coords1, xytext=detect_coords2, xycoords=ax1.transAxes, textcoords=ax2.transAxes, arrowprops={'arrowstyle' : })
    
    traj_times = full_times[-len(uk_traj_full):] - int(window_size / 2)
    both_times = [i for i in traj_times if i in full_times]
    mask_s     = np.isin(full_times, both_times)
    mask_traj  = np.isin(traj_times, both_times)
    s_new      = full_inf[mask_s]
    traj_new   = uk_traj_full[mask_traj]
    #times    = inf_data['times']
    #times    = times[:len(uk_traj_full)]
    #full_inf = full_inf[:len(uk_traj_full)]
    
    f = open('/Users/brianlee/SARS-CoV-2-Data/2021-08-14/alpha-s-tv.csv', mode='w')
    f.write('time,frequency,selection\n')
    for i in range(len(both_times)):
        #f.write('%d,%.8f,%.8f\n' % (both_times[i], uk_traj_full[i], full_inf[i]))
        f.write('%d,%.8f,%.8f\n' % (both_times[i], traj_new[i], s_new[i]))
    f.close()
    
    print(len(full_times), len(uk_traj_full), len(full_inf))
    
    fig.savefig(os.path.join(fig_path, out_file + '.png'), dpi=1200)
    
    
def b117_plot_alt(tv_file, variant_file, link_file, full_tv_file=None, start=355, end_time=370, 
              out_file='-detection', nosmooth=False, window_size=10, matt_fig=False, group='b.1.1.7'):
    """ Plots the trajectory and the inferred coefficient over time for the B.1.1.7 group. Also compares to the null distribution."""
    if   group=='b.1.1.7' or group=='alpha':
        site = 'S-570-1-A'
    elif group=='delta':
        site = 'S-19-1-G'
    inf_data  = np.load(tv_file, allow_pickle=True)
    inferred  = inf_data['selection']
    alleles   = inf_data['allele_number']
    labels    = [get_label(i[:-2]) + '-' +  i[-1] for i in alleles]
    traj      = inf_data['traj']
    muts      = inf_data['mutant_sites']
    times     = inf_data['times']
    locs      = inf_data['locations']
    tv_data   = np.load(tv_file, allow_pickle=True)
    """
    if len(np.shape(inferred))==1:
        inf_new = np.zeros((len(inferred), len(inferred[0])))
        for i in range(len(inferred)):
            inf_new[i] = inferred[]
    """
    if nosmooth:
        traj  = inf_data['traj_nosmooth']
        times = inf_data['times_full']

    # Finding the B.1.1.7 trajectory
    site_traj  = []
    site_times = []
    for i in range(len(traj)):
        muts_temp = [dp.get_label2(muts[i][j]) for j in range(len(muts[i]))]
        if site in muts_temp:
            site_traj.append(traj[i][:, muts_temp.index(site)])
            site_times.append(times[i])

    # Finding the inferred coefficient for the linked group over all time
    labels_link, inf_link, err_link = calculate_linked_coefficients(tv_file, variant_file, tv=True)
    #print(inf_link)
    site_index = []
    for i in range(len(labels_link)):
        labels_temp = [labels_link[i][j] for j in range(len(labels_link[i]))]
        if site in labels_temp:
            site_index.append(i)
    if len(site_index)>1:
        print('site appears in multiple places', site_index)
    site_inf = inf_link[:, site_index[0]]
    
    # getting the times corresponding to the inference for the region of interest
    times_all  = inf_data['times_inf']
        
    if start != times_all[0]:
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
    
    # find null distribution
    if not full_tv_file:
        inferred_null = find_null_distribution_alt(tv_file, link_file)
    else:
        inferred_null = find_null_distribution_alt(full_tv_file, link_file)
    max_null    = np.amax(inferred_null)
    min_null    = np.amin(inferred_null)
    num_largest = int(round(len(inferred_null) * 2.5 / 100))    # the number of coefficients in the top 2.5% of the null distribution
    soft_cutoff = np.sort(inferred_null)[::-1][num_largest]
    print('soft cutoff =', soft_cutoff)
    
    ### Find when the inferred coefficient exceeds the null distribution
    locs_sep     = [i.split('-') for i in locs]
    #print(np.array(site_times)[np.array(locs_sep)=='lond'])
    uk_idxs      = np.array(locs_sep)[np.array(locs_sep)=='england_wales_scotland']
    uk_times     = [site_times[i] for i in range(len(site_times))]
    uk_index     = 0
    for i in range(len(uk_times)):
        if start in uk_times[i]:
            uk_index = i
            break
    #uk_index     = np.argmax([np.amax(i) for i in site_traj])
    uk_traj      = site_traj[uk_index]
    uk_time      = list(site_times[uk_index])
    uk_traj      = uk_traj[uk_time.index(start):uk_time.index(end_time)+1]
    uk_time      = uk_time[uk_time.index(start):uk_time.index(end_time)+1]
    
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
    #print('frequency at detected time:', traj_detected)
    
    #site_inf = 100*site_inf
    
    ## Find first day where the frequency was greater than 0. 
    #t_arose = uk_time[uk_traj>0.01]
    #print('frequency above 1% at times:', t_arose)
    
    # colors
    palette2 = sns.hls_palette(3)
    color1 = palette2[0]
    color2 = palette2[1]
    color3 = palette2[2]
    
    # create the figure parameters
    top_plots      = 0.4    # percentage of figure taken up by top plots
    left           = 0.2
    l_bottom       = 0.1    # The bottom for the lower plots
    u_bottom       = 1 - top_plots  # the bottom for the upper plots
    l_width        = 0.6    # the width of the lower plots
    u_width        = 0.75    # the width of the upper plots
    u_height       = 0.15   # the height of each upper plot
    l_height       = 0.2   # the height of each lower plot
    spacing        = 0.01
    rect_u_s       = [left, u_bottom, u_width, u_height]    # The box for the upper selection plot 
    rect_u_traj    = [left, u_bottom + u_height + spacing, u_width, u_height]    # the box for the upper frequency plot
    rect_l_s       = [left, l_bottom, l_width, l_height]    # the box for the lower selection plot
    rect_l_traj    = [left, l_bottom + l_height + spacing, l_width, l_height]    # the box for the lower frequency plot
    rect_hist      = [left + l_width, l_bottom, 0.2, l_height]    # the box for the histogram plot
    
    ymin1, ymax1 = 0, 0.2    # the y-limits for the lower selection plot
    ymin3, ymax3 = 0, 0.6    # the y-limits for the lower frequency plot

    # The figure
    fig = plt.figure(figsize=[SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 1.618])
    
    ### the lower plots, focusing on the region in the time series that is especially of interest
    # The inferred coefficient over time
    ymax1 = site_inf[-1] * 1.1
    ax1 = fig.add_axes(rect_l_s)
    ax1.set_ylabel('Selection coefficient', color='k', fontsize=AXES_FONTSIZE)
    ax1.set_xlabel('Date', fontsize=AXES_FONTSIZE)
    ax1.set_ylim(ymin1, ymax1)
    ax1.set_xlim(start_time, times_all[-1])
    ax1.axhline(y=inf_detected, xmin=((t_detected-start_time)/(times_all[-1]-start_time)), xmax=1, color='k', linestyle='dashed')
    #ax1.axvline(x=t_detected, ymin=0, ymax=1, color='k', linestyle='dashed')
    #ax1.text(0.05+((t_detected-start_time)/(times_all[-1]-start_time)), 0.025, f"{t_detected}", fontsize=8, transform=ax1.transAxes, ha='center', va='center', color='k')
    #ax1.text(0.075+((t_detected-start_time)/(times_all[-1]-start_time)), ((inf_detected - ymin1)/(ymax1 - ymin1))+0.05, '%.3f' % inf_detected, fontsize=8,
    #         transform=ax1.transAxes, ha='center', va='center', color='k')
    ax1.text(0.9, ((inf_detected - ymin1)/(ymax1 - ymin1))+0.05, '%.3f' % inf_detected, fontsize=AXES_FONTSIZE,
             transform=ax1.transAxes, ha='center', va='center', color='k')
    ax1.tick_params(axis='y', labelcolor='k', labelsize=TICK_FONTSIZE, width=SPINE_LW)
    ax1.tick_params(axis='x', labelsize=TICK_FONTSIZE, width=SPINE_LW)
    #print(ax1.get_yticks())
    ax1.set_yticks(ax1.get_yticks()[:-2])
    #print(ax1.get_yticks())
    sns.lineplot(np.arange(start_time, start_time + len(site_inf)), site_inf, ax=ax1, color=color1)
    # convert xticks to dates
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2020,1,1)) for i in ax1.get_xticks()]
    if matt_fig:
        xticklabels     = [f'{i.day}/{i.month}' for i in xticklabels]
    else:
        xticklabels     = [f'{i.month}/{i.day}' for i in xticklabels]
    voc_detect_date = (dt.date(2021,12,1) - dt.date(2021,1,1)).days + 1
    detect_xcoord   = (voc_detect_date - start_time) 
    ax1.text((detect_xcoord / (end_time - start_time)) + 0.055, 0.075, 'reported', color=color2, transform=ax1.transAxes, ha='left', va='center', fontsize=8)
    ax1.axvline(x=voc_detect_date, ymin=0, ymax=1, color=color2, linestyle='dashed')
    ax1.set_xticklabels(xticklabels)
    ax1.text(-0.3, 1, 'd', fontweight='bold', transform=ax1.transAxes, fontsize=AXES_FONTSIZE)
    for line in ['top', 'right']:
        ax1.spines[line].set_visible(False)
    
    # The null distribution
    ax2 = fig.add_axes(rect_hist, sharey=ax1)
    ax2.tick_params(axis='y', labelleft=False, width=SPINE_LW)
    ax2.text(0.5, 0.5, 'Null\nDistribution', fontsize=8, transform=ax2.transAxes, ha='center', va='center')
    ax2.hist(inferred_null, bins=50, color=color1, orientation='horizontal', log=True)
    for line in ['top', 'right']:
        ax2.spines[line].set_visible(False)
    ax2.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
    ax2.set_xticks([1, 100, 10000])
    ax2.text(0.15, 0.8, 'e', fontweight='bold', transform=ax2.transAxes, fontsize=8)
    
    # The frequency trajectory
    ymax3 = uk_traj[-len(site_inf):][-1] * 1.1
    ax3 = fig.add_axes(rect_l_traj, sharex=ax1)
    ax3.tick_params(axis='x', labelbottom=False, width=SPINE_LW)
    ax3.tick_params(axis='y', labelcolor='k', labelsize=TICK_FONTSIZE, width=SPINE_LW)
    ax3.set_ylabel('Frequency (%)', color='k', fontsize=AXES_FONTSIZE)
    ax3.set_yticklabels(['%.1f' % (100*float(i)) for i in ax3.get_yticks()])
    sns.lineplot(np.arange(start_time, start_time + len(site_inf)), uk_traj[-len(site_inf):], color=color3, ax=ax3)
    #ax3.axvline(x=t_detected, ymin=ymin3, ymax=(traj_detected / ymax3), color='k', linestyle='dashed')
    ax3.axhline(y=traj_detected, xmin=((t_detected-start_time)/(times_all[-1]-start_time)), xmax=1, color='k', linestyle='dashed')
    ax3.text(1.085, traj_detected, '%.3f' % traj_detected, fontsize=8, transform=ax3.transAxes, ha='center', va='center')
    ax3.set_ylim(bottom=ymin3, top=ymax3)
    ax3.set_yticks([0, ymax3/2, ymax3])
    ax3.set_yticklabels([0, int(ymax3*50), int(ymax3*100)])
    ax3.text(-0.3, 0.9, 'c', fontweight='bold', transform=ax3.transAxes, fontsize=8)
    
    # adding a dotted line at the time of detection on both lower plots
    """
    detect_coords1 = [t_detected, 0]
    detect_coords2 = [t_detected, traj_detected]
    kw = dict(linestyle="--", color="k", linewidth=1.5)
    cp = matplotlib.patches.ConnectionPatch(xyA=detect_coords1, xyB=detect_coords2, coordsA="data", coordsB="data", axesA=ax1, axesB=ax3, clip_on=False, **kw)
    ax3.add_artist(cp)
    #plt.annotate('', xy=detect_coords1, xytext=detect_coords2, xycoords=ax1.transAxes, textcoords=ax2.transAxes, arrowprops={'arrowstyle' : })
    """
    fig.savefig(os.path.join(fig_path, out_file + '.png'), dpi=1200)
    

def selection_dist_tv_plot(file, link_file, out_file='inferred-distribution-tv', neutral_tol=0.01):
    """ Given a file containing the inferred coefficients, plots 2 histograms comparing 
    1. the distribution of selection coefficients
    2. the distribution of selection coefficients when coefficients for linked sites are summed"""
    
    # loading and processing the data
    data          = np.load(file, allow_pickle=True)
    linked_sites  = np.load(link_file, allow_pickle=True)
    linked_all    = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    inferred      = data['selection']
    labels        = [get_label(i) for i in data['allele_number']]
    inferred_link = []
    digits        = len(str(len(inferred)*len(inferred[0])))
    inferred_nl   = []    # the coefficients for sites not linked
    for i in range(len(linked_sites)):
        for k in range(len(inferred)):
            link_coefficient = 0
            for j in range(len(linked_sites[i])):
                if np.any(linked_sites[i][j]==np.array(labels)):
                    link_coefficient += inferred[k][np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
            if link_coefficient != 0:
                inferred_link.append(link_coefficient)
    for i in range(len(labels)):
        if labels[i] not in linked_all:
            for k in range(len(inferred)):                    
                inferred_nl.append(inferred[k][i])
    inferred_new  = inferred_nl + inferred_link
    inferred_neut = find_null_distribution(file, link_file, neutral_tol=neutral_tol)
    print('largest  null coefficient:', np.amax(inferred_neut))
    print('smallest null coefficient:', np.amin(inferred_neut))
    
    # plotting the data
    weights    = np.ones(len(inferred)) / len(inferred) 
    weights2   = np.ones(len(inferred_new)) / len(inferred_new) 
    x_min      = np.amin(inferred_new) - 0.025
    x_max      = np.amax(inferred_new) + 0.025
    tick_space = (x_max - x_min) / 6
    neg_ticks  = [-i for i in np.arange(0, -x_min, tick_space)]
    pos_ticks  = [i for i in np.arange(0, x_max, tick_space)]
    tick_locs  = neg_ticks[::-1] + pos_ticks
    fig  = plt.figure(figsize=[SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 1.618])
    grid = matplotlib.gridspec.GridSpec(2, 1, hspace=0.02, figure=fig)
    ax2 = fig.add_subplot(grid[0,0])
    ax2.set_ylabel('Counts (Log)', fontsize=AXES_FONTSIZE, labelpad=20)
    #ax2.set_xlabel("Inferred selection\ncoefficients (%)", fontsize=8, labelpad=15)
    ax2.hist(inferred_new, bins=100, range=[x_min-0.025, x_max+0.025], log=True, color='cornflowerblue') 
    #ax2.set_xticks(tick_locs)
    #ax2.set_xticklabels([f'{int(round(i))}'.replace('-', '\N{MINUS SIGN}') for i in (100 * np.array(tick_locs))])
    ax2.set_ylim(0.75, 10**digits)
    ax2.text(0.75,0.75, "Linked sites\ncombined", fontsize=AXES_FONTSIZE, transform=ax2.transAxes, ha='center', va='center')
    ax2.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
    for line in ['right', 'top']:
        ax2.spines[line].set_visible(False)
        
    # plot the null distribution
    ax1 = fig.add_subplot(grid[1,0])
    ax1.set_xlabel("Inferred selection\ncoefficients (%)", fontsize=AXES_FONTSIZE, labelpad=15)
    ax1.set_ylabel('Counts (Log)', fontsize=AXES_FONTSIZE, labelpad=20)
    ax1.hist(inferred_neut, bins=100, range=[x_min-0.025, x_max+0.025], log=True, color='cornflowerblue') 
    ax1.set_xticks(tick_locs)
    ax1.set_xticklabels([f'{int(round(i))}'.replace('-', '\N{MINUS SIGN}') for i in (100 * np.array(tick_locs))])
    ax1.set_ylim(0.75, 10**digits)
    ax1.text(0.75,0.75, "Linked sites\ncombined", fontsize=AXES_FONTSIZE, transform=ax2.transAxes, ha='center', va='center')
    ax1.tick_params(labelsize=6, width=SPINE_LW)
    for line in ['right', 'top']:
        ax1.spines[line].set_visible(False)
    
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.20)
    plt.savefig(os.path.join(fig_path, out_file+'.png'), dpi=1200)
    
    
def auroc_comp_plot(folder, parameter, parameter_label=None, fig_width=5.5):
    """ Given a folder containing files that have replicate simulations varying some parameter, 
    plots the AUROC score for the different values of this parameter."""
    
    if not parameter_label:
        parameter_label = parameter
    param_values, auc_ben, auc_del = [], [], []
    for file in sorted(os.listdir(folder)):
        filename = os.fsdecode(file)
        filepath = os.path.join(folder, filename)
        data     = np.load(filepath, allow_pickle=True)
        inferred = data['inferred']
        actual   = data['actual']
        idx      = filename.find(parameter)
        idx2     = min(filename.find('-', idx), filename.find('.npz', idx))
        if idx2 == -1:
            idx2 = -4
        param_values.append(int(filename[idx + len(parameter):idx2]))
        auc_ben.append(calculate_AUROC(actual, inferred, deli=False))
        auc_del.append(calculate_AUROC(actual, inferred, ben=False))
    indices      = np.argsort(param_values)
    param_values = np.array(param_values)[indices]
    auc_ben      = np.array(auc_ben)[indices]
    auc_del      = np.array(auc_del)[indices]
    
    fig  = plt.figure(figsize=[cm_to_inch(fig_width), cm_to_inch(5.5/(1.618**2))])
    grid = matplotlib.gridspec.GridSpec(3, len(param_values)+2, figure=fig, wspace=0, hspace=0)
    ax0a  = fig.add_subplot(grid[0, 0:2])
    ax0a.text(0.5, 0.5, parameter_label.capitalize(), fontsize=6, transform=ax0a.transAxes, ha='center', va='center')
    ax0b = fig.add_subplot(grid[1, 0:2])
    ax0b.text(0.5, 0.5, 'AUROC\nBeneficial', fontsize=6, transform=ax0b.transAxes, ha='center', va='center')
    ax0c = fig.add_subplot(grid[2, 0:2])
    ax0c.text(0.5, 0.5, 'AUROC\nDeleterious', fontsize=6, transform=ax0c.transAxes, ha='center', va='center')
    axes = []
    axes.append(ax0a)
    axes.append(ax0b)
    axes.append(ax0c)
    for i in range(len(param_values)):
        ax0 = fig.add_subplot(grid[0, i+2])
        ax0.text(0.5, 0.5, param_values[i], fontsize=5, transform=ax0.transAxes, ha='center', va='center')
        ax1 = fig.add_subplot(grid[1, i+2])
        ax1.text(0.5, 0.5, round(auc_ben[i],3), fontsize=5, transform=ax1.transAxes, ha='center', va='center')
        #ax1.set_title(param_values[i], fontsize=6)
        ax2 = fig.add_subplot(grid[2, i+2])
        ax2.text(0.5, 0.5, round(auc_del[i],3), fontsize=5, transform=ax2.transAxes, ha='center', va='center')
        axes.append(ax1)
        axes.append(ax2)
        axes.append(ax0)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for line in ['right', 'left', 'top', 'bottom']:
            axis.spines[line].set_linewidth(SPINE_LW)
    plt.savefig(os.path.join(fig_path, f'AUROC-{parameter_label}.png'), dpi=1200)
    
    
def auroc_plots_combined(sim_dir, t_dir, samp_dir, n_dir):
    """ Given folders containing replicate simulations varying the parameters, make a combined plot of them."""
    
    def find_auroc(folder, parameter):
        """ Finds and sorts the auroc for the files in the folder and return the auroc's and the paramater values"""
        param_values, auc_ben, auc_del = [], [], []
        for file in sorted(os.listdir(folder)):
            filename = os.fsdecode(file)
            filepath = os.path.join(folder, filename)
            data     = np.load(filepath, allow_pickle=True)
            inferred = data['inferred']
            actual   = data['actual']
            idx      = filename.find(parameter)
            idx2     = min(filename.find('-', idx), filename.find('.npz', idx))
            if idx2 == -1:
                idx2 = -4
            #print(filename, idx, idx2)
            param_values.append(int(filename[idx + len(parameter):idx2]))
            auc_ben.append(calculate_AUROC(actual, inferred, deli=False))
            auc_del.append(calculate_AUROC(actual, inferred, ben=False))
        indices      = np.argsort(param_values)
        param_values = np.array(param_values)[indices]
        auc_ben      = np.array(auc_ben)[indices]
        auc_del      = np.array(auc_del)[indices]
        return param_values, auc_ben, auc_del
    
    def make_subplot(param_vals, beni, deli, parameter=None, axis=None, xticks=None):
        """ Make the subplot for the given parameters"""
        sns.lineplot(param_vals, beni, color='cornflowerblue', lw=1, ax=axis)
        sns.lineplot(param_vals, deli, color='lightcoral',     lw=1, ax=axis)
        axis.set_ylim(0.6, 1)
        for line in ['right', 'top']:
            axis.spines[line].set_visible(False)
        for line in ['bottom', 'left']:
            axis.spines[line].set_linewidth(SPINE_LW)
        axis.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW, length=3)
        axis.set_ylabel('AUROC', fontsize=AXES_FONTSIZE)
        axis.set_xlabel(parameter.capitalize(), fontsize=AXES_FONTSIZE)
        if xticks:
            axis.set_xticks(xticks)
    
    sim,  ben_sim,  del_sim  = find_auroc(sim_dir,  'sim')
    t,    ben_t,    del_t    = find_auroc(t_dir,    'generations')
    samp, ben_samp, del_samp = find_auroc(samp_dir, 'sample')
    n,    ben_n,    del_n    = find_auroc(n_dir,    'N')
    
    fig  = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH/(1.618)])
    grid = matplotlib.gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.4)
    ax0  = fig.add_subplot(grid[1, 1])
    make_subplot(samp, ben_samp, del_samp, parameter='samples per generation',  axis=ax0)
    ax1  = fig.add_subplot(grid[0, 1])
    make_subplot(t,    ben_t,    del_t,    parameter='generations',             axis=ax1)
    ax2  = fig.add_subplot(grid[1, 0])
    sim_ticks = [1, 5, 10, 15, 20]
    make_subplot(sim,  ben_sim,  del_sim,  parameter='simulations',             axis=ax2, xticks=sim_ticks)
    ax3  = fig.add_subplot(grid[0, 0])
    pop_ticks = [100, 400, 700, 1000]
    make_subplot(n,    ben_n,    del_n,    parameter='population size',         axis=ax3, xticks=pop_ticks)
    plt.savefig(os.path.join(fig_path, 'auroc-comparison.png'), dpi=1200)
    
    
def selection_plot(infer_file, syn_file, link_file, protein_file, n_coefficients=8, independent=True, 
                   selection_tv=False, out_file='inferred-SARS', color_tol=0.01, variant_file=None):
    """ Plots the selection coefficients next to their trajectories in the various regions."""
    
    def traj_mean(traj, times, muts):
        """ Given the trajectories in a continent, finds and returns the average trajectory, 
        new times for the average trajectory, and the name of all of the mutant sites"""
        muts_all = np.unique([muts[i][j] for i in range(len(muts)) for j in range(len(muts[i]))])
        #print(traj)
        #times_all = np.arange(np.amin(times), np.amax(times)+1)
        #traj_all  = np.zeros((len(times_all), len(muts_all)))
        traj_all, times_all = [], []    # average frequency trajectories and times for each mutation
        for i in range(len(muts_all)):
            # Determine which subregions have the mutation and the times that it shows up in each
            traj_spec, times_spec = [], []
            for j in range(len(muts)):
                if muts_all[i] in muts[j]:
                    traj_spec.append(traj[j][:, list(muts[j]).index(muts_all[i])])
                    times_spec.append(times[j])
            t_min    = np.amin([np.amin(t) for t in times_spec])
            t_max    = np.amax([np.amax(t) for t in times_spec])
            mut_time = np.arange(t_min, t_max+1)
            traj_av  = np.zeros(len(mut_time))    # the averaged trajectory
            times_all.append(mut_time)
            # Average the trajectories at the times where multiple show up
            for j in range(len(mut_time)):
                norm = max(np.sum([mut_time[j] in times_spec[k] for k in range(len(times_spec))]), 1)
                for k in range(len(traj_spec)):
                    if mut_time[j] in times_spec[k]:
                        traj_av[j] += traj_spec[k][list(times_spec[k]).index(mut_time[j])]
                traj_av[j] /= norm
                if np.sum([mut_time[j] in times_spec[k] for k in range(len(times_spec))])==0:
                    traj_av[j] = np.nan
            traj_all.append(traj_av)
        return traj_all, times_all, muts_all
        
    def filter_linked(traj, times, muts, linked):
        """ Given the linked sites, and the trajectories, times, and mutant sites in a continent,
        filters out and returns the ones that correspond to the linked sites."""
        new_traj, new_times, new_muts = [], [], []
        for i in range(len(linked)):
            if linked[i][0] in list(muts):
                # CHANGE THIS SO IT DOESN'T ONLY USE THE FIRST IN THE LIST OF THE LINKED SITES
                new_traj.append(traj[list(muts).index(linked[i][0])])
                new_times.append(times[list(muts).index(linked[i][0])])
                new_muts.append(linked[i][0])
        return new_traj, new_times, new_muts
    
    def traj_mean_alt(times_link, traj_link):
        """ Given the linked trajectories and times in a specific continent find the averaged trajectories"""
        new_traj, new_times = [], []
        for i in range(len(traj_link)):
            if len(times_link[i])>0:
                t_min  = np.amin([np.amin(t) for t in times_link[i]])
                t_max  = np.amax([np.amax(t) for t in times_link[i]])
                t_temp = np.arange(t_min, t_max+1)
                traj_temp = np.zeros(len(t_temp))
                for j in range(len(t_temp)):
                    norm = max(np.sum([t_temp[j] in times_link[i][k] for k in range(len(times_link[i]))]), 1)
                    for k in range(len(traj_link[i])):
                        if t_temp[j] in times_link[i][k]:
                            traj_temp[j] += traj_link[i][k][list(times_link[i][k]).index(t_temp[j])]
                    traj_temp[j] /= norm
                    if np.sum([t_temp[j] in times_link[i][k] for k in range(len(times_link[i]))])==0:
                        traj_temp[j] = np.nan
                new_times.append(t_temp)
                new_traj.append(traj_temp)
            else:
                new_times.append([])
                new_traj.append([])
        return new_times, new_traj
    
    def find_linked_coefficients(linked_sites, labels, inferred, error, n_sims, indices):
        """ Finds the sum of the coefficients for linked sites"""
        inferred_link = np.zeros(len(linked_sites))
        error_link    = np.zeros(len(linked_sites))
        for i in range(len(linked_sites)):
            for j in range(len(linked_sites[i])):
                if linked_sites[i][j] in list(labels):
                    loc = list(labels).index(linked_sites[i][j])
                    inferred_link[i] += inferred[loc]
                    error_link[i]    += error[loc]
        error_link = np.sqrt(error_link)
        ### Since the labels have already been ordered, maybe don't reapply the indices
        #error_link    = np.array(error_link)[indices]
        #inferred_link = inferred_link[indices]
        return inferred_link, error_link                    
                
    
    inf_data = np.load(infer_file, allow_pickle=True)   # Data transfered over from inference script
    traj     = inf_data['traj']                         # Frequency trajectories for each region and each allele
    inferred = inf_data['selection']                    # Selection coefficients from inference script
    error    = inf_data['error_bars']                   # Error bars for selection coeffiicents
    times    = inf_data['times']                        # The times at which the genomes were collected (with January 1 as a starting point)
    mutant_sites  = inf_data['mutant_sites']            # The genome positions at which there are mutations in each region
    allele_number = inf_data['allele_number']           # The genome positions at which there are mutations across all regions
    locations     = inf_data['locations']               # The labels for the locations (and the dates) from which sequences were drawn
    inferred_ind  = inf_data['selection_independent']   # The inferred coefficients ignoring linkage between sites (i.e., ignoring the covariances)
    print('number of mutations:', len(inferred))
    
    labels = []        # mutation locations across all populations
    for site in allele_number:
        if isinstance(allele_number[0], int):
            labels.append(get_label(site))
        else:
            labels.append(get_label(int(site[:-2])) + site[-2:])
    
    full_labels = []       # mutation locations for each population
    for i in range(len(mutant_sites)):
        labels_temp = []
        for j in range(len(mutant_sites[i])):
            loc = (mutant_sites[i][j] == allele_number).nonzero()[0][0]
            labels_temp.append(labels[loc])
        full_labels.append(np.array(labels_temp))
    
    continents    = np.array([i.split('-')[0] for i in locations])     # The continents
    locations     = ['-'.join(i.split('-')[1:]) for i in locations]    # The locations without the continent labels
    
    max_time = np.amax(np.array([i[-1] for i in times]))
    min_time = np.amin(np.array([i[0] for i in times]))
    total_days  = max_time - min_time + 1
    simulations = len(traj)
    
    type_data = np.load(syn_file, allow_pickle=True)
    mutant_types_temp  = type_data['types']
    sites = type_data['locations']
    mutant_types = []
    for i in allele_number:
        if i in sites:
            mutant_types.append(mutant_types_temp[list(sites).index(i)])
        else:
            mutant_types.append('unknown')
    
    # Finding location labels and times
    locations_short = []         # just the locations without the times included
    time_stamps     = []         # just the timestamps without the location
    for i in range(len(locations)):
        if locations[i][:3] == 'usa':
            locations_short.append(locations[i][:locations[i].find('-', 4)])
        elif locations[i][:14] == 'united kingdom':
            locations_short.append(locations[i][:locations[i].find('-', 15)])
        else:
            locations_short.append(locations[i][:locations[i].find('-')])
        time_stamps.append(locations[i][-11:])
        
    # Finding the absolute times corresponding to each trajectory
    real_times = []
    for loc in locations:
        """
        time_start = dt.date.fromisoformat('2020-' + loc[-11:-6]) - dt.date(2020, 1, 1)
        time_end   = dt.date.fromisoformat('2020-' + loc[-5:])    - dt.date(2020, 1, 1)
        """
        number_idxs = np.array([loc.find(i) for i in np.array(np.arange(10), dtype=str)])
        number_idxs = number_idxs[number_idxs>=0]
        loc_dates   = loc[np.amin(number_idxs):].split('-')
        if len(loc_dates)>6:
            loc_dates = loc_dates[-6:]
        loc_new     = []
        for i in loc_dates:
            if len(i)==1:
                num = '0' + i
            else:
                num = i
            loc_new.append(num)
        loc_dates   = loc_new
        start_dates = '-'.join(loc_dates[:3])
        end_dates   = '-'.join(loc_dates[3:])
        time_start  = dt.date.fromisoformat(start_dates) - dt.date(2020, 1, 1)
        time_end    = dt.date.fromisoformat(end_dates)   - dt.date(2020, 1, 1)
        real_times.append(np.arange(time_start.days, time_end.days+1))
    
    """
    # Find the averaged trajectories in Europe, North America, and elsewhere.
    traj_av_e, times_av_e, muts_av_e = traj_mean(traj_e, times_e, muts_e)
    traj_av_a, times_av_a, muts_av_a = traj_mean(traj_a, times_a, muts_a)
    traj_av_o, times_av_o, muts_av_o = traj_mean(traj_o, times_o, muts_o)
    muts_av_e = [get_label(i) for i in muts_av_e]
    muts_av_a = [get_label(i) for i in muts_av_a]
    muts_av_o = [get_label(i) for i in muts_av_o]
    """
    
    greys         = cm.get_cmap('Greys', 4)
    #colormap      = cm.get_cmap('Dark2', 4)
    #colors_main   = [colormap(0), colormap(1), 'k']
    #colors_all    = [colormap(0), colormap(1)] + [greys(i) for i in range(4)]
    colors_main   = sns.husl_palette(3, l=0.4)
    colors_all    = [i for i in colors_main[:2]] + [greys(i) for i in range(4)]
    colors_group  = sns.husl_palette(n_coefficients)    # colors for the different groups of linked mutations

    # Process linked data and order coefficients 
    linked_sites  = np.load(link_file, allow_pickle=True)
    
    if variant_file:
        variants = np.load(variant_file, allow_pickle=True)
        
        # removing groups of linked sites that contain sites that are in the defined variants
        
        ### FUTURE: change this so that it if a variant is present in both the liked sites file and the variant file, the two lists are combined
        variant_muts = [variants[i][j] for i in range(len(variants)) for j in range(len(variants[i]))]
        idxs_remove   = []
        for i in range(len(variant_muts)):
            for j in range(len(linked_sites)):
                if variant_muts[i] in linked_sites[j] and j not in idxs_remove:
                    idxs_remove.append(j)
        idxs_keep = [i for i in range(len(linked_sites)) if i not in idxs_remove]
        linked_sites = list(linked_sites[np.array(idxs_keep)])  
        
        # adding the groups of mutations belonging to the variants to the linked sites
        for i in range(len(variants)):
            linked_sites.append(variants[i])
    
    inferred_link = np.zeros(len(linked_sites))    # the sum of the inferred coefficients for the linked sites
    ind_link      = np.zeros(len(linked_sites))    # the inferred coefficients ignoring covariances
    error_link    = np.zeros(len(linked_sites))    # the error for the inferred summed coefficients
    error_ind     = np.zeros(len(linked_sites))    # the error for the coefficients ignorning linkage
    counter       = np.zeros(len(linked_sites))    # counts the number of linked sites in each group
    traj_link, link_labels, label_types, traj_locs, traj_times = [], [], [], [], []
    traj_cont = []
    for i in range(len(linked_sites)):
        traj_temp, labels_temp, types_temp, traj_locs_temp, traj_times_temp = [], [], [], [], []
        traj_cont_temp = []
        for j in range(simulations):
            if np.any(linked_sites[i][0]==np.array(full_labels[j])):
                loc2 = np.where(linked_sites[i][0]==np.array(full_labels[j]))[0][0]
                traj_temp.append(traj[j][:, loc2])
                traj_locs_temp.append(locations[j])
                traj_times_temp.append(times[j])
                traj_cont_temp.append(continents[j])
        for j in range(len(linked_sites[i])):
            if np.any(linked_sites[i][j]==np.array(labels)):
                loc = np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]
                if not selection_tv:
                    inferred_link[i] += inferred[loc]
                    error_link[i]    += error[loc] ** 2
                    ind_link[i]      += inferred_ind[loc]   
                else:
                    inferred_link[i] += inferred[-1][loc]
                    error_link[i]    += error[-1][loc] ** 2
                counter[i]       += 1
                labels_temp.append(labels[loc])
                types_temp.append(mutant_types[loc])
        #link_labels.append(' '.join(labels_temp))
        link_labels.append(labels_temp)
        traj_link.append(traj_temp)
        label_types.append(types_temp)
        traj_locs.append(traj_locs_temp)
        traj_times.append(traj_times_temp)
        traj_cont.append(traj_cont_temp)
    counter[counter==0] = 1
    #error_link = np.sqrt(error_link) / np.sqrt(counter)
    error_link = np.sqrt(error_link)
    ind_link   = ind_link / counter
        
    # Filter out linked sites groups that are not found in the data.
    label_lengths = [len(link_labels[i]) for i in range(len(link_labels))]
    inferred_link = inferred_link[np.array(label_lengths).nonzero()[0]]
    if not selection_tv:
        indices   = np.argsort(np.absolute(inferred_link))[::-1][:n_coefficients]
    else:
        indices   = np.argsort(np.absolute(inferred_link))[::-1]
    link_labels   = np.array(link_labels)[np.array(label_lengths).nonzero()[0]][indices]
    traj_link     = np.array(traj_link)[np.array(label_lengths).nonzero()[0]][indices]
    label_types   = np.array(label_types)[np.array(label_lengths).nonzero()[0]][indices]
    traj_locs     = np.array(traj_locs)[np.array(label_lengths).nonzero()[0]][indices]
    error_link    = np.array(error_link)[np.array(label_lengths).nonzero()[0]][indices]
    traj_times    = np.array(traj_times)[np.array(label_lengths).nonzero()[0]][indices]
    traj_cont     = np.array(traj_cont)[np.array(label_lengths).nonzero()[0]][indices]
    ind_link      = np.array(ind_link)[np.array(label_lengths).nonzero()[0]][indices]
    error_ind     = np.array(error_ind)[np.array(label_lengths).nonzero()[0]][indices]
    inferred_link = inferred_link[indices]
    label_lengths = np.array([len(link_labels[i]) for i in range(len(link_labels))])
    #for i in range(len(inferred_link)):
    #    print(f'Group {i}', inferred_link[i])
    
    # relabeling the B117 variant 
    """
    for i in range(len(link_labels)):
        if len(link_labels[i])>10 and 'S-143' in [j[:-2] for j in link_labels[i]]:
            link_labels[i] = ['B1.1.7']
    """
            
    # finding the protein changes that correspond to the linked sites
    initial_aas  = []
    final_aas    = []
    prot_labels  = []
    for line in open(protein_file).readlines():
        temp = line.split(',')
        initial_aas.append(temp[1][0])
        final_aas.append(temp[1][2])
        prot_labels.append(temp[0])
    aa_changes_full = [initial_aas[i] + prot_labels[i][prot_labels[i].find('-')+1:-2] + final_aas[i] for i in range(len(prot_labels))]
    aa_changes_link = []
    for i in range(len(link_labels)):
        changes_temp = []
        for j in range(len(link_labels[i])):
            #print(link_labels[i][j])
            changes_temp.append(aa_changes_full[prot_labels.index(link_labels[i][j])])
        aa_changes_link.append(changes_temp)        
    
    # separate data into large regions and find average trajectories
    cont_filtered = []
    for i in range(len(traj_cont)):
        cont_temp = []
        for cont in traj_cont[i]:
            if cont!='europe' and cont!='north america':
                cont_temp.append('other')
            else:
                cont_temp.append(cont)
        cont_filtered.append(np.array(cont_temp, dtype=str))
    cont_all  = ['north america', 'europe', 'oceania', 'asia', 'africa', 'south america']
    cont_main = ['north america', 'europe', 'other']    # regions in which frequency trajectories will be averaged
    
    #print(cont_filtered)
    traj_e    = [np.array(traj_link[i])[np.array(cont_filtered[i])=='europe'] for i in range(len(traj_link))]
    times_e   = [np.array(traj_times[i])[np.array(cont_filtered[i])=='europe'] for i in range(len(traj_times))]
    traj_a    = [np.array(traj_link[i])[np.array(cont_filtered[i])=='north america'] for i in range(len(traj_link))]
    times_a   = [np.array(traj_times[i])[np.array(cont_filtered[i])=='north america'] for i in range(len(traj_times))]
    traj_o    = [np.array(traj_link[i])[np.array(cont_filtered[i])=='other'] for i in range(len(traj_link))]
    times_o   = [np.array(traj_times[i])[np.array(cont_filtered[i])=='other'] for i in range(len(traj_times))]
    times_e, traj_e = traj_mean_alt(times_e, traj_e)
    times_a, traj_a = traj_mean_alt(times_a, traj_a)
    times_o, traj_o = traj_mean_alt(times_o, traj_o)
    
    # create figure parameters
    fig_len   = DOUBLE_COLUMN_WIDTH
    fig_width = DOUBLE_COLUMN_WIDTH
    left      = 0.075
    bottom    = 0.05
    width     = 0.875
    sep       = 0.01
    l_height  = 0.5
    u_height  = 0.385
    
    # get the protein lengths in the proper order
    prot_lengths = []
    for i in PROTEINS:
        prot_lengths.append(PROTEIN_LENGTHS[i])
    norm_pl = prot_lengths / np.sum(prot_lengths)
    
    #prot_lengths = [6, 2, 6, 2, 1, 4, 1, 1, 1, 1, 6, 6, 6, 1, 2, 10, 4, 1, 2, 1, 1, 1, 6, 4, 2]  
    #prot_labels   = ['nsp1', '2', 'nsp3', '4', '5', 'nsp6', '7', '8', '9', '10', 'nsp12', 'nsp13', 'nsp14', 
    #                '15', '16', 'S', 'orf3a', 'E', 'M', '6', '7a', '7b', 'ORF8', 'N', '10']
    
    prot_labels   = ['nsp1', 'nsp2', 'nsp3', 'nsp4', 'nsp5', 'nsp6', 'nsp7', 'nsp8', 'nsp9', 'nsp10', 'nsp12', 'nsp13', 'nsp14', 
                    'nsp15', 'nsp16', 'S', 'orf3a', 'E', 'M', 'orf6', 'orf7a', 'orf7b', 'orf8', 'N', 'orf10']
    height_ratios = [1, 4, 1, 2, 1, 2, 2, 1, 1] 
    label_offset  = 0.1
    
    # Making paramters for lines connecting to labels
    connect_lw     = 0.1
    connect_width  = 0.01
    text_length    = 0.6  # length of lines connecting labels to proteins in axes coordinates
    text_width     = 0.2
    connect_length = 0.002
    connect_style  = matplotlib.patches.ConnectionStyle.Angle(angleA=90)
    
    # Make the figure
    s_max = np.amax(inferred_link) + 0.02
    fig   = plt.figure(figsize=[fig_width, fig_len])
    
    # make the genome plot
    gen_grid  = matplotlib.gridspec.GridSpec(len(link_labels)+1, len(PROTEINS), width_ratios=prot_lengths, height_ratios=height_ratios, hspace=0, 
                                             wspace=0, figure=fig, left=left, bottom=bottom+sep+l_height, right=left+width, top=bottom+sep+l_height+u_height)
    inv_fig = fig.transFigure.inverted()
    # the plot of the genome
    facecolor = plt.get_cmap('Greys')(0.2)
    """
    for i in range(len(PROTEINS)):
        ax = fig.add_subplot(gen_grid[0, i])
        ax.set_xticks([])
        ax.set_yticks([])
        if PROTEINS[i] in ['NSP8', 'NSP10', 'NSP15', 'ORF7a']:y_height = 1.1
        else: y_height = 0.5
        if PROTEINS[i]=='ORF7a':
            ax.text(0.6, y_height, '7a/b', transform=ax.transAxes, fontsize=6, ha='center', va='center')
        elif PROTEINS[i]!='ORF7b':
            ax.text(0.5, y_height, prot_labels[i], transform=ax.transAxes, fontsize=6, ha='center', va='center')
        if (i % 2)==0:
            ax.set_facecolor(facecolor)
        if i!=0:
            ax.spines['left'].set_visible(False)
        if i!=len(PROTEINS)-1:
            ax.spines['right'].set_visible(False)
    """
            
            
    ### Code for doing the protein labels
    for i in range(len(PROTEINS)):
        ax = fig.add_subplot(gen_grid[0, i])
        ax.set_xticks([])
        ax.set_yticks([])
        if PROTEINS[i] in ['NSP1', 'NSP9', 'NSP15', 'ORF7b', 'ORF10']:
            ax.text(0.5 + (connect_width + connect_length)/norm_pl[i], 1 + text_length, PROTEINS[i].lower(), transform=ax.transAxes, fontsize=AXES_FONTSIZE, ha='left', va='center', zorder=998)
            #ax.plot([0.5, 0.5], [1, 1 + connect_length],   lw=connect_width, clip_on=False, color='k')
            con = matplotlib.patches.ConnectionPatch(xyA=(0.5, 1), xyB=(0.5 + connect_width/norm_pl[i], 1 + text_length), coordsA=ax.transAxes, coordsB=ax.transAxes, 
                                                     axesA=ax, axesB=ax, arrowstyle='-', ls='-', lw=connect_lw, connectionstyle=connect_style, zorder=997, alpha=0.5)
            ax.add_artist(con)
        elif PROTEINS[i] in ['NSP5', 'ORF3a', 'ORF6']:
            ax.text(0.5 - (connect_width + connect_length)/norm_pl[i], 1 + text_length, PROTEINS[i].lower(), transform=ax.transAxes, fontsize=AXES_FONTSIZE, ha='right', va='center', zorder=998)
            #ax.plot([0.5, 0.5], [1, 1 + connect_length],   lw=connect_width, clip_on=False, color='k')
            con = matplotlib.patches.ConnectionPatch(xyA=(0.5, 1), xyB=(0.5 - connect_width/norm_pl[i], 1 + text_length), coordsA=ax.transAxes, coordsB=ax.transAxes, 
                                                     axesA=ax, axesB=ax, arrowstyle='-', ls='-', lw=connect_lw, connectionstyle=connect_style, zorder=997, alpha=0.5)
            ax.add_artist(con)
        elif PROTEINS[i] in ['NSP6', 'ORF7a']:
            ax.text(0.5 - (connect_width + connect_length)/norm_pl[i], 1 + text_length * 2.5, PROTEINS[i].lower(), transform=ax.transAxes, fontsize=AXES_FONTSIZE, ha='right', va='center', zorder=998)
            #ax.plot([0.5, 0.5], [1, 1 + connect_length],   lw=connect_width, clip_on=False, color='k')
            con = matplotlib.patches.ConnectionPatch(xyA=(0.5, 1), xyB=(0.5 - connect_width/norm_pl[i], 1 + text_length*2.5), coordsA=ax.transAxes, coordsB=ax.transAxes, 
                                                     axesA=ax, axesB=ax, arrowstyle='-', ls='-', lw=connect_lw, connectionstyle=connect_style, zorder=997, alpha=0.5)
            ax.add_artist(con)
        elif PROTEINS[i] in ['NSP10', 'NSP16', 'ORF8']:
            ax.text(0.5 + (connect_width + connect_length)/norm_pl[i], 1 + text_length * 2.5, PROTEINS[i].lower(), transform=ax.transAxes, fontsize=AXES_FONTSIZE, ha='left', va='center', zorder=998)
            #ax.plot([0.5, 0.5], [1, 1 + connect_length*2], lw=connect_width, clip_on=False, color='k')
            con = matplotlib.patches.ConnectionPatch(xyA=(0.5, 1), xyB=(0.5 + connect_width/norm_pl[i], 1 + text_length*2.5), coordsA=ax.transAxes, coordsB=ax.transAxes, 
                                                     axesA=ax, axesB=ax, arrowstyle='-', ls='-', lw=connect_lw, connectionstyle=connect_style, zorder=997, alpha=0.5)
            ax.add_artist(con)
        elif PROTEINS[i]=='NSP8':
            ax.text(0.5, 1 + text_length * 2.5, PROTEINS[i].lower(), transform=ax.transAxes, fontsize=AXES_FONTSIZE, ha='center', va='center', zorder=998)
            #ax.plot([0.5, 0.5], [1, 1 + connect_length],   lw=connect_width, clip_on=False, color='k')
            con = matplotlib.patches.ConnectionPatch(xyA=(0.5, 1), xyB=(0.5 , 1 + text_length*2.5 - 0.1), coordsA=ax.transAxes, coordsB=ax.transAxes, 
                                                     axesA=ax, axesB=ax, arrowstyle='-', ls='-', lw=connect_lw, connectionstyle=connect_style, zorder=997, alpha=0.5)
            ax.add_artist(con)    
        elif PROTEINS[i]=='NSP7':
            ax.text(0.5 - (connect_width + connect_length - 0.005)/norm_pl[i], 1 + text_length, PROTEINS[i].lower(), transform=ax.transAxes, fontsize=AXES_FONTSIZE, ha='right', va='center', zorder=998)
            #ax.plot([0.5, 0.5], [1, 1 + connect_length],   lw=connect_width, clip_on=False, color='k')
            con = matplotlib.patches.ConnectionPatch(xyA=(0.5, 1), xyB=(0.5 - connect_width/norm_pl[i], 1 + text_length), coordsA=ax.transAxes, coordsB=ax.transAxes, 
                                                     axesA=ax, axesB=ax, arrowstyle='-', ls='-', lw=connect_lw, connectionstyle=connect_style, zorder=997, alpha=0.5)
            ax.add_artist(con)
        elif PROTEINS[i] in ['S', 'E', 'N', 'M']:
            ax.text(0.5, 0.5, PROTEINS[i], transform=ax.transAxes, fontsize=AXES_FONTSIZE, ha='center', va='center')
        else:
            ax.text(0.5, 0.5, PROTEINS[i].lower(), transform=ax.transAxes, fontsize=AXES_FONTSIZE, ha='center', va='center')
        #if PROTEINS[i]=='ORF7a':
        #    ax.text(0.6, y_height, '7a/b', transform=ax.transAxes, fontsize=6, ha='center', va='center')
        #elif PROTEINS[i]!='ORF7b':
        #    ax.text(0.5, y_height, prot_labels[i], transform=ax.transAxes, fontsize=6, ha='center', va='center')
        if (i % 2)==0:
            ax.set_facecolor(facecolor)
        if i!=0:
            ax.spines['left'].set_visible(False)
            for line in ['right', 'top', 'bottom']:
                ax.spines[line].set_linewidth(SPINE_LW)
        elif i!=len(PROTEINS)-1:
            ax.spines['right'].set_visible(False)
            for line in ['left', 'top', 'bottom']:
                ax.spines[line].set_linewidth(SPINE_LW)
            
    """
    # the group labels in the first column
    for i in range(len(link_labels)):
        ax = fig.add_subplot(gen_grid[i+1, 0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, f'Group {i+1}', transform=ax.transAxes, fontsize=8, ha='center', va='center', color=colors_group[i])
    """
    # Finding the labels for the different groups of linked sites
    #variant_identifiers = {'N-67-0' : 'B.1.2', 'S-13-1' : 'B.1.429', 'S-613-2' : 'A.23.1', 'S-570-1' : 'B.1.1.7', 'S-477-1' : '20E_EU1', 'S-614-1' : 'B.1', 'S-190-2' : 'P.1', 'S-142-1' : 'B.1.617',
    #                       'S-222-1' : '20E(EU1)'}
    variant_identifiers = {'S-13-1-T' : 'epsilon', 'S-570-1-A' : 'alpha', 'S-222-1-T' : '20E_EU1', 'S-1176-0-T' : 'gamma', 
                           'S-19-1-G' : 'delta', 'S-80-1-C' : 'beta', 'S-76-1-T' : 'lambda', 'S-5-0-T' : 'iota'}
    group_labels = [f'None' for i in range(len(link_labels))]
    num_unnamed  = 0
    if variant_file:
        for i in range(len(link_labels)):
            for j in variant_identifiers:
                if j in link_labels[i]:
                    group_labels[i] = variant_identifiers[j]
            if group_labels[i] == 'None':
                group_labels[i] = f'Group {num_unnamed + 1}'
                num_unnamed +=1
    else:
        group_labels = [f'Group {i+1}' for i in range(len(link_labels))]
        
    # plotting the locations of the mutations in each group of linked sites
    xcoords, ycoords, txt = [], [], []    # the text coordinates
    datax, datay = [], []    # the data coords
    dot_colors   = []    # the colors for the data markers
    for i in range(len(link_labels)):
        positions_group = []
        labels_group    = []
        aa_group        = []
        axes_prot       = []
        dot_color_temp  = []
        for k in range(len(PROTEINS)):
            positions_temp = []
            labels_temp    = []
            aa_temp        = []
            for j in range(len(link_labels[i])):
                label_temp   = link_labels[i][j]
                if link_labels[i][j][:link_labels[i][j].find('-')]==PROTEINS[k]:
                    if int(label_temp[label_temp.find('-')+1:-4]) not in positions_temp:
                        aa = aa_changes_link[i][j].replace('?','-')
                        if aa[-1]=='-':
                            aa = '\u0394' + aa[1:-1]    # replace deletions notation
                        if aa[0]==aa[-1]: 
                            dot_color_temp.append('gray')    # color synonymous mutations grey
                            aa = aa[1:-1]    # don't list the amino acid since no mutation has occured
                        else:             
                            dot_color_temp.append(colors_group[i])
                        positions_temp.append(int(label_temp[label_temp.find('-')+1:-4]))
                        labels_temp.append(label_temp)
                        aa_temp.append(aa)
            positions_group.append(positions_temp)
            labels_group.append(labels_temp)
            aa_group.append(aa_temp)
        dot_colors.append(dot_color_temp)
        #print(aa_group)
        #print(positions_group)
        for j in range(len(PROTEINS)):
            # shift the vertical locations of dots and labels to prevent overlapping when mutations are too close
            #n_muts      = len(positions_group[j])
            #y_cycle     = np.array([0.75, 0.6, 0.45, 0.3, 0.15])
            #y_positions = [y_cycle[i % 5] for i in range(n_muts)]
            #if i==0:
            #    axg = fig.add_subplot(gen_grid[1:3, j])    # give the first group two columns due to the number of mutations
            #    label_offset = 0.05
            #else:
            #    axg = fig.add_subplot(gen_grid[i+2,j])
            #    label_offset = 0.1
            axg  = fig.add_subplot(gen_grid[i+1, j])
            axg.set_xticks([])
            axg.set_yticks([])
            axg.set_xlim(0, PROTEIN_LENGTHS[PROTEINS[j]])
            axg.set_ylim(0, 1)
            #data_to_fig = axg.transData + inv_fig
            
            if j==0:
                axg.set_ylabel(group_labels[i], fontsize=AXES_FONTSIZE, color=colors_group[i], rotation='horizontal', labelpad=20)
            if i==0:
                axg.spines['bottom'].set_visible(False)
                axg.spines['top'].set_linewidth(SPINE_LW)
            elif i==len(link_labels)-1:
                axg.spines['top'].set_visible(False)
                axg.spines['bottom'].set_linewidth(SPINE_LW)
            else:
                for line in ['top', 'bottom']:
                    axg.spines[line].set_visible(False)
            if j==0:
                axg.spines['right'].set_visible(False)
                for line in ['left']:
                    axg.spines[line].set_linewidth(SPINE_LW)
            elif j==len(PROTEINS)-1:
                axg.spines['left'].set_visible(False)
                for line in ['right']:
                    axg.spines[line].set_linewidth(SPINE_LW)
            else:
                for line in ['right', 'left']:
                    axg.spines[line].set_visible(False)
            if (j % 2)==0:
                axg.set_facecolor(facecolor)
            axes_prot.append(axg)
        
        plt.draw()
        x_temp, y_temp =[], []
        for j in range(len(PROTEINS)):
            # adjust the locations of the dots to help them not to overlap
            n_muts = len(positions_group[j])
            if n_muts == 1:
                y_cycle = [0.5]
            elif 1 < n_muts < 4:
                y_cycle = np.linspace(0 + 1/n_muts, 1, n_muts, endpoint=False)
            else:    
                y_cycle = np.linspace(0.1, 1, 5, endpoint=False)
            y_positions = [y_cycle[i % len(y_cycle)] for i in range(n_muts)]
            #axg = fig.add_subplot(gen_grid[i+1, j])
            axg = axes_prot[j]
            data_to_fig = axg.transData + inv_fig
            if len(positions_group[j])>0:
                #axg.plot(positions_group[j], y_positions, '.', linewidth=0, color=colors_group[i])
                for k in range(len(positions_group[j])):
                    #print(aa_group[j][k])
                    if   aa_group[j][k]=='N501Y':
                        coords = data_to_fig.transform((-20*label_offset    + positions_group[j][k], -5*label_offset/height_ratios[i+1]    + y_positions[k]))
                    elif aa_group[j][k] in ['A570V', 'S235F', 'T716I', 'D3Y', 'S13I', 'Q38P', 'V202I', 'S982A', 'Q27*', 'P504L', 'D3L', 'V30L', 'A222V',
                                            'L452R', 'F1089F', 'A890D', 'T183I', 'I1683T', 'D614G', 'P67T', 'I65V', 'V530V']:
                        coords = data_to_fig.transform((7.5*label_offset*200 + positions_group[j][k],                                       y_positions[k]))
                    #elif aa_group[j][k]=='R52I':
                    #    coords = data_to_fig.transform((-100*label_offset*200 + positions_group[j][k],                                      y_positions[k]))
                    elif aa_group[j][k] in ['\u0394'+'144', '\u0394'+'143', 'H69Y']:
                        coords = data_to_fig.transform((-11500*label_offset    + positions_group[j][k], 2.5*label_offset/height_ratios[i+1] + y_positions[k]))
                    elif aa_group[j][k] in ['\u0394' + '70', '\u0394' + '68', '\u0394' + '69']:
                        coords = data_to_fig.transform((-7500*label_offset    + positions_group[j][k], 2.5*label_offset/height_ratios[i+1]  + y_positions[k]))
                    else:    
                        coords = data_to_fig.transform((150*label_offset     + positions_group[j][k], 2.5*label_offset/height_ratios[i+1]   + y_positions[k]))
                    #coords = data_to_fig.transform((positions_group[j][k], y_positions[k]))
                    xcoords.append(coords[0])
                    ycoords.append(coords[1])
                    txt.append(aa_group[j][k])
                    
                    data_coords = data_to_fig.transform((positions_group[j][k], y_positions[k]))
                    x_temp.append(data_coords[0])
                    y_temp.append(data_coords[1])
                    #axg.annotate(aa_group[j][k], (positions_group[j][k], 0.6), xycoords=axg.transAxes)   
                    #axg.text(positions_group[j][k], 0.1 + y_positions[k], aa_group[j][k], transform=axg.transData, fontsize=6)
        datax.append(x_temp)
        datay.append(y_temp)
    for i in range(len(xcoords)):
        plt.annotate(txt[i], (xcoords[i], ycoords[i]), xycoords='figure fraction', fontsize=6)
    #print(txt)
    ax_full = fig.add_subplot(gen_grid[:,:])
    ax_full.axis('off')
    #print(len(datax))
    #print(len(dot_colors))
    for i in range(len(datax)):
        #ax_full.plot(datax[i], datay[i], '.', linewidth=0, color=colors_group[i], transform=fig.transFigure)
        ax_full.scatter(datax[i], datay[i], marker='.', color=dot_colors[i], transform=fig.transFigure)
        
    # grid for selection and trajectory plots 
    wspace = 0.01
    hspace = wspace * n_coefficients / 2
    grid   = matplotlib.gridspec.GridSpec(len(link_labels), 2, wspace=wspace, hspace=hspace, figure=fig,
                                         left=left, bottom=bottom, right=left+width, top=bottom+l_height)
    # make the selection plot
    ax_s = fig.add_subplot(grid[:, 1])
    ax_s.set_xlabel("Selection coefficients (%)", fontsize=AXES_FONTSIZE)
    ax_s.grid(b=True, axis='x', linewidth=0.3)
    ax_s.set_xlim(-s_max*100, s_max*100)
    ax_s.set_ylim(0, len(inferred_link))
    ax_s.set_yticks([])
    ax_s.tick_params(labelsize=TICK_FONTSIZE, length=2, width=SPINE_LW)

    # Set up colors for the inferred coefficients
    colors_link = []
    for i in inferred_link:
        if i>=color_tol:
            colors_link.append(MAIN_COLOR)
        elif i<=-color_tol:
            colors_link.append(COMP_COLOR)
        else:
            colors_link.append('dimgrey')
    
    # plot the selection coefficients
    if not selection_tv:
        #ax_s.errorbar(100*inferred_link[::-1], np.arange(len(inferred_link))+0.5, xerr=100*error_link[::-1],
        #              lw=0, elinewidth=0.5, color=colors_link[::-1], label='Sum of linked sites', fmt='.')
        ax_s.errorbar(100*inferred_link[::-1], np.arange(len(inferred_link))+0.5, xerr=100*error_link[::-1],
                      lw=0, elinewidth=0.5, color='k', fmt='none', ecolor=np.array(colors_group)[::-1])
        ax_s.scatter(100*inferred_link[::-1], np.arange(len(inferred_link))+0.5, s=5,
                     color=np.array(colors_group)[::-1])
        if independent:
            #sns.scatterplot(100*ind_link[::-1], np.arange(len(ind_link))+0.6, label="ignoring linkage", ax=ax_s, size=0.5, legend='brief', color='silver')
            ax_s.errorbar(100*ind_link[::-1], np.arange(len(ind_link))+0.6, xerr=100*error_link[::-1],
                          lw=0, elinewidth=0.5, color='silver', label='Ignoring linkage', fmt='.')
            ax_s.legend()
    else:
        color_map = cm.get_cmap('Blues')
        colors    = [color_map(i/len(inferred)) for i in range(len(inferred))]
        for i in range(len(inferred)):
            inferred_link, error_link = find_linked_coefficients(link_labels, labels, inferred[i], error[i], simulations, indices)
            ax_s.plot(100*inferred_link[::-1], np.arange(len(inferred_link))+0.5, color=colors[i], markersize=3, linewidth=0, marker='.')
        
    for line in ['right', 'left', 'top', 'bottom']:
        ax_s.spines[line].set_linewidth(SPINE_LW)
    ax_s.legend(loc='upper left', fontsize=6, frameon=True)
    for i in range(len(inferred_link)):
        print(f'group {i} has a selection coefficient of {inferred_link[i]}')
    
    
    # Create labels for months for the trajectory plots
    print('initial time in days after January 1st', min_time)
    min_month = 1
    if 31 < min_time < 59:
        min_month  = 2
        min_time2  = min_time - 31
    else:
        min_time2  = min_time
    num_months   = int(total_days / 30) + 1
    months_start = [dt.date(2020, i, 1) for i in range(min_month, 13)] + [dt.date(2021, i, 1) for i in range(1, 13)]
    month_locs   = [(months_start[i] - dt.date(2020, min_month, min_time2)).days for i in range(len(months_start)) if i<num_months]        # x tick locations
    months       = ['Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
                    'Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:num_months]  # x tick labels
    print(month_locs)
    print(months)
    months       = np.array(months)[np.array(month_locs)>0]
    month_locs   = np.array(month_locs)[np.array(month_locs)>0]
    print(month_locs)
    print(months)
    
    # Plot y-axis label for the frequency trajectories
    ax_traj = fig.add_subplot(grid[:, 0])
    for line in ['left', 'right', 'top', 'bottom']:
        ax_traj.spines[line].set_visible(False)
    ax_traj.set_yticks([])
    ax_traj.set_xticks([])
    ax_traj.set_ylabel('Frequency (%)', fontsize=AXES_FONTSIZE, labelpad=20)
    
    # plot the frequency trajectories
    for i in range(len(inferred_link)):
        ax = fig.add_subplot(grid[i, 0])
        if i == len(inferred_link)-1:
            #ax.set_xlabel("Frequency trajectories", fontsize=8, labelpad=10)
            ax.set_xlabel('Date', fontsize=AXES_FONTSIZE, labelpad=5)
            ax.set_xticks(month_locs)
            ax.set_xticklabels(months)
        else:
            ax.set_xticks([])
        
        ax.set_xlim(min_time, max_time)
        ax.set_ylim(0,1)
        ax.set_yticks([])
        """
        if label_lengths[i] <= 10:
            tick_locations = [(j+1/2)/label_lengths[i] for j in range(label_lengths[i])]
            tick_labels    = link_labels[i]
        else:
            tick_locations = [(j+1/2)/10 for j in range(10)]
            tick_labels    = link_labels[i][:10]
            label_types[i] = label_types[i][:10]
        ax.set_yticks(tick_locations)
        ax.set_yticklabels(tick_labels)
        """
        ax.set_ylabel(group_labels[i], color=colors_group[i], fontsize=AXES_FONTSIZE)
        #ax.text(0.05, 0.5, f'Group {i+1}', color=colors_group[i], fontsize=6, ha='center', va='center', transform=ax.transAxes, rotation='vertical')
        ax.tick_params(labelsize=TICK_FONTSIZE, length=2, width=SPINE_LW)
        """
        for j in range(len(label_types[i])):
            if label_types[i][j] == 'S':
                plt.gca().get_yticklabels()[j].set_color("lightgrey")
            elif label_types[i][j] == 'NNS':
                plt.gca().get_yticklabels()[j].set_color("grey")
        """
        
        for k in range(len(traj_link[i])):
            ax.plot(traj_times[i][k] - min_time, traj_link[i][k], color=colors_all[cont_all.index(traj_cont[i][k])], alpha=0.25, lw=0.5)
        
        """
        if link_labels[i][0] in muts_e:
            ax.plot(av_times_e[muts_e.index(link_labels[i][0])], av_link_e[muts_e.index(link_labels[i][0])], color=colors_main[0], lw=0.5)
        if link_labels[i][0] in muts_a:
            ax.plot(av_times_a[muts_a.index(link_labels[i][0])], av_link_a[muts_a.index(link_labels[i][0])], color=colors_main[1], lw=0.5)
        if link_labels[i][0] in muts_o:
            ax.plot(av_times_o[muts_o.index(link_labels[i][0])], av_link_o[muts_o.index(link_labels[i][0])], color=colors_main[2], lw=0.5)
        """
        ax.plot(times_e[i] - min_time, traj_e[i], color=colors_main[1], lw=1, label='Europe')
        ax.plot(times_a[i] - min_time, traj_a[i], color=colors_main[0], lw=1, label='North America')
        ax.plot(times_o[i] - min_time, traj_o[i], color=colors_main[2], lw=1, label='Other')
        #for line in ['right', 'left', 'top', 'bottom']:
        for line in ['top', 'bottom']:
            ax.spines[line].set_linewidth(SPINE_LW)
        for line in ['left', 'right']:
            ax.spines[line].set_visible(False)
        if i == len(inferred_link) - 1:
            ax.legend(loc='upper right', fontsize=6, frameon=False)
    plt.savefig(os.path.join(fig_path, out_file+'.png'), dpi=2500)
    
    
def link_selection_table(infer_file, link_file, prot_file, out_file='linked-coefficients-table'):
    """ Makes a table of the total selection coefficient for groups of linked sites"""
    
    infer_data   = np.load(infer_file, allow_pickle=True)
    linked_sites = np.load(link_file,  allow_pickle=True)
    inferred     = infer_data['selection']
    nuc_sites    = infer_data['allele_number']
    errors       = infer_data['error_bars']
    mut_sites    = [get_label(i) for i in nuc_sites]
    infer_link   = np.zeros(len(linked_sites))
    error_link   = np.zeros(len(linked_sites))
    counter      = np.zeros(len(linked_sites))
    link_labels  = []
    for i in range(len(linked_sites)):
        labels_temp = []
        for j in range(len(linked_sites[i])):
            if linked_sites[i][j] in mut_sites:
                infer_link[i] += inferred[mut_sites.index(linked_sites[i][j])]
                error_link[i] += errors[mut_sites.index(linked_sites[i][j])] ** 2
                labels_temp.append(mut_sites[mut_sites.index(linked_sites[i][j])])
                counter[i]    += 1
        link_labels.append(labels_temp)
    counter[counter==0]=1
    error_link /= counter 
    
    # getting rid of sites in the linked file that don't show up in the data
    infer_temp, error_temp, labels_temp = [], [], []
    for i in range(len(link_labels)):
        if len(link_labels[i])>1:
            infer_temp.append(infer_link[i])
            error_temp.append(error_link[i])
            labels_temp.append(link_labels[i])
    infer_link  = np.array(infer_temp)
    error_link  = np.array(error_temp)
    link_labels = np.array(labels_temp)
    
    proteins    = [[i[:i.find('-')]       for i in link_labels[j]] for j in range(len(link_labels))]
    aa_nums     = [[i[(i.find('-')+1):-2] for i in link_labels[j]] for j in range(len(link_labels))]
    locations   = []
    for i in range(len(link_labels)):
        temp_labels = []
        for j in range(len(link_labels[i])):
            if link_labels[i][j][:2]!='NC': temp_labels.append(link_labels[i][j][:-2])
            else:                           temp_labels.append(link_labels[i][j])
        locations.append(temp_labels)
    link_labels = locations
    
    ### TEST CODE ###
    initial_aas  = []
    final_aas    = []
    prot_labels  = []
    for line in open(protein_file).readlines():
        temp = line.split(',')
        initial_aas.append(temp[1][0])
        final_aas.append(temp[1][2])
        prot_labels.append(temp[0])
    aa_changes_full = [initial_aas[i] + prot_labels[i][prot_labels[i].find('-')+1:-2] + final_aas[i] for i in range(len(prot_labels))]
    aa_changes      = []
    for i in range(len(link_labels)):
        changes_temp = []
        for j in range(len(link_labels[i])):
            changes_temp.append(aa_changes_full[prot_labels.index(link_labels[i][j])])
        aa_changes.append(changes_temp)  
    locations = [[link_labels[:link_labels[i][j].find('-') + 1] + aa_changes[i][j] for j in range(len(aa_changes[i]))] for i in range(len(aa_changes))]
    ### END OF TEST CODE ###

    group_names = []
    for i in range(len(locations)):
        s_mutation = None
        for j in range(len(locations[i])):
            if locations[i][j][0]=='S':
                if locations[i][j]=="S-143":
                    s_mutation = "S-143"
                    break
                else:
                    s_mutation = locations[i][j]
        if s_mutation: group_names.append(s_mutation)
        else:          group_names.append(locations[i][0])
    
    prots   = [' '.join(i) for i in proteins]
    aa_idxs = [' '.join(i) for i in aa_nums]
    
    #arrays  = [proteins, aa_nums]
    #tuples  = list(zip(*arrays))
    
    # saving the data frame with each mutation on a different row
    dic = {'Variant Name' : group_names, 'Location': locations,  'Selection Coefficient' : infer_link, 'Error': error_link}
    df  = pd.DataFrame.from_dict(dic)
    df.explode('Location')
    df.to_latex(os.path.join(fig_path, out_file+'-alternate.html'), index=False, float_format="%.5f")
    
    # saving the data frame with the linked sites all listed together
    pd.set_option('display.max_colwidth', None)
    locations = [list(np.unique(i)) for i in locations]
    locs    = [(', '.join(i)).lower() for i in locations]
    inf     = ["%.3f" % (100*i) for i in infer_link]
    err     = ["%.3f" % (100*i) for i in error_link]
    inf_err = [f'{inf[i]}'+u"\u00B1"+f'{err[i]}' for i in range(len(inf))]
    locs    = pd.Series(locs, dtype="string")
    dic2    = {'Location': locs,  'Selection Coefficient' : inf_err}
    df2     = pd.DataFrame.from_dict(dic2)
    #df2.to_latex(os.path.join(fig_path, out_file+'.html'), index=False, float_format="%.5f", column_format='p{10cm}c', label='linked-table')
    latex = df2.to_latex(index=False, float_format="%.5f", column_format='p{10cm}r', label='table:linked-table')
    latex = latex.replace('\\\n', '\\ \hline\n')
    f     = open(os.path.join(fig_path, out_file+'.html'), mode='w')
    f.write(latex)
    f.close()
    df2.to_csv(os.path.join(fig_path, out_file+'.csv'))
    #print(latex)
    
    
def s_compare_regs(files=[], labels=[], out='regularization-comparison'):
    """ Given selection coefficients with different regularizations, make a scatter plot of the first coefficients with each of the others."""
    
    if len(files)==0:
        print('no files given')
    #gridspec_kw = {'hspace' : 0.02}
    #fig, axes = plt.subplots(1, len(files) - 1, figsize=[DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH / 1.618])
    left   = 0.075
    bottom = 0.2
    width  = 0.9
    height = 0.75
    fig    = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH / 1.618])
    grid   = matplotlib.gridspec.GridSpec(1, len(files) - 1, figure=fig, left=left, bottom=bottom, right = left + width, top = bottom + height)
    for i in range(len(files) - 1):
        ax = fig.add_subplot(grid[0, i])
        if i==0:
            #s_compare_scatter(files[i+1], files[0], ax=axes[i], label1=labels[i+1], label2=labels[0])
            s_compare_scatter(files[i+1], files[0], ax=ax, label1=labels[i+1], label2=labels[0])
        else:
            #s_compare_scatter(files[i+1], files[0], ax=axes[i], label1=labels[i+1])
            #axes[i].set_yticklabels([])
            s_compare_scatter(files[i+1], files[0], ax=ax, label1=labels[i+1])
            ax.set_yticklabels([])
            
    plt.savefig(os.path.join(fig_path, out+'.png'), dpi=2500)
    
    
def s_compare_regs_cutoff(files_reg=[], labels_reg=[], files_cutoff=[], labels_cutoff=[], out='regularization-comparison'):
    """ Given selection coefficients with different regularizations, make a scatter plot of the first coefficients with each of the others."""

    left     = 0.075
    bottom   = 0.1
    width    = 0.9
    height   = 0.4
    spacing  = 0.15
    fig      = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 1.2])
    low_grid = matplotlib.gridspec.GridSpec(1, len(files_reg) - 1, figure=fig, 
                                            left=left, bottom=bottom, right = left + width, 
                                            top = bottom + height)
    top_grid = matplotlib.gridspec.GridSpec(1, len(files_reg) - 1, figure=fig, 
                                            left=left, bottom=bottom + spacing + height, 
                                            right = left + width, top = bottom + 2*height + spacing)
    for i in range(len(files_reg) - 1):
        ax = fig.add_subplot(low_grid[0, i])
        if i==0:
            #s_compare_scatter(files[i+1], files[0], ax=axes[i], label1=labels[i+1], label2=labels[0])
            s_compare_scatter(files_reg[i+1], files_reg[0], ax=ax, label1=labels_reg[i+1], label2=labels_reg[0])
        else:
            #s_compare_scatter(files[i+1], files[0], ax=axes[i], label1=labels[i+1])
            #axes[i].set_yticklabels([])
            s_compare_scatter(files_reg[i+1], files_reg[0], ax=ax, label1=labels_reg[i+1])
            ax.set_yticklabels([])
            
    for i in range(len(files_cutoff) - 1):
        ax = fig.add_subplot(top_grid[0, i])
        if i==0:
            s_compare_scatter(files_cutoff[i+1], files_cutoff[0], ax=ax, label1=labels_cutoff[i+1], label2=labels_cutoff[0])
        else:
            s_compare_scatter(files_cutoff[i+1], files_cutoff[0], ax=ax, label1=labels_cutoff[i+1])
            ax.set_yticklabels([])
            
    plt.savefig(os.path.join(fig_path, out+'.png'), dpi=2500)
    
    
def s_compare_regs_cutoff_window(files_reg=[],    labels_reg=[], 
                                 files_cutoff=[], labels_cutoff=[], 
                                 files_wind=[],   labels_wind=[],
                                 out='regularization-comparison'):
    """ Given selection coefficients with different regularizations, make a scatter plot of the first coefficients with each of the others."""
    
    def min_max_s(files):
        """Find the smallest and largest selection coefficients in a list of files"""
        s = [np.load(file, allow_pickle=True)['selection'] for file in files]
        s_min = np.amin([np.amin(i) for i in s])
        s_max = np.amax([np.amax(i) for i in s])
        return s_min, s_max

    left     = 0.075
    bottom   = 0.1
    width    = 0.9
    height   = 0.225
    spacing  = 0.1
    fig      = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 1.618])
    low_grid = matplotlib.gridspec.GridSpec(1, len(files_reg) - 1, figure=fig, 
                                            left=left, bottom=bottom, right = left + width, 
                                            top = bottom + height)
    mid_grid = matplotlib.gridspec.GridSpec(1, len(files_reg) - 1, figure=fig, 
                                            left=left, bottom=bottom + spacing + height, 
                                            right = left + width, 
                                            top = bottom + 2*height + spacing)
    top_grid = matplotlib.gridspec.GridSpec(1, len(files_reg) - 1, figure=fig, 
                                            left=left, bottom=bottom + 2*spacing + 2*height, 
                                            right = left + width, 
                                            top = bottom + 3*height + 2*spacing)
    
    s_min, s_max = min_max_s(files_reg)
    for i in range(len(files_reg) - 1):
        ax = fig.add_subplot(low_grid[0, i])
        #ax.set_ylim(s_min * 100 - 1, s_max * 100 + 1)
        #ax.set_xlim(s_min * 100 - 1, s_max * 100 + 1)
        if i==0:
            s_compare_scatter(files_reg[i+1], files_reg[0], ax=ax, label1=labels_reg[i+1], label2=labels_reg[0])
        else:
            s_compare_scatter(files_reg[i+1], files_reg[0], ax=ax, label1=labels_reg[i+1])
            ax.set_yticklabels([])
    
    s_min, s_max = min_max_s(files_cutoff)
    for i in range(len(files_cutoff) - 1):
        ax = fig.add_subplot(mid_grid[0, i])
        #ax.set_ylim(s_min * 100 - 1, s_max * 100 + 1)
        #ax.set_xlim(s_min * 100 - 1, s_max * 100 + 1)
        if i==0:
            s_compare_scatter(files_cutoff[i+1], files_cutoff[0], ax=ax, label1=labels_cutoff[i+1], label2=labels_cutoff[0])
        else:
            s_compare_scatter(files_cutoff[i+1], files_cutoff[0], ax=ax, label1=labels_cutoff[i+1])
            ax.set_yticklabels([])
    
    s_min, s_max = min_max_s(files_wind)
    for i in range(len(files_wind) - 1):
        ax = fig.add_subplot(top_grid[0, i])
        #ax.set_ylim(s_min * 100 - 1, s_max * 100 + 1)
        #ax.set_xlim(s_min * 100 - 1, s_max * 100 + 1)
        if i==0:
            s_compare_scatter(files_wind[i+1], files_wind[0], ax=ax, label1=labels_wind[i+1], label2=labels_wind[0])
        else:
            s_compare_scatter(files_wind[i+1], files_wind[0], ax=ax, label1=labels_wind[i+1])
            ax.set_yticklabels([])
            
    fig.text(0.05,  0.98,  'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.24,  0.98,  'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.05,  0.68,  'c', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.24,  0.68,  'd', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.05,  0.35,  'e', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.24,  0.35,  'f', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.43,  0.35,  'g', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.62,  0.35,  'h', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.81,  0.35,  'i', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
            
    plt.savefig(os.path.join(fig_path, out+'.png'), dpi=800)
    
    
    
def s_compare_scatter(file1, file2, ax=None, label1=None, label2=None, out=None):
    """ Given two files, scatter the inferred coefficients against one another and calculate the R^2 value."""
    data1    = np.load(file1, allow_pickle=True)
    s1       = data1['selection']
    muts1    = data1['allele_number']
    
    data2    = np.load(file2, allow_pickle=True)
    s2       = data2['selection']
    muts2    = data2['allele_number']
    
    muts_all = list(set.intersection(*map(set, [muts1, muts2])))
    mask1    = np.isin(muts1, muts_all)
    muts1    = muts1[mask1]
    s1       = s1[mask1]
    mask2    = np.isin(muts2, muts_all)
    muts2    = muts2[mask2]
    s2       = s2[mask2]
    assert((muts1==muts2).all())
    
    if not isinstance(label1, type(None)):
        ax.set_xlabel(label1, fontsize=AXES_FONTSIZE)
    if not isinstance(label2, type(None)):
        ax.set_ylabel(label2, fontsize=AXES_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE, length=2, width=SPINE_LW)
    for line in ['right', 'top']:
        ax.spines[line].set_visible(False)
    for line in ['bottom', 'left']:
        ax.spines[line].set_linewidth(SPINE_LW)
    ax.scatter(s1 * 100, s2 * 100, marker='o', s=0.15, color='k', alpha=1, edgecolors='k')
    #ax.scatter(s1 * 100, s2 * 100, marker=',', color='k', alpha=1, edgecolors=None)
    Rsquared = spstats.pearsonr(s1, s2)[0] ** 2
    ax.text(0.1, 0.75, '$R^2$ = %.3f' % Rsquared, fontsize=AXES_FONTSIZE, transform=ax.transAxes)
    
    max_s = max(np.amax(s1), np.amax(s2))
    min_s = min(np.amin(s1), np.amin(s2))
    xy_line = np.linspace(min_s * 100 - 1, max_s * 100 + 1, 100)
    ax.plot(xy_line, xy_line, linewidth=SIZELINE, markersize=0, scalex=False, scaley=False, alpha=0.5)
    
    min_y = np.amin(s2)
    max_y = np.amax(s2)
    ax.set_ylim(min_y * 100 - 1, max_y * 100 + 1)
    
    yinterval = 3
    while (max_y - min_y) * 100 / yinterval > 6:
        yinterval += 1
    ax.yaxis.set_major_locator(plt.MultipleLocator(yinterval))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(yinterval))
    
    xinterval = 3
    while (max_s - min_s) * 100 / xinterval > 6:
        xinterval += 1
    ax.xaxis.set_major_locator(plt.MultipleLocator(xinterval))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(xinterval))
    
    
    
    
def compare_selection(files, sort_file_idx=-1, labels=None, log=False, sort_individual=False, sort=False, out=None, s_cutoff=None):
    """ Given a list containing files that contain inferred selection coefficients for various parameters, plot their comparison.
    sort_file_idx the index of the file in files that will be used to sort the selection coefficients"""
    if sort_file_idx == -1:
        sort_file_idx = len(files) - 1
    colors    = sns.husl_palette(len(files))
    selection = []
    alleles   = []
    for file in files:
        data = np.load(file, allow_pickle=True)
        selection.append(data['selection'])
        alleles.append(data['allele_number'])
    alleles_all = list(set.intersection(*map(set, alleles)))
    print(len(alleles_all))
    
    masks       = [np.isin(alleles[i], alleles_all) for i in range(len(alleles))]
    selection   = [selection[i][masks[i]] for i in range(len(alleles))]
    alleles     = [alleles[i][masks[i]] for i in range(len(alleles))]
    if sort:
        if not sort_individual:
            sorter      = np.argsort(np.absolute(selection[sort_file_idx]))
            if not isinstance(s_cutoff, type(None)):
                sorter  = sorter[np.absolute(selection[sort_file_idx][sorter])>s_cutoff]
            alleles_all = np.array(alleles_all)[sorter]
            if not log:
                selection = [selection[i][sorter] for i in range(len(selection))]
            else:
                selection = [np.absolute(selection[i][sorter]) for i in range(len(selection))]
        else:
            #selection   = [i[np.argsort(np.absolute(i))] for i in selection]
            selection = [np.sort(np.absolute(i)) for i in selection]
            if not isinstance(s_cutoff, type(None)):
                selection = [i[i>s_cutoff] for i in selection]
    
    ## eliminate most sites that dont appear in the data
    #selection   = [i[8000:] for i in selection]
    
    pearsonr    = []
    for i in range(len(selection)):
        pearson_temp = []
        for j in range(len(selection)):
            pearson_temp.append(spstats.pearsonr(selection[i], selection[j])[0] ** 2)
        pearsonr.append(pearson_temp)
    #r_squared   = np.array(pearsonr) ** 2 
    
    #for i in range(len(pearsonr)):
        #print(pearsonr[i])
    
    df = pd.DataFrame(data=pearsonr, columns=labels, index=labels)
    print(df)
    
    fig, ax = plt.subplots(1,1, figsize=[20,15])
    axes    = []
    for i in range(len(selection)):
        if i != 0: 
            ax_new = ax.twinx()
        else:      
            ax_new  = ax
        if isinstance(labels, type(None)):
            line   = ax_new.plot(np.arange(len(selection[i])), selection[i], lw=0, marker='.', color=colors[i], alpha=0.5)
        else:
            ax_new.plot(np.arange(len(selection[i])), selection[i], label=labels[i], lw=0, marker='.', color=colors[i], alpha=0.5)
        max_s = np.amax(np.absolute(selection[i]))
        if not log:
            ax_new.set_ylim(-1.01 * max_s, 1.01 * max_s)
        else:
            ax_new.set_yscale('log')
            if log:
                if i==0:
                    ylimits = ax_new.get_ylim()
                else:
                    ax_new.set_ylim(ylimits)
        axes.append(ax_new)
        
    if not isinstance(labels, type(None)):
        fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
        #labs = [l.get_label() for l in lines] 
        #ax.legend(lines, labs)
    for i in range(len(axes)):
        axes[i].set_ylabel(labels[i])
        
    if not isinstance(out, type(None)):
        fig.savefig(os.path.join(fig_path, out + '.png'), dpi=1200)
        
    plt.show()
    
    
    
def selection_time_plot(infer_file, syn_file, link_file, n_coefficients=20, out_file='inferred-tv'):
    """ Plots the way the selection coefficients change as more data is usednext to their trajectories in the various regions."""
    
    def traj_mean(traj, times, muts):
        """ Given the trajectories in a continent, finds and returns the average trajectory, 
        new times for the average trajectory, and the name of all of the mutant sites"""
        muts_all = np.unique([muts[i][j] for i in range(len(muts)) for j in range(len(muts[i]))])
        traj_all, times_all = [], []    # average frequency trajectories and times for each mutation
        for i in range(len(muts_all)):
            # Determine which subregions have the mutation and the times that it shows up in each
            traj_spec, times_spec = [], []
            for j in range(len(muts)):
                if muts_all[i] in muts[j]:
                    traj_spec.append(traj[j][:, list(muts[j]).index(muts_all[i])])
                    times_spec.append(times[j])
            t_min    = np.amin([np.amin(t) for t in times_spec])
            t_max    = np.amax([np.amax(t) for t in times_spec])
            mut_time = np.arange(t_min, t_max+1)
            traj_av  = np.zeros(len(mut_time))    # the averaged trajectory
            times_all.append(mut_time)
            # Average the trajectories at the times where multiple show up
            for j in range(len(mut_time)):
                norm = max(np.sum([mut_time[j] in times_spec[k] for k in range(len(times_spec))]), 1)
                for k in range(len(traj_spec)):
                    if mut_time[j] in times_spec[k]:
                        traj_av[j] += traj_spec[k][list(times_spec[k]).index(mut_time[j])]
                traj_av[j] /= norm
                if np.sum([mut_time[j] in times_spec[k] for k in range(len(times_spec))])==0:
                    traj_av[j] = np.nan
            traj_all.append(traj_av)
        return traj_all, times_all, muts_all
        
    def filter_linked(traj, times, muts, linked):
        """ Given the linked sites, and the trajectories, times, and mutant sites in a continent,
        filters out and returns the ones that correspond to the linked sites."""
        new_traj, new_times, new_muts = [], [], []
        for i in range(len(linked)):
            if linked[i][0] in list(muts):
                # CHANGE THIS SO IT DOESN'T ONLY USE THE FIRST IN THE LIST OF THE LINKED SITES
                new_traj.append(traj[list(muts).index(linked[i][0])])
                new_times.append(times[list(muts).index(linked[i][0])])
                new_muts.append(linked[i][0])
        return new_traj, new_times, new_muts
    
    def traj_mean_alt(times_link, traj_link):
        """ Given the linked trajectories and times in a specific continent find the averaged trajectories"""
        new_traj, new_times = [], []
        for i in range(len(traj_link)):
            if len(times_link[i])>0:
                t_min  = np.amin([np.amin(t) for t in times_link[i]])
                t_max  = np.amax([np.amax(t) for t in times_link[i]])
                t_temp = np.arange(t_min, t_max+1)
                traj_temp = np.zeros(len(t_temp))
                for j in range(len(t_temp)):
                    norm = max(np.sum([t_temp[j] in times_link[i][k] for k in range(len(times_link[i]))]), 1)
                    for k in range(len(traj_link[i])):
                        if t_temp[j] in times_link[i][k]:
                            traj_temp[j] += traj_link[i][k][list(times_link[i][k]).index(t_temp[j])]
                    traj_temp[j] /= norm
                    if np.sum([t_temp[j] in times_link[i][k] for k in range(len(times_link[i]))])==0:
                        traj_temp[j] = np.nan
                new_times.append(t_temp)
                new_traj.append(traj_temp)
            else:
                new_times.append([])
                new_traj.append([])
        return new_times, new_traj
    
    def find_linked_coefficients(linked_sites, labels, inferred, error, n_sims, indices):
        """ Finds the sum of the coefficients for linked sites"""
        inferred_link = np.zeros(len(linked_sites))
        error_link    = np.zeros(len(linked_sites))
        for i in range(len(linked_sites)):
            for j in range(len(linked_sites[i])):
                if linked_sites[i][j] in list(labels):
                    loc = list(labels).index(linked_sites[i][j])
                    inferred_link[i] += inferred[loc]
                    error_link[i]    += error[loc]
        error_link = np.sqrt(error_link)
        ### Since the labels have already been ordered, maybe don't reapply the indices
        #error_link    = np.array(error_link)[indices]
        #inferred_link = inferred_link[indices]
        return inferred_link, error_link                                
    
    inf_data = np.load(infer_file, allow_pickle=True)   # Data transfered over from inference script
    traj     = inf_data['traj']                         # Frequency trajectories for each region and each allele
    inferred = inf_data['selection']                    # Selection coefficients from inference script
    error    = inf_data['error_bars']                   # Error bars for selection coeffiicents
    times    = inf_data['times']                        # The times at which the genomes were collected (with January 1 as a starting point)
    mutant_sites  = inf_data['mutant_sites']            # The genome positions at which there are mutations in each region
    allele_number = inf_data['allele_number']           # The genome positions at which there are mutations across all regions
    locations     = inf_data['locations']               # The labels for the locations (and the dates) from which sequences were drawn
    inferred_ind  = inf_data['selection_independent']   # The inferred coefficients ignoring linkage between sites (i.e., ignoring the covariances)
    
    labels = []        # mutation locations across all populations
    for site in allele_number:
        labels.append(get_label(site))
    
    full_labels = []       # mutation locations for each population
    for i in range(len(mutant_sites)):
        labels_temp = []
        for j in range(len(mutant_sites[i])):
            loc = (mutant_sites[i][j] == allele_number).nonzero()[0][0]
            labels_temp.append(labels[loc])
        full_labels.append(np.array(labels_temp))
    
    continents    = np.array([i.split('-')[0] for i in locations])     # The continents
    locations     = ['-'.join(i.split('-')[1:]) for i in locations]    # The locations without the continent labels
    
    max_time = np.amax(np.array([np.amax(i) for i in times]))
    min_time = np.amin(np.array([np.amin(i) for i in times]))
    total_days  = max_time - min_time + 1
    simulations = len(traj)
    
    type_data = np.load(syn_file, allow_pickle=True)
    mutant_types_temp  = type_data['types']
    sites = type_data['locations']
    mutant_types = []
    for i in allele_number:
        if i in sites:
            mutant_types.append(mutant_types_temp[list(sites).index(i)])
        else:
            mutant_types.append('unknown')
    
    # Finding location labels and times
    locations_short = []         # just the locations without the times included
    time_stamps     = []         # just the timestamps without the location
    for i in range(len(locations)):
        if locations[i][:3] == 'usa':
            locations_short.append(locations[i][:locations[i].find('-', 4)])
        elif locations[i][:14] == 'united kingdom':
            locations_short.append(locations[i][:locations[i].find('-', 15)])
        else:
            locations_short.append(locations[i][:locations[i].find('-')])
        time_stamps.append(locations[i][-11:])
        
    # Finding the absolute times corresponding to each trajectory
    real_times = []
    for loc in locations:
        time_start = dt.date.fromisoformat('2020-' + loc[-11:-6]) - dt.date(2020, 1, 1)
        time_end   = dt.date.fromisoformat('2020-' + loc[-5:])    - dt.date(2020, 1, 1)
        real_times.append(np.arange(time_start.days, time_end.days+1))
    
    colormap    = cm.get_cmap('Dark2', 4)
    greys       = cm.get_cmap('Greys', 4)
    colors_main = [colormap(0), colormap(1), 'k']
    colors_all  = [colormap(0), colormap(1)] + [greys(i) for i in range(4)]

    # Process linked data and order coefficients 
    linked_sites  = np.load(link_file, allow_pickle=True)
    inferred_link = np.zeros(len(linked_sites))    # the sum of the inferred coefficients for the linked sites
    error_link    = np.zeros(len(linked_sites))    # the error for the inferred summed coefficients
    error_ind     = np.zeros(len(linked_sites))    # the error for the coefficients ignorning linkage
    counter       = np.zeros(len(linked_sites))    # counts the number of linked sites in each group
    traj_link, link_labels, label_types, traj_locs, traj_times = [], [], [], [], []
    traj_cont = []
    for i in range(len(linked_sites)):
        traj_temp, labels_temp, types_temp, traj_locs_temp, traj_times_temp = [], [], [], [], []
        traj_cont_temp = []
        for j in range(simulations):
            if np.any(linked_sites[i][0]==np.array(full_labels[j])):
                loc2 = np.where(linked_sites[i][0]==np.array(full_labels[j]))[0][0]
                traj_temp.append(traj[j][:, loc2])
                traj_locs_temp.append(locations[j])
                traj_times_temp.append(times[j])
                traj_cont_temp.append(continents[j])
        for j in range(len(linked_sites[i])):
            if np.any(linked_sites[i][j]==np.array(labels)):
                loc = np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]
                inferred_link[i] += inferred[-1][loc]
                error_link[i]    += error[-1][loc] ** 2
                counter[i]       += 1
                labels_temp.append(labels[loc])
                types_temp.append(mutant_types[loc])
        #link_labels.append(' '.join(labels_temp))
        link_labels.append(labels_temp)
        traj_link.append(traj_temp)
        label_types.append(types_temp)
        traj_locs.append(traj_locs_temp)
        traj_times.append(traj_times_temp)
        traj_cont.append(traj_cont_temp)
    counter[counter==0] = 1
    #error_link = np.sqrt(error_link) / np.sqrt(counter)
    error_link = np.sqrt(error_link)
    
    
    """
    av_link_e, av_times_e, muts_e = filter_linked(traj_av_e, times_av_e, muts_av_e, linked_sites)   # the average trajectories and times for the linked groups based on continents
    av_link_a, av_times_a, muts_a = filter_linked(traj_av_a, times_av_a, muts_av_a, linked_sites)
    av_link_o, av_times_o, muts_o = filter_linked(traj_av_o, times_av_o, muts_av_o, linked_sites)
    #print(av_link_e, traj_av_e)
    """
        
    # Filter out linked sites groups that are not found in the data.
    label_lengths = [len(link_labels[i]) for i in range(len(link_labels))]
    inferred_link = inferred_link[np.array(label_lengths).nonzero()[0]]
    indices   = np.argsort(np.absolute(inferred_link))[::-1]
    link_labels   = np.array(link_labels)[np.array(label_lengths).nonzero()[0]][indices]
    traj_link     = np.array(traj_link)[np.array(label_lengths).nonzero()[0]][indices]
    label_types   = np.array(label_types)[np.array(label_lengths).nonzero()[0]][indices]
    traj_locs     = np.array(traj_locs)[np.array(label_lengths).nonzero()[0]][indices]
    error_link    = np.array(error_link)[np.array(label_lengths).nonzero()[0]][indices]
    traj_times    = np.array(traj_times)[np.array(label_lengths).nonzero()[0]][indices]
    traj_cont     = np.array(traj_cont)[np.array(label_lengths).nonzero()[0]][indices]
    error_ind     = np.array(error_ind)[np.array(label_lengths).nonzero()[0]][indices]
    inferred_link = inferred_link[indices]
    label_lengths = np.array([len(link_labels[i]) for i in range(len(link_labels))])
    
    # separate data into large regions and find average trajectories
    cont_filtered = []
    for i in range(len(traj_cont)):
        cont_temp = []
        for cont in traj_cont[i]:
            if cont!='europe' and cont!='north america':
                cont_temp.append('other')
            else:
                cont_temp.append(cont)
        cont_filtered.append(cont_temp)
    cont_all  = ['north america', 'europe', 'oceania', 'asia', 'africa', 'south america']
    cont_main = ['north america', 'europe', 'other']    # regions in which frequency trajectories will be averaged
    
    traj_e    = [np.array(traj_link[i])[np.array(cont_filtered[i])=='europe'] for i in range(len(traj_link))]
    times_e   = [np.array(traj_times[i])[np.array(cont_filtered[i])=='europe'] for i in range(len(traj_times))]
    traj_a    = [np.array(traj_link[i])[np.array(cont_filtered[i])=='north america'] for i in range(len(traj_link))]
    times_a   = [np.array(traj_times[i])[np.array(cont_filtered[i])=='north america'] for i in range(len(traj_times))]
    traj_o    = [np.array(traj_link[i])[np.array(cont_filtered[i])=='other'] for i in range(len(traj_link))]
    times_o   = [np.array(traj_times[i])[np.array(cont_filtered[i])=='other'] for i in range(len(traj_times))]
    times_e, traj_e = traj_mean_alt(times_e, traj_e)
    times_a, traj_a = traj_mean_alt(times_a, traj_a)
    times_o, traj_o = traj_mean_alt(times_o, traj_o)
    
    # Make the figure and selection plot
    #tick_locations = [2*i + 2*j / (label_lengths[::-1][i] + 1) for i in range(len(link_labels)) for j in range(len(link_labels[::-1][i]))]
    fig_len   = DOUBLE_COLUMN_WIDTH * 10
    fig_width = DOUBLE_COLUMN_WIDTH * 1.5
    s_max   = np.amax(inferred_link) + 0.02
    fig     = plt.figure(figsize=[fig_width, fig_len])
    grid    = matplotlib.gridspec.GridSpec(len(link_labels), 2, wspace=0.1, hspace=0.05, figure=fig)
    
    # Find linked selection coefficients for each time
    inferred_full = np.zeros((len(inferred), len(inferred_link)))
    for i in range(len(inferred)):
        inferred_link, error_link = find_linked_coefficients(link_labels, labels, inferred[i], error[i], simulations, indices)
        inferred_full[i] = inferred_link
    
    # Plot the inferred coefficients as a function of time
    right_grid = grid[:, 1].subgridspec(len(inferred_link), 1, wspace=0)
    t          = np.arange(len(inferred_full))
    for i in range(len(inferred_link)):
        ax_s   = fig.add_subplot(right_grid[i, 0])
        ax_s.set_xticks([])
        ax_s.set_ylim(100*(np.amin(inferred_full[:, i])-0.01), 100*(np.amax(inferred_full[:, i])+0.01))
        ax_s.tick_params(labelsize=TICK_FONTSIZE, length=2, width=SPINE_LW)
        ax_s.plot(t, 100*inferred_full[:, i])
    
    """
    ax_s    = fig.add_subplot(grid[:, 1])
    ax_s.set_xlabel("Selection coefficients (%)", fontsize=8)
    ax_s.grid(b=True, axis='x', linewidth=0.3)
    ax_s.set_xlim(-s_max*100, s_max*100)
    ax_s.set_ylim(0, len(inferred_link))
    ax_s.set_yticks([])
    ax_s.tick_params(labelsize=8, length=2, width=0.3)
    #sns.scatterplot(100*inferred_link[::-1], np.arange(len(inferred_link))+0.5, label="sum of linked sites", ax=ax_s, size=0.5, legend='brief', color='dimgrey')
    color_map = cm.get_cmap('Blues')
    colors    = [color_map(i/len(inferred)) for i in range(len(inferred))]
    for i in range(len(inferred)):
        inferred_link, error_link = find_linked_coefficients(link_labels, labels, inferred[i], error[i], simulations, indices)
        ax_s.plot(100*inferred_link[::-1], np.arange(len(inferred_link))+0.5, color=colors[i], markersize=3, linewidth=0, marker='.')
        
    for line in ['right', 'left', 'top', 'bottom']:
        ax_s.spines[line].set_linewidth(0.3)
    ax_s.legend()
    """
    
    
    # Create labels for months for the trajectory plots
    num_months   = int(total_days / 30) + 1
    months_start = [dt.date(2020, i, 1) for i in range(2, 13)] + [dt.date(2021, i, 1) for i in range(1, 13)]
    month_locs   = [(months_start[i] - dt.date(2020, 1, min_time)).days for i in range(len(months_start)) if i<=num_months]        # x tick locations
    months       = ['Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb'][:num_months]  # x tick labels
    
    # Plot the trajectories
    for i in range(len(inferred_link)):
        ax = fig.add_subplot(grid[i, 0])
        if i == len(inferred_link)-1:
            ax.set_xlabel("Frequency trajectories", fontsize=AXES_FONTSIZE, labelpad=10)
            ax.set_xticks(month_locs)
            ax.set_xticklabels(months)
        else:
            ax.set_xticks([])
        
        ax.set_xlim(min_time, max_time)
        ax.set_ylim(0,1)
        if label_lengths[i] <= 8:
            tick_locations = [(j+1/2)/label_lengths[i] for j in range(label_lengths[i])]
            tick_labels    = link_labels[i]
        else:
            tick_locations = [(j+1/2)/8 for j in range(8)]
            tick_labels    = link_labels[i][:8]
            label_types[i] = label_types[i][:8]
        ax.set_yticks(tick_locations)
        ax.set_yticklabels(tick_labels)
        ax.tick_params(labelsize=TICK_FONTSIZE, length=2, width=SPINE_LW)
        for j in range(len(label_types[i])):
            if label_types[i][j] == 'S':
                plt.gca().get_yticklabels()[j].set_color("lightgrey")
            elif label_types[i][j] == 'NNS':
                plt.gca().get_yticklabels()[j].set_color("grey")
        
        for k in range(len(traj_link[i])):
            ax.plot(traj_times[i][k], traj_link[i][k], color=colors_all[cont_all.index(traj_cont[i][k])], alpha=0.5, lw=0.5)
        
        """
        if link_labels[i][0] in muts_e:
            ax.plot(av_times_e[muts_e.index(link_labels[i][0])], av_link_e[muts_e.index(link_labels[i][0])], color=colors_main[0], lw=0.5)
        if link_labels[i][0] in muts_a:
            ax.plot(av_times_a[muts_a.index(link_labels[i][0])], av_link_a[muts_a.index(link_labels[i][0])], color=colors_main[1], lw=0.5)
        if link_labels[i][0] in muts_o:
            ax.plot(av_times_o[muts_o.index(link_labels[i][0])], av_link_o[muts_o.index(link_labels[i][0])], color=colors_main[2], lw=0.5)
        """
        ax.plot(times_e[i], traj_e[i], color=colors_main[1], lw=1, label='Europe')
        ax.plot(times_a[i], traj_a[i], color=colors_main[0], lw=1, label='North America')
        ax.plot(times_o[i], traj_o[i], color=colors_main[2], lw=1, label='Other')
        for line in ['right', 'left', 'top', 'bottom']:
            ax.spines[line].set_linewidth(SPINE_LW)
        if i == len(inferred_link) - 1:
            ax.legend(loc='upper right', fontsize=8, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.05, top=0.99)
    plt.savefig(os.path.join(fig_path, out_file+'.png'), dpi=1200)



def calculate_AUROC(actual, inferred, ben=True, deli=True):
    """ Calculates the AUROC for distinguishing beneficial mutations and for distinguishing deleterious mutations. Replaces the below.
    (Check why it produces slightly different answers from the below function.)"""
    
    ben_score, del_score = [], []
    for i in range(len(inferred)):
        act = np.array(actual[i])
        inf = np.array(inferred[i])
        i_ben = list(inf[act>0])
        i_del = list(inf[act<0])
        i_neu = list(inf[act==0])
        ben_score.append(np.sum([1 for i in i_ben for j in (i_del+i_neu) if i>j]) / (len(i_ben) * (len(i_del) + len(i_neu))))
        del_score.append(np.sum([1 for i in i_del for j in (i_ben+i_neu) if i<j]) / (len(i_del) * (len(i_ben) + len(i_neu))))
    if ben and deli: return np.average(ben_score), np.average(del_score)
    elif ben:        return np.average(ben_score)
    elif deli:       return np.average(del_score)

"""
def calculate_AUROC(actual, inferred):
    
    actual = np.array(actual)
    i_ben = list(inferred[actual>0])
    i_del = list(inferred[actual<0])
    i_neu = list(inferred[actual==0])
    benefit_score = np.sum([1 for i in i_ben for j in (i_del+i_neu) if i>j]) / (len(i_ben) * (len(i_del) + len(i_neu)))
    delet_score   = np.sum([1 for i in i_del for j in (i_ben+i_neu) if i<j]) / (len(i_del) * (len(i_ben) + len(i_neu)))
    return benefit_score, delet_score
"""


def calculate_AUROC_av(actual, inferred):
    """ Calculates the average AUROC over the replicate simulations (the above calculates the AUROC considering all of them together, 
    which is not correct)"""
    
    i_ben = [inferred[i][np.array(actual[i])>0] for i in range(len(actual))]
    i_del = [inferred[i][np.array(actual[i])<0] for i in range(len(actual))]
    i_neu = [inferred[i][np.array(actual[i])==0] for i in range(len(actual))]
    benefit_score = np.average([np.sum([1 for i in i_ben[k] for j in (list(i_del[k])+list(i_neu[k])) if i>j])/(len(i_ben[k]) * (len(i_del[k]) + len(i_neu[k])))
                                for k in range(len(i_ben))])
    delet_score   = np.average([np.sum([1 for i in i_del[k] for j in (list(i_ben[k])+list(i_neu[k])) if i<j])/(len(i_del[k]) * (len(i_ben[k]) + len(i_neu[k])))
                                for k in range(len(i_del))])
    return benefit_score, delet_score


def plot_AUROC_sampling(folder, ben_plot=True, del_plot=True):
    """ Plots the AUROC score vs. the number of samples per generation for a simulation using the actual N, and one using a constant N.
    Assumes that the file for the one using the true N ends in -tv.npz and for the constant N ends in -const.npz and that the files"""
    
    tv_inf, tv_act, const_inf, const_act = [], [], [], []
    tv_file, const_file = [], []
    for file in sorted(os.listdir(os.fsencode(folder))):
        filename = os.fsdecode(file)
        filepath = os.path.join(folder, filename)
        data     = np.load(filepath, allow_pickle=True)
        if filename[-6:] == 'tv.npz':
            tv_inf.append(data['inferred'])
            tv_act.append(data['actual'])
            tv_file.append(filename)
        elif filename[-9:] == 'const.npz':
            const_inf.append(data['inferred'])
            const_act.append(data['actual'])
            const_file.append(filename)
        #else:
        #    print('unknown file:\t', filename)
    
    samples_tv = []
    for b in tv_file:
        ind1 = b.find('sample')+6
        ind2 = b.find('-', ind1)
        samples_tv.append(int(b[ind1:ind2]))
    indices_tv = np.argsort(samples_tv)
    #tv_inf     = np.array(tv_inf)[indices_tv]
    #tv_act     = np.array(tv_act)[indices_tv]
    samples_const = []
    for b in const_file:
        ind1 = b.find('sample')+6
        ind2 = b.find('-', ind1)
        samples_const.append(int(b[ind1:ind2]))
    indices_const = np.argsort(samples_const)
    #const_inf     = np.array(const_inf)[indices_const]
    #const_act     = np.array(const_act)[indices_const]
    AUC_ben_tv    = np.array([calculate_AUROC(tv_act[i], tv_inf[i], deli=False) for i in range(len(tv_act))])[indices_tv]
    AUC_ben_const = np.array([calculate_AUROC(const_act[i], const_inf[i], deli=False) for i in range(len(const_act))])[indices_const]
    AUC_del_tv    = np.array([calculate_AUROC(tv_act[i], tv_inf[i], ben=False) for i in range(len(tv_act))])[indices_tv]
    AUC_del_const = np.array([calculate_AUROC(const_act[i], const_inf[i], ben=False) for i in range(len(const_act))])[indices_const]
    minimum       = min(list(AUC_ben_tv) + list(AUC_del_tv) + list(AUC_ben_const) + list(AUC_del_const))
    fig = plt.figure(figsize=[18,8])
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_ylim(minimum,1)
    ax1.set_xlabel('AUROC Beneficial')
    #sns.lineplot(np.sort(samples_tv), AUC_ben_tv, label='time-varying', ax=ax1, log=True)
    #sns.lineplot(np.sort(samples_const), AUC_ben_const, label='constant', ax=ax1, log=True)
    ax1.semilogx(np.sort(samples_tv), AUC_ben_tv, color='red', label='time-varying')
    ax1.semilogx(np.sort(samples_const), AUC_ben_const, color='blue', label='constant')
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_ylim(minimum,1)
    ax2.set_xlabel('AUROC Deleterious')
    ax2.semilogx(np.sort(samples_tv), AUC_del_tv, color='red', label='time-varying')
    ax2.semilogx(np.sort(samples_const), AUC_del_const, color='blue', label='constant')
    #sns.lineplot(np.sort(samples_tv), AUC_del_tv, label='time-varying', ax=ax2, log=True)
    #sns.lineplot(np.sort(samples_const), AUC_del_const, label='constant', ax=ax2, log=True)
    plt.legend()
            

def plot_histogram(file, index, cutoff, n_subplots):
    """Given a file containing the results of many repeated simulations, plot the histogram of inferred coefficients.
    index is the index number of the subplot"""
    # plot histograms            
    rep_data = np.load(file, allow_pickle=True)
    benefit_score, delet_score = [], []
    positive, negative, zero = [], [], []
    for sim in range(len(rep_data['inferred'])):
        traj = trajectory_reshape(rep_data['traj'][sim])
        inferred = rep_data['inferred'][sim]
        actual = rep_data['actual'][sim]
        mutants = rep_data['mutant_sites'][sim]
        error = rep_data['errors'][sim]
        b_score, d_score = calculate_AUROC(actual, inferred)
        benefit_score.append(b_score)
        delet_score.append(d_score)
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
        actual = [selection_new[i] for i in allele_number]
        mutant_sites_all = np.sort(np.unique(np.array([mutants[i][j] for i in range(len(mutants))
                                               for j in range(len(mutants[i]))])))
        inferred = filter_sites_infer(traj, inferred, mutants, mutant_sites_all, allele_number)   
        inf_minus, inf_zero, inf_plus = sort_by_sign(selection_new, inferred)
        positive += inf_plus
        negative += inf_minus
        zero += inf_zero
    s_value = np.amax(np.absolute(rep_data['actual'][0]))
    hist_pos  = {"weights" : np.ones(len(positive)) / len(positive)}
    hist_neg  = {"weights" : np.ones(len(negative)) / len(negative)}
    hist_zero = {"weights" : np.ones(len(zero)) / len(zero)}
    #plt.subplot(grid[:, 2])
    #plt.figure(figsize=[cm_to_inch(8.8), 8])
    #plt.title('mean AUROC ben/del: \n' + str(np.average(benefit_score)) + '/' +  str(np.average(delet_score)))
    plt.xlabel('mean AUROC ben/del: \n' + str(np.average(benefit_score)) + '/' +  str(np.average(delet_score)))
    plt.subplot(n_subplots,1,index)
    plt.axvline(x=-s_value, color='r')
    plt.axvline(x=0, color=mcolors.CSS4_COLORS['grey'])
    plt.axvline(x=s_value, color='b')
    if index!=n_subplots:
        plt.xticks([])
    plt.ylim(0, 0.1)
    sns.distplot(positive, kde=False, bins=50, color="b", label="beneficial", hist_kws=hist_pos)
    sns.distplot(zero, kde=False, bins=50, color=mcolors.CSS4_COLORS['grey'], label="neutral", hist_kws=hist_zero)
    sns.distplot(negative, kde=False, bins=50, color='r', label="deleterious", hist_kws=hist_neg)
    plt.legend()
    
    
def plot_histogram_object_oriented(file, index, cutoff, n_subplots, ax, title=None):
    """Given a file containing the results of many repeated simulations, plot the histogram of inferred coefficients.
    index is the index number of the subplot"""
    # plot histograms            
    rep_data = np.load(file, allow_pickle=True)
    benefit_score, delet_score = [], []
    #benefit_score, delet_score = calculate_AUROC_av(rep_data['actual'], rep_data['inferred'])
    positive, negative, zero = [], [], []
    for sim in range(len(rep_data['inferred'])):
        traj = trajectory_reshape(rep_data['traj'][sim])
        inferred = rep_data['inferred'][sim]
        actual = rep_data['actual'][sim]
        mutants = rep_data['mutant_sites'][sim]
        error = rep_data['errors'][sim]
        b_score, d_score = calculate_AUROC(actual, inferred)
        benefit_score.append(b_score)
        delet_score.append(d_score)
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
        actual = [selection_new[i] for i in allele_number]
        mutant_sites_all = np.sort(np.unique(np.array([mutants[i][j] for i in range(len(mutants))
                                               for j in range(len(mutants[i]))])))
        inferred = filter_sites_infer(traj, inferred, mutants, mutant_sites_all, allele_number)   
        inf_minus, inf_zero, inf_plus = sort_by_sign(selection_new, inferred)
        positive += inf_plus
        negative += inf_minus
        zero += inf_zero
    s_value = np.amax(np.absolute(rep_data['actual'][0]))
    transparancy = 0.2
    hist_pos  = {"weights" : np.ones(len(positive)) / len(positive), "alpha" : transparancy}
    hist_neg  = {"weights" : np.ones(len(negative)) / len(negative), "alpha" : transparancy}
    hist_zero = {"weights" : np.ones(len(zero)) / len(zero), "alpha" : transparancy}
    #plt.subplot(grid[:, 2])
    #plt.figure(figsize=[cm_to_inch(8.8), 8])
    #plt.title('mean AUROC ben/del: \n' + str(np.average(benefit_score)) + '/' +  str(np.average(delet_score)))
    #ax.set_xlabel('mean AUROC ben/del: \n' + str(np.average(benefit_score)) + '/' +  str(np.average(delet_score)))
    ax.axvline(x=-s_value, color='r')
    ax.axvline(x=0, color=mcolors.CSS4_COLORS['grey'])
    ax.axvline(x=s_value, color='b')
    ax.set_ylim(0, 0.1)
    ax.set_yticks([0, 0.1])
    if title:
        ax.set_title(title)
    sns.distplot(positive, kde=False, bins=50, color="b", label="beneficial", hist_kws=hist_pos, ax=ax)
    sns.distplot(zero, kde=False, bins=50, color=mcolors.CSS4_COLORS['grey'], label="neutral", hist_kws=hist_zero, ax=ax)
    sns.distplot(negative, kde=False, bins=50, color='r', label="deleterious", hist_kws=hist_neg, ax=ax)
    #ax.text(1, 0.9, 'AUROC ben/del: \n' + str(round(np.average(benefit_score),3)) + '/' +  str(round(np.average(delet_score),3)),
    #        transform=ax.transAxes, ha='right', va='top')
    ax.text(0.8, 0.9, 'AUROC \n' + str(round(np.average(benefit_score),3)),
            transform=ax.transAxes, ha='center', va='top')
    
    
def number_nonsynonymous_nonlinked(inferred, num_large, linked, site_types, labels_all, mutant_types):
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
    if num_large > len(inferred) - 2:
        print(f'number synonymous and linked to a nonsynonymous is {num_linked}')
    return (num_large - len(syn) + num_linked) / num_large

def number_nonsynonymous_nonlinked_alt(inferred, num_large, linked_all, labels_all, mutant_types):
    """ Find out the percentage of the num_large largest selection coefficients that are nonsynonymous 
    and are not linked to some nonsynonymous mutation. """
        
    ## Filter out linked groups if all mutations are synonymous or nonsynonymous.
    #linked_new = []
    #for i in range(len(site_types)):
    #    if 0 < len(np.array(site_types[i])[np.array(site_types[i])=='NS']) and len(np.array(site_types[i])[np.array(site_types[i])=='NS']) < len(site_types[i]):
    #        linked_new.append(linked[i])
    #linked_all = [linked_new[i][j] for i in range(len(linked_new)) for j in range(len(linked_new[i]))]
        
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
    if num_large > len(inferred) - 2:
        print(f'number synonymous and linked to a nonsynonymous is {num_linked}')
    return (num_large - len(syn) + num_linked) / num_large


def number_nonsynonymous(linked, site_types, labels, mutant_types):
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
    
    
def number_nonsynonymous_alt(linked_all, labels, mutant_types):
    """ Given a group of mutations, find the percentage that are nonsynonymous.
    Mutant_types are the types (synonymous or nonsynonymous) for the group of mutations.
    site_types are the types for the linked sites."""
    
    # Filter out linked groups if all mutations are synonymous or nonsynonymous.
    #linked_new = []
    #for i in range(len(site_types)):
    #    if 0 < len(np.array(site_types[i])[np.array(site_types[i])=='NS']) and len(np.array(site_types[i])[np.array(site_types[i])=='NS']) < len(site_types[i]):
    #        linked_new.append(linked[i])
    #linked_all = [linked_new[i][j] for i in range(len(linked_new)) for j in range(len(linked_new[i]))]
    
    # find percentage nonsynonymous
    syn        = np.array(labels)[np.array(mutant_types)=='S']
    num_linked = len([site for site in syn if site in linked_all])
    if len(labels)==0:
        return -1
    else:
        return (len(labels) - len(syn) + num_linked) / len(labels)


def syn_percentage_plots_old(file, link_file, link_file_all, syn_file, index_file=None, full_tv_file=None,
                         syn_lower_limit=55, correlation_plots=False, out_file='syn_dist_plot'):
    """ Makes a plot containing the distribution of inferred coefficients and the enrichment of nonsynonymous mutations.
    Combines selection_dist_plot and syn_plot functions.
    correlation_plots determines whether or not the 3 plots containing the correlations are made."""
    
    # loading and processing the data
    data            = np.load(file, allow_pickle=True)
    linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in EVERY region
    linked_anywhere = np.load(link_file_all, allow_pickle=True)    # sites that are fully linked in ANY region
    syn_data        = np.load(syn_file, allow_pickle=True)
    allele_number   = data['allele_number']
    labels          = [get_label(i[:-2]) + '-' + i[-1] for i in allele_number]
    types_temp      = syn_data['types']
    type_locs       = syn_data['locations']
    linked_all      = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    inferred        = data['selection']
    inferred_link   = np.zeros(len(linked_sites))
    digits          = len(str(len(inferred)))
    print('number of sites:', len(inferred))
    
    mutant_types  = []
    for i in labels:
        if i in type_locs:
            mutant_types.append(types_temp[list(type_locs).index(i)])
        else:
            mutant_types.append('unknown')
        
    inferred_new  = []    # the sum of the inferred coefficients for the linked sites
    for i in range(len(linked_sites)):
        for j in range(len(linked_sites[i])):
            if np.any(linked_sites[i][j]==np.array(labels)):
                inferred_link[i] += inferred[np.where(np.array(linked_sites[i][j])==np.array(labels))[0][0]]
    for i in range(len(labels)):
        if labels[i] not in linked_all:
            inferred_new.append(inferred[i])
    inferred_new = inferred_new + list(inferred_link)
    
    # Null distribution
    if full_tv_file:
        inferred_null = find_null_distribution(full_tv_file, link_file)
    
    # creating figure and subfigures
    fig  = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH / 1.618])
    grid = matplotlib.gridspec.GridSpec(1, 3, hspace=0.3, wspace=0.2, figure=fig, bottom=0.175, left=0.1, right=0.975, top=0.875)

    fig.text(0.125,   0.965,   'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.425,   0.965,   'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.725,   0.965,   'c', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    
    #fig.text(0.15,    0.925, 'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    #fig.text(0.55,    0.925, 'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    #fig.text(0.15,    0.575, 'c', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    #fig.text(0.55,    0.575, 'd', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    
    # plotting the distribution
    weights    = np.ones(len(inferred)) / len(inferred) 
    weights2   = np.ones(len(inferred_new)) / len(inferred_new) 
    x_min      = np.amin(inferred_new) - 0.025
    x_max      = np.amax(inferred_new) + 0.025
    tick_space = (x_max - x_min) / 6
    neg_ticks  = [-i for i in np.arange(0, -x_min, tick_space)]
    pos_ticks  = [i for i in np.arange(0, x_max, tick_space)]
    tick_locs  = neg_ticks[::-1] + pos_ticks
        
    ax2 = fig.add_subplot(grid[0, 0])
    ax2.hist(inferred_new, bins=50, range=[x_min+0.025, x_max-0.025], log=True, color=MAIN_COLOR) 
    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels([f'{int(round(i))}'.replace('-', '\N{MINUS SIGN}') for i in (100 * np.array(tick_locs))])
    ax2.set_ylim(0.75, 10**(digits-0.5))
    
    axes = [ax2]
    ax2.set_ylabel('Counts (Log)', fontsize=AXES_FONTSIZE)
    ax2.set_xlabel("Inferred selection coefficients (%)", fontsize=AXES_FONTSIZE)
    ax2.tick_params(width=SPINE_LW)
    for axis in axes:
        axis.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
        for line in ['right', 'top']:
            axis.spines[line].set_visible(False)
        for line in ['left', 'bottom']:
            axis.spines[line].set_linewidth(SPINE_LW)
    
    print('finding nonsynonymous')
    # Finding enrichment of nonsynonymous sites
    site_types = []
    for group in linked_anywhere:
        site_types_temp = []
        for site in group:
            if site in type_locs:
                site_types_temp.append(types_temp[list(type_locs).index(site)])
            else:
                site_types_temp.append('null')
            """
            if site in list(labels):
                site_types_temp.append(mutant_types[list(labels).index(site)])
            else:
                site_types_temp.append('null')
            """
        site_types.append(site_types_temp)
        
    for i in range(len(site_types)):
        if 'null' in site_types[i]:
            print(site_types[i])
            print(linked_anywhere[i])
            
    # Filter out linked groups if all mutations are synonymous or nonsynonymous.
    linked_new = []
    for i in range(len(site_types)):
        if 0 < len(np.array(site_types[i])[np.array(site_types[i])=='NS']) and len(np.array(site_types[i])[np.array(site_types[i])=='NS']) < len(site_types[i]):
            linked_new.append(linked_anywhere[i])
    linked_all = np.unique([linked_new[i][j] for i in range(len(linked_new)) for j in range(len(linked_new[i]))])
    
    nonsyn_nonlinked  = np.zeros(len(inferred) - 1)
    for i in range(len(inferred)-1):
        #nonsyn_nonlinked[i] = number_nonsynonymous_nonlinked(inferred, i+1, linked_anywhere, site_types, labels, mutant_types)
        nonsyn_nonlinked[i] = number_nonsynonymous_nonlinked_alt(inferred, i+1, linked_all, labels, mutant_types)
    
    print('finding nonsynonymous in intervals')
    # Find the percentage nonsynonymous with selection coefficients between given intervals
    #base       =  1 / 3
    #base       = 1 / 2
    #intervals  = np.power(base, np.arange(2,12))
    #intervals = list(np.power(base, np.arange(2,11))) + [0]
    base       = 1 / 1.75
    intervals  = list(np.power(base, np.arange(3, 16))) + [0]
    
    percent_ns = []
    for i in range(len(intervals) - 1):
        idxs_temp  = np.nonzero(np.logical_and(np.absolute(inferred)<intervals[i], np.absolute(inferred)>=intervals[i+1]))
        sites_temp = np.array(labels)[idxs_temp]
        types_temp = np.array(mutant_types)[idxs_temp]
        #num_nonsyn = number_nonsynonymous(linked_anywhere, site_types, sites_temp, types_temp)
        num_nonsyn = number_nonsynonymous_alt(linked_all, sites_temp, types_temp)
        percent_ns.append(num_nonsyn)
    # eliminate intervals at the end containing zero coefficients
    while percent_ns[0]==-1:
        percent_ns = percent_ns[1:]
        intervals  = intervals[1:]
        
    # Plotting the enrichment of nonsynonymous mutations vs. number of coefficients
    ax1  = fig.add_subplot(grid[0, 1])
    ax1.set_ylim(syn_lower_limit, 101)
    ax1.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW, length=2)
    for line in ['right',  'top']:
        ax1.spines[line].set_visible(False)
    for line in ['left', 'bottom']:
        ax1.spines[line].set_linewidth(SPINE_LW)
    sns.lineplot(np.arange(len(nonsyn_nonlinked))+1, 100*nonsyn_nonlinked, ax=ax1, lw=1, color=MAIN_COLOR)
    ax1.axhline(100*nonsyn_nonlinked[-1], lw=1, color=COMP_COLOR, alpha=0.75)
    ax1.text(0.25, (100*nonsyn_nonlinked[-1]-syn_lower_limit)/(100-syn_lower_limit) - 0.05, 'Background', fontsize=6, transform=ax1.transAxes, ha='center', va='center')
    ax1.set_xscale('log')
    ax1.set_xlim(1, len(nonsyn_nonlinked)+1)
    ax1.set_ylabel('Nonsynonymous (%)', fontsize=AXES_FONTSIZE)
    ax1.set_xlabel("Number of largest coefficients", fontsize=AXES_FONTSIZE)
    print('background:', 100*nonsyn_nonlinked[-1])
    
    # Plotting the percentage nonsynonymous in the different intervals
    ax3 = fig.add_subplot(grid[0, 2])
    ax3.tick_params(labelsize=6, width=SPINE_LW, length=2)
    for line in ['right',  'top']:
        ax3.spines[line].set_visible(False)
    for line in ['left', 'bottom']:
        ax3.spines[line].set_linewidth(SPINE_LW)
    ax3.set_xticks(intervals[:-1])
    ax3.plot(intervals[:-1], percent_ns, ds='steps-post', lw=1, color=MAIN_COLOR)
    ax3.set_xlabel('Interval of inferred coefficients', fontsize=AXES_FONTSIZE)
    ax3.set_ylabel('Nonsynonymous (%)', fontsize=AXES_FONTSIZE)
    ax3.set_xscale('log')
    print('non-synonymous percentage:', percent_ns)
    print('intervals:', intervals)
    #ax3.set_ylim(ax3.get_ylim()[::-1])
    
    # Plotting the null distribution
    """
    ax4 = fig.add_subplot(grid[0, 3])
    ax4.tick_params(labelsize=6, width=SPINE_LW, length=2)
    ax4.hist(inferred_null, bins=50, color=MAIN_COLOR, log=True)
    for line in ['top', 'right']:
        ax4.spines[line].set_visible(False)
    ax4.set_xticks([1, 100, 10000])
    ax4.text(0.15, 0.8, 'e', fontweight='bold', transform=ax2.transAxes, fontsize=8)
    """            
    
    plt.gcf().subplots_adjust(bottom=0.3, left=0.1, right=0.9)
    fig.savefig(os.path.join(fig_path, out_file + '.png'), dpi=1200)
    
    
def syn_percentage_plots(file, link_file, link_file_all, syn_file, index_file=None, full_tv_file=None,
                         syn_lower_limit=80, correlation_plots=False, out_file='syn_dist_plot'):
    """ Makes a plot containing the distribution of inferred coefficients and the enrichment of nonsynonymous mutations.
    Combines selection_dist_plot and syn_plot functions.
    correlation_plots determines whether or not the 3 plots containing the correlations are made."""
    
    # loading and processing the data
    data            = np.load(file, allow_pickle=True)
    linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in EVERY region
    linked_anywhere = np.load(link_file_all, allow_pickle=True)    # sites that are fully linked in ANY region
    syn_data        = np.load(syn_file, allow_pickle=True)
    allele_number   = data['allele_number']
    labels          = [get_label(i[:-2]) + '-' + i[-1] for i in allele_number]
    types_temp      = syn_data['types']
    type_locs       = syn_data['locations']
    linked_all      = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    inferred        = data['selection']
    inferred_link   = np.zeros(len(linked_sites))
    digits          = len(str(len(inferred)))
    print('number of sites:', len(inferred))
    
    mutant_types  = []
    for i in labels:
        if i in type_locs:
            mutant_types.append(types_temp[list(type_locs).index(i)])
        else:
            mutant_types.append('unknown')
    
    # creating figure and subfigures
    fig  = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH / 1.4])
    grid = matplotlib.gridspec.GridSpec(1, 2, hspace=0.3, wspace=0.3, figure=fig, bottom=0.175, left=0.1, right=0.975, top=0.875)

    fig.text(0.050,   0.965,   'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.525,   0.965,   'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    
    print('finding nonsynonymous')
    # Finding enrichment of nonsynonymous sites
    print('\tfinding site types')
    site_types = []
    for group in linked_anywhere:
        site_types_temp = []
        for site in group:
            if site in type_locs:
                site_types_temp.append(types_temp[list(type_locs).index(site)])
            else:
                site_types_temp.append('null')
        site_types.append(site_types_temp)
    
    ### DELETE THE BELOW FOR THE PAPER OR CHANGE DIRECTORY
    np.savez_compressed('/Users/brianlee/Desktop/linked-types-regional.npz', linked=linked_anywhere, types=site_types)
    np.savez_compressed('/Users/brianlee/Desktop/inference-data.npz', 
                        allele_number=allele_number, types=mutant_types, inferred=inferred, labels=labels)
        
    for i in range(len(site_types)):
        if 'null' in site_types[i]:
            print(site_types[i])
            print(linked_anywhere[i])
            
    # Filter out linked groups if all mutations are synonymous or nonsynonymous.
    print('\tfiltering out linked groups composed entirely of synonymous or nonsynonmyous mutations')
    linked_new = []
    for i in range(len(site_types)):
        mask     = np.array(site_types[i])=='NS'
        nonsyn_n = np.count_nonzero(mask)
        #if 0 < len(np.array(site_types[i])[np.array(site_types[i])=='NS']) and len(np.array(site_types[i])[np.array(site_types[i])=='NS']) < len(site_types[i]):
        if 0 < nonsyn_n < len(site_types[i]):
            linked_new.append(linked_anywhere[i])
    linked_all = np.unique([linked_new[i][j] for i in range(len(linked_new)) for j in range(len(linked_new[i]))])
    
    ### DELETE THE BELOW FOR THE PAPER OR CHANGE DIRECTORY
    np.save('/Users/brianlee/Desktop/linked-tononsyn-regional.npy', linked_all)
    
    linked_all = set(linked_all)
    
    nonsyn_nonlinked  = np.zeros(len(inferred) - 1)
    indices           = np.argsort(np.absolute(inferred))[::-1]
    mut_types_ordered = np.array(mutant_types)[indices]
    s_ordered         = np.array(inferred)[indices]
    labs_ordered      = np.array(labels)[indices]
    num_nonsyn_nolink = []
    running_total     = 0
    print('\tfinding percentage of nonsynonymous mutations ')
    for i in tqdm(range(len(labs_ordered))):
        if mut_types_ordered[i]=='NS' or labs_ordered[i] in linked_all:
            running_total += 1
        num_nonsyn_nolink.append(running_total)
    nonsyn_nonlinked = np.array(num_nonsyn_nolink) / (np.arange(1, len(num_nonsyn_nolink) + 1))
    """
    large_inferred    = inferred[indices]
    large_mut_types, large_labels = [], []
    for i in range(num_large):
        large_mut_types.append(mutant_types[indices[i]])
        large_labels.append(labels_all[indices[i]])
    large_mut_types = np.array(large_mut_types)
    #nonsyn = np.array(large_labels)[np.array(large_mut_types)=='NS']
    syn = np.array(large_labels)[np.array(large_mut_types)=='S']
    num_linked = len([site for site in syn if site in linked_all])
    if num_large > len(inferred) - 2:
        print(f'number synonymous and linked to a nonsynonymous is {num_linked}')
    return (num_large - len(syn) + num_linked) / num_large
    """
    #for i in range(len(inferred)-1):
    #    #nonsyn_nonlinked[i] = number_nonsynonymous_nonlinked(inferred, i+1, linked_anywhere, site_types, labels, mutant_types)
    #    nonsyn_nonlinked[i] = number_nonsynonymous_nonlinked_alt(inferred, i+1, linked_all, labels, mutant_types)
    
    print('finding nonsynonymous in intervals')
    # Find the percentage nonsynonymous with selection coefficients between given intervals
    #base       =  1 / 3
    #base       = 1 / 2
    #intervals  = np.power(base, np.arange(2,12))
    #intervals = list(np.power(base, np.arange(2,11))) + [0]
    base       = 1 / 1.75
    intervals  = list(np.power(base, np.arange(3, 16))) + [0]
    
    percent_ns = []
    for i in range(len(intervals) - 1):
        idxs_temp  = np.nonzero(np.logical_and(np.absolute(inferred)<intervals[i], np.absolute(inferred)>=intervals[i+1]))
        sites_temp = np.array(labels)[idxs_temp]
        types_temp = np.array(mutant_types)[idxs_temp]
        #num_nonsyn = number_nonsynonymous(linked_anywhere, site_types, sites_temp, types_temp)
        num_nonsyn = number_nonsynonymous_alt(linked_all, sites_temp, types_temp)
        percent_ns.append(num_nonsyn)
    # eliminate intervals at the end containing zero coefficients
    while percent_ns[0]==-1:
        percent_ns = percent_ns[1:]
        intervals  = intervals[1:]
        
    # Plotting the enrichment of nonsynonymous mutations vs. number of coefficients
    ax1  = fig.add_subplot(grid[0, 0])
    ax1.set_ylim(syn_lower_limit, 101)
    ax1.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW, length=2)
    for line in ['right',  'top']:
        ax1.spines[line].set_visible(False)
    for line in ['left', 'bottom']:
        ax1.spines[line].set_linewidth(SPINE_LW)
    sns.lineplot(np.arange(len(nonsyn_nonlinked))+1, 100*nonsyn_nonlinked, ax=ax1, lw=1, color=MAIN_COLOR)
    ax1.axhline(100*nonsyn_nonlinked[-1], lw=1, color=COMP_COLOR, alpha=0.75)
    ax1.text(0.25, (100*nonsyn_nonlinked[-1]-syn_lower_limit)/(100-syn_lower_limit) - 0.05, 'Background', fontsize=6, transform=ax1.transAxes, ha='center', va='center')
    ax1.set_xscale('log')
    ax1.set_xlim(1, len(nonsyn_nonlinked)+1)
    ax1.set_ylabel('Fraction\nnonsynonymous (%)', fontsize=AXES_FONTSIZE)
    ax1.set_xlabel("Number of largest coefficients", fontsize=AXES_FONTSIZE)
    print('background:', 100*nonsyn_nonlinked[-1])
    
    # Plotting the percentage nonsynonymous in the different intervals
    ax3 = fig.add_subplot(grid[0, 1])
    ax3.tick_params(labelsize=6, width=SPINE_LW, length=2)
    for line in ['right',  'top']:
        ax3.spines[line].set_visible(False)
    for line in ['left', 'bottom']:
        ax3.spines[line].set_linewidth(SPINE_LW)
    ax3.set_xticks(intervals[:-1])
    ax3.plot(intervals[:-1], percent_ns, ds='steps-post', lw=1, color=MAIN_COLOR)
    ax3.set_xlabel('Magnitude of selection coefficients (%)', fontsize=AXES_FONTSIZE)
    ax3.set_ylabel('Fraction\nnonsynonymous (%)', fontsize=AXES_FONTSIZE)
    ax3.set_xscale('log')
    print('non-synonymous percentage:', percent_ns)
    print('intervals:', intervals)
    #ax3.set_ylim(ax3.get_ylim()[::-1])
    
    # Plotting the null distribution
    """
    ax4 = fig.add_subplot(grid[0, 3])
    ax4.tick_params(labelsize=6, width=SPINE_LW, length=2)
    ax4.hist(inferred_null, bins=50, color=MAIN_COLOR, log=True)
    for line in ['top', 'right']:
        ax4.spines[line].set_visible(False)
    ax4.set_xticks([1, 100, 10000])
    ax4.text(0.15, 0.8, 'e', fontweight='bold', transform=ax2.transAxes, fontsize=8)
    """            
    
    plt.gcf().subplots_adjust(bottom=0.3, left=0.1, right=0.9)
    fig.savefig(os.path.join(fig_path, out_file + '.png'), dpi=1200)
    
    
def syn_percentage_plots_fast(file, link_file, link_file_all, syn_file, index_file=None, full_tv_file=None,
                              syn_lower_limit=85, correlation_plots=False, out_file='syn_dist_plot'):
    """ Makes a plot containing the distribution of inferred coefficients and the enrichment of nonsynonymous mutations.
    Combines selection_dist_plot and syn_plot functions.
    correlation_plots determines whether or not the 3 plots containing the correlations are made."""
    
    # loading and processing the data
    data            = np.load(file, allow_pickle=True)
    #linked_sites    = np.load(link_file, allow_pickle=True)        # sites that are fully linked in EVERY region
    linked_anywhere = np.load(link_file_all, allow_pickle=True)    # sites that are fully linked in ANY region
    syn_data        = np.load(syn_file, allow_pickle=True)
    allele_number   = data['allele_number']
    labels          = [get_label(i[:-2]) + '-' + i[-1] for i in allele_number]
    types_temp      = syn_data['types']
    type_locs       = syn_data['locations']
    #linked_all      = [linked_sites[i][j] for i in range(len(linked_sites)) for j in range(len(linked_sites[i]))]
    inferred        = data['selection']
    #inferred_link   = np.zeros(len(linked_sites))
    #digits          = len(str(len(inferred)))
    print('number of sites:', len(inferred))
    
    mutant_types  = []
    for i in labels:
        if i in type_locs:
            mutant_types.append(types_temp[list(type_locs).index(i)])
        else:
            mutant_types.append('unknown')
            
    np.savez_compressed('/Users/brianlee/Desktop/inference-data.npz', 
                        allele_number=allele_number, types=mutant_types, inferred=inferred, labels=labels)
    inf_data      = np.load('/Users/brianlee/Desktop/inference-data.npz', allow_pickle=True)
    allele_number = inf_data['allele_number']
    mutant_types  = inf_data['types']
    inferred      = inf_data['inferred']
    labels        = inf_data['labels']
    
    # creating figure and subfigures
    fig  = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH / 1.3])
    grid = matplotlib.gridspec.GridSpec(1, 2, hspace=0.3, wspace=0.3, figure=fig, bottom=0.175, left=0.1, right=0.975, top=0.875)

    fig.text(0.050,   0.95,   'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.525,   0.95,   'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    
    print('finding nonsynonymous')
    # Finding enrichment of nonsynonymous sites
    
    ### DELETE THE BELOW FOR THE PAPER OR CHANGE DIRECTORY
    #np.savez_compressed('/Users/brianlee/Desktop/linked-types-regional.npz', linked=linked_anywhere, types=site_types)
    type_data       = np.load('/Users/brianlee/Desktop/linked-types-regional.npz', allow_pickle=True)
    linked_anywhere = type_data['linked']
    site_types      = type_data['types']
    
    ### DELETE THE BELOW FOR THE PAPER OR CHANGE DIRECTORY
    #np.save('/Users/brianlee/Desktop/linked-tononsyn-regional.npy', linked_all)
    linked_all = np.load('/Users/brianlee/Desktop/linked-tononsyn-regional.npy', allow_pickle=True)
    
    linked_all = set(linked_all)
    
    nonsyn_nonlinked  = np.zeros(len(inferred) - 1)
    indices           = np.argsort(np.absolute(inferred))[::-1]
    mut_types_ordered = np.array(mutant_types)[indices]
    s_ordered         = np.array(inferred)[indices]
    labs_ordered      = np.array(labels)[indices]
    num_nonsyn_nolink = []
    running_total     = 0
    print('\tfinding percentage of nonsynonymous mutations ')
    for i in tqdm(range(len(labs_ordered))):
        if mut_types_ordered[i]=='NS' or labs_ordered[i] in linked_all:
            running_total += 1
        num_nonsyn_nolink.append(running_total)
    nonsyn_nonlinked = np.array(num_nonsyn_nolink) / (np.arange(1, len(num_nonsyn_nolink) + 1))
    
    
    print('finding nonsynonymous in intervals')
    # Find the percentage nonsynonymous with selection coefficients between given intervals
    base       = 1 / 1.75
    intervals  = list(np.power(base, np.arange(3, 16))) + [0]
    
    percent_ns = []
    for i in range(len(intervals) - 1):
        idxs_temp  = np.nonzero(np.logical_and(np.absolute(inferred)<intervals[i], np.absolute(inferred)>=intervals[i+1]))
        sites_temp = np.array(labels)[idxs_temp]
        types_temp = np.array(mutant_types)[idxs_temp]
        num_nonsyn = number_nonsynonymous_alt(linked_all, sites_temp, types_temp)
        percent_ns.append(num_nonsyn)
    # eliminate intervals at the end containing zero coefficients
    while percent_ns[0]==-1:
        percent_ns = percent_ns[1:]
        intervals  = intervals[1:]
        
    # Plotting the enrichment of nonsynonymous mutations vs. number of coefficients
    ax1  = fig.add_subplot(grid[0, 0])
    ax1.set_ylim(syn_lower_limit, 101)
    ax1.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW, length=2)
    for line in ['right',  'top']:
        ax1.spines[line].set_visible(False)
    for line in ['left', 'bottom']:
        ax1.spines[line].set_linewidth(SPINE_LW)
    sns.lineplot(np.arange(len(nonsyn_nonlinked))+1, 100*nonsyn_nonlinked, ax=ax1, lw=1, color=MAIN_COLOR)
    ax1.axhline(100*nonsyn_nonlinked[-1], lw=1, color=COMP_COLOR, alpha=0.75)
    ax1.text(0.25, (100*nonsyn_nonlinked[-1]-syn_lower_limit)/(100-syn_lower_limit) - 0.05, 'Background', fontsize=6, transform=ax1.transAxes, ha='center', va='center')
    ax1.set_xscale('log')
    ax1.set_xlim(1, len(nonsyn_nonlinked)+1)
    ax1.set_ylabel('Fraction\nnonsynonymous (%)', fontsize=AXES_FONTSIZE)
    ax1.set_xlabel("Number of largest coefficients", fontsize=AXES_FONTSIZE)
    print('background:', 100*nonsyn_nonlinked[-1])
    
    # Plotting the percentage nonsynonymous in the different intervals
    ax3 = fig.add_subplot(grid[0, 1])
    ax3.tick_params(labelsize=6, width=SPINE_LW, length=2)
    for line in ['right',  'top']:
        ax3.spines[line].set_visible(False)
    for line in ['left', 'bottom']:
        ax3.spines[line].set_linewidth(SPINE_LW)
    ax3.set_xticks(intervals[:-1])
    ax3.plot(intervals[:-1], percent_ns, ds='steps-post', lw=1, color=MAIN_COLOR)
    ax3.set_xlabel('Magnitude of selection coefficients (%)', fontsize=AXES_FONTSIZE)
    ax3.set_ylabel('Fraction\nnonsynonymous (%)', fontsize=AXES_FONTSIZE)
    ax3.set_xscale('log')
    print('non-synonymous percentage:', percent_ns)
    print('intervals:', intervals)
    
    plt.gcf().subplots_adjust(bottom=0.3, left=0.1, right=0.9)
    fig.savefig(os.path.join(fig_path, out_file + '.png'), dpi=1200)


def syn_enrichment_protein(inf_file, syn_file, out_file='nonsynonymous-enrichment-protein'):
    """ Plots the enrichment of nonsynoymous mutatations in each protein"""
    
    def find_num_nonsyn(inferred, num_large, labels, types):
        """ Find the number of nonsynonymous and not linked to nonsynonymous mutations in the top num_large mutations"""
        indices       = np.argsort(np.absolute(inferred))[-num_large:]
        large_inf     = np.array(inferred)[indices]
        large_labels  = np.array(labels)[indices]
        large_types   = np.array(types)[indices]
        nonsynonymous = np.array(large_labels)[large_types=='NS']
        count         = len(nonsynonymous)
        return count / num_large
    
    #linked_sites   = np.load(link_file, allow_pickle=True)
    type_data      = np.load(syn_file,  allow_pickle=True)
    inf_data       = np.load(inf_file,  allow_pickle=True)
    syn_sites      = type_data['locations']
    types_temp     = type_data['types']
    inferred       = inf_data['selection']
    allele_number  = inf_data['allele_number']
    labels         = [get_label(i) for i in allele_number]
    syn_labels_temp= [get_label(i) for i in syn_sites]
    syn_types      = []
    syn_labels     = []
    for i in allele_number:
        if i in syn_sites:
            syn_types.append(types_temp[list(syn_sites).index(i)])
            syn_labels.append(syn_labels_temp[list(syn_sites).index(i)])
        else:
            syn_types.append('Unknown')
            
    # Filter out linked groups if they are entirely synonymous or entirely nonsynonymous
    """
    link_types = []
    for i in range(len(linked_sites)):
        types_temp = []
        for j in range(len(linked_sites[i])):
            if linked_sites[i][j] in labels:
                types_temp.append(syn_types[syn_labels.index(linked_sites[i][j])])
        if len(types_temp)>0:
            link_types.append(types_temp)
            
    linked_new     = []
    link_types_new = []
    for i in range(len(linked_sites)):
        if 0 < len(np.array(link_types[i])[np.array(link_types[i])=='NS']) and len(np.array(link_types[i])[np.array(link_types[i])=='NS']) < len(link_types[i]):
            linked_new.append(linked_sites[i])
            link_types_new.append(link_types[i])
    linked_all     = [linked_new[i][j] for i in range(len(linked_new)) for j in range(len(linked_new[i]))]
    link_types_all = [link_types_new[i][j] for i in range(len(link_types_new)) for j in range(len(link_types_new[i]))]
    """
            
    # Get the selection coefficients filtered by which protein the mutation is in
    proteins, site_labels, inf = dp.separate_by_protein(inf_file)
    
    # Find the type of each mutation in the different proteins
    types_protein = []
    for i in range(len(site_labels)):
        types_temp = []
        for j in range(len(site_labels[i])):
            types_temp.append(syn_types[syn_labels.index(site_labels[i][j])])
        types_protein.append(types_temp)
    
    # Separate the linked sites into the proteins that they are in
    """
    link_protein = []   # the sites that are linked to other sites, separated by protein
    syn_protein  = []
    for i in range(len(proteins)):
        syn_temp, link_temp = [], []
        for j in range(len(linked_all)):
            if linked_all[j][:len(proteins[i])] == proteins[i]:
                link_temp.append(linked_all[j])
                syn_temp.append(link_types_all[j])
        link_protein.append(link_temp)
        syn_protein.append(syn_temp)
    """
    
    # Find the nonsynonymous enrichment in each protein
    nonsyn_enrichment = []
    for i in range(len(proteins)):
        enrichment_temp = []
        for j in range(len(inf[i]) - 1):
            enrichment_temp.append(find_num_nonsyn(inf[i], j+1, site_labels[i], types_protein[i]))
        nonsyn_enrichment.append(enrichment_temp)
    nonsyn_enrichment = np.array(nonsyn_enrichment)
    
    # Make figure and subplots
    rows = int(len(proteins) / 3) + 1
    fig  = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 1.618])
    grid = matplotlib.gridspec.GridSpec(rows, 3, wspace=0.01, hspace=0.5, figure=fig)
    
    ax1  = fig.add_subplot(grid[:,:])
    for line in ['left', 'right', 'top', 'bottom']:
        ax1.spines[line].set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('Number of coefficients', fontsize=AXES_FONTSIZE, labelpad=20)
    ax1.set_ylabel('Nonsynonymous (%)', fontsize=AXES_FONTSIZE, labelpad=25)
    
    # Plot the nonsynonymous enrichment in each protein
    for i in range(rows):
        for j in range(3):
            if 3*i + j < len(proteins):
                ax = fig.add_subplot(grid[i,j])
                ax.set_title(proteins[3*i + j], fontsize=AXES_FONTSIZE)
                ax.plot(np.arange(len(nonsyn_enrichment[3*i+j])), 100*np.array(nonsyn_enrichment[3*i+j]), color=MAIN_COLOR)
                #ax.set_xlabel('Number of coefficients')
                ax.set_ylim(30, 100)
                if j != 0:
                    ax.set_yticks([])
                #    ax.set_ylabel('Proportion nonsynonymous (%%)')
                ax.axhline(100*nonsyn_enrichment[3*i+j][-1], lw=0.3, color=COMP_COLOR, alpha=0.75)
                ax.text(0.25, (100*nonsyn_enrichment[3*i+j][-1]-30)/(100-30) - 0.075, 'background', color=COMP_COLOR, 
                        fontsize=6, transform=ax.transAxes, ha='center', va='center')
                ax.tick_params(labelsize=TICK_FONTSIZE, pad=0.05, length=0.5, width=SPINE_LW)
                for line in ['right', 'top']:
                    ax.spines[line].set_visible(False)
    fig.savefig(os.path.join(fig_path, out_file+'.png'), dpi=1200)
    
def get_label2(i):
    return get_label(i[:-2]) + '-' + i[-1]
        

#def syn_enrichment_scatter(inf_file, syn_file, index_file, ax1):
def syn_enrichment_scatter(inf_file, syn_file, locations, counts, counts_normed, ax1):
    """ Plots the enrichment of nonsynoymous mutatations in each protein vs. the mutation density"""
    
    def find_num_nonsyn(inferred, num_large, labels, types):
        """ Find the number of nonsynonymous and not linked to nonsynonymous mutations in the top num_large mutations"""
        indices       = np.argsort(np.absolute(inferred))[-num_large:]
        large_inf     = np.array(inferred)[indices]
        large_labels  = np.array(labels)[indices]
        large_types   = np.array(types)[indices]
        nonsynonymous = np.array(large_labels)[large_types=='NS']
        count         = len(nonsynonymous)
        return count / num_large
    
    #linked_sites   = np.load(link_file, allow_pickle=True)
    type_data      = np.load(syn_file,  allow_pickle=True)
    inf_data       = np.load(inf_file,  allow_pickle=True)
    syn_sites      = type_data['locations']
    types_temp     = type_data['types']
    inferred       = inf_data['selection']
    allele_number  = inf_data['allele_number']
    labels         = [get_label2(i) for i in allele_number]
    syn_labels_temp= [i for i in syn_sites]
    syn_types      = []
    syn_labels     = []
    for i in labels:
        if i in syn_sites:
            syn_types.append(types_temp[list(syn_sites).index(i)])
            syn_labels.append(syn_labels_temp[list(syn_sites).index(i)])
        else:
            syn_types.append('Unknown')
            
    # Get the selection coefficients filtered by which protein the mutation is in
    proteins, site_labels, inf = dp.separate_by_protein(inf_file)
    
    # Find the type of each mutation in the different proteins
    types_protein = []
    for i in range(len(site_labels)):
        types_temp = []
        for j in range(len(site_labels[i])):
            types_temp.append(syn_types[syn_labels.index(site_labels[i][j])])
        types_protein.append(types_temp)
    
    # Find the nonsynonymous enrichment in each protein
    nonsyn_enrichment = []
    for i in range(len(proteins)):
        nonsyn_enrichment.append(find_num_nonsyn(inf[i], len(inf[i]), site_labels[i], types_protein[i]))
    nonsyn_enrichment = np.array(nonsyn_enrichment)
    
    # finding the mutation density in each protein
    #locations, counts, counts_normed = dp.mutations_per_protein(index_file)
    counts_new = []
    for i in range(len(proteins)):
        counts_new.append(counts_normed[list(locations).index(proteins[i])])
    
    locations_new = []
    for i in proteins:
        if i not in ['S', 'E', 'N', 'M']:
            locations_new.append(i.lower())
        else:
            locations_new.append(i)
    proteins = locations_new
    
    print(counts_new)
    
    # Make figure and subplots
    #fig  = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH])
    #ax1  = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    for line in ['right', 'top']:
        ax1.spines[line].set_visible(False)
    ax1.set_ylabel('Nonsynonymous (%)', fontsize=AXES_FONTSIZE)
    ax1.set_xlabel('Mutation density', fontsize=AXES_FONTSIZE)
    ax1.plot(counts_new, 100*nonsyn_enrichment, '.-', linewidth=0)
    ax1.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
    for i, txt in enumerate(proteins):
        ax1.annotate(txt, (counts_new[i] + 0.002, 100*(nonsyn_enrichment[i] + 0.01)), fontsize=6)
    
    #fig.savefig(os.path.join(fig_path, out_file+'.png'), dpi=1200)
    
    correlation = calculate_correlation(counts_new, nonsyn_enrichment)
    print("correlation between nonsynonymous enrichment and mutation density:", correlation)
    ax1.text(0.8, 0.15, 'correlation: %.2f' % correlation, transform=ax1.transAxes, fontsize=8)
    
    
#def nonsyn_selection_plot(inf_file, syn_file, index_file, ax1, median=True, n_elim=0):
def nonsyn_selection_plot(inf_file, syn_file, locations, counts, counts_normed, ax1, median=True, n_elim=0):
    """ Plots the nonsynonymous enrichment in each protein vs the average inferred selection coefficient"""
    
    def find_num_nonsyn(inferred, num_large, labels, types):
        """ Find the number of nonsynonymous and not linked to nonsynonymous mutations in the top num_large mutations"""
        indices       = np.argsort(np.absolute(inferred))[-num_large:]
        large_inf     = np.array(inferred)[indices]
        large_labels  = np.array(labels)[indices]
        large_types   = np.array(types)[indices]
        nonsynonymous = np.array(large_labels)[large_types=='NS']
        count         = len(nonsynonymous)
        return count / num_large
    
    #linked_sites   = np.load(link_file, allow_pickle=True)
    type_data      = np.load(syn_file,  allow_pickle=True)
    inf_data       = np.load(inf_file,  allow_pickle=True)
    syn_sites      = type_data['locations']
    types_temp     = type_data['types']
    inferred       = inf_data['selection']
    allele_number  = inf_data['allele_number']
    labels         = [get_label2(i) for i in allele_number]
    syn_labels_temp= [i for i in syn_sites]
    syn_types      = []
    syn_labels     = []
    for i in labels:
        if i in syn_sites:
            syn_types.append(types_temp[list(syn_sites).index(i)])
            syn_labels.append(syn_labels_temp[list(syn_sites).index(i)])
        else:
            syn_types.append('Unknown')
            
    # Get the selection coefficients filtered by which protein the mutation is in
    proteins, site_labels, inf = dp.separate_by_protein(inf_file)
    
    # Find the type of each mutation in the different proteins
    types_protein = []
    for i in range(len(site_labels)):
        types_temp = []
        for j in range(len(site_labels[i])):
            types_temp.append(syn_types[syn_labels.index(site_labels[i][j])])
        types_protein.append(types_temp)
    
    # Find the nonsynonymous enrichment in each protein
    nonsyn_enrichment = []
    for i in range(len(proteins)):
        nonsyn_enrichment.append(find_num_nonsyn(inf[i], len(inf[i]), site_labels[i], types_protein[i]))
    nonsyn_enrichment = np.array(nonsyn_enrichment)
    
    # finding the average selection coefficient in each protein
    if n_elim>0 and not median:
        idxs     = np.argsort(inferred)[::-1][:-n_elim]
        labs_all = np.array(labels)[idxs]
        new_labels, new_inf = [], []
        for i in range(len(inf)):
            temp_labels, temp_inf = [], []
            for j in range(len(inf[i])):
                if labels[i][j] in labs_all:
                    temp_labels.append(labels[i][j])
                    temp_inf.append(inf[i][j])
            new_labels.append(temp_labels)
            new_inf.append(temp_inf)
        labels  = new_labels
        inf     = new_inf
    
    if median:
        inf_average = [np.median(i) for i in inf]
    else:
        inf_average = [np.mean(i) for i in inf]
        
    locations_new = []
    for i in proteins:
        if i not in ['S', 'E', 'N', 'M']:
            locations_new.append(i.lower())
        else:
            locations_new.append(i)
    proteins = locations_new
    
    # Make figure and subplots
    #fig  = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH])
    #ax1  = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    for line in ['right', 'top']:
        ax1.spines[line].set_visible(False)
    ax1.set_ylabel('Nonsynonymous (%)', fontsize=AXES_FONTSIZE)
    ax1.set_xlabel('Average inferred selection (%)', fontsize=AXES_FONTSIZE)
    ax1.plot(100*np.array(inf_average), 100*nonsyn_enrichment, '.-', linewidth=0)
    ax1.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
    for i, txt in enumerate(proteins):
        ax1.annotate(txt, (np.array(inf_average[i])*100 + 0.0002, 100*(nonsyn_enrichment[i] + 0.01)), fontsize=6)
    
    #fig.savefig(os.path.join(fig_path, out_file+'.png'), dpi=1200)
    
    correlation = calculate_correlation(inf_average, nonsyn_enrichment)
    print("correlation between nonsynonymous and average inferred coefficient:", correlation)
    ax1.text(0.8, 0.15, 'correlation: %.2f' % correlation, transform=ax1.transAxes, fontsize=8)
        
    
def syn_plot(inf_file, syn_file, link_file):
    """ Plot the percentage of the largest mutations that are nonsynonymous or not linked to a nonsynonymous mutation
    versus the number of the largest coefficients considered"""
    data              = np.load(inf_file, allow_pickle=True)
    inferred          = data['selection']
    allele_number     = data['allele_number']
    labels            = [get_label(i) for i in allele_number]
    linked_sites      = np.load(link_file, allow_pickle=True)
    type_data         = np.load(syn_file, allow_pickle=True)
    mutant_types_temp = type_data['types']
    sites             = type_data['locations']
    mutant_types      = []
    for i in allele_number:
        if i in sites:
            mutant_types.append(mutant_types_temp[list(sites).index(i)])
        else:
            mutant_types.append('unknown')
            
    site_types = []
    for group in linked_sites:
        site_types_temp = []
        for site in group:
            if np.any(site == np.array(labels)):
                site_types_temp.append(mutant_types[list(labels).index(site)])
            else:
                site_types_temp.append('null')
        site_types.append(site_types_temp)
    
    nonsyn_nonlinked  = np.zeros(len(inferred) - 1)
    for i in range(len(inferred)-1):
        nonsyn_nonlinked[i] = number_nonsynonymous_nonlinked(inferred, i+1, linked_sites, site_types, labels, mutant_types)
        
    fig = plt.figure(figsize=[SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH/1.618])
    ax  = fig.add_subplot(1,1,1)
    #ax.grid(b=True, axis='y', linewidth=0.2)
    ax.set_ylim(60, 100)
    #plt.yticks(np.arange(11)/10)
    ax.set_title("Non-synonymous enrichment", fontsize=AXES_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW, length=2)
    for line in ['left', 'right',  'top', 'bottom']:
        ax.spines[line].set_linewidth(SPINE_LW)
    ax.set_xlabel("Number of largest\ncoefficients", fontsize=AXES_FONTSIZE)
    ax.set_ylabel("Nonsynonymous (%)", fontsize=AXES_FONTSIZE)
    sns.lineplot(np.arange(len(nonsyn_nonlinked))+1, 100*nonsyn_nonlinked, ax=ax, lw=0.3)
    ax.axhline(100*nonsyn_nonlinked[-1], lw=0.3, color='lightcoral', alpha=0.75)
    ax.text(-0.065, 0.175, f'{round(100*nonsyn_nonlinked[-1],1)}', color='lightcoral', fontsize=4, transform=ax.transAxes, ha='center', va='center')
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.gcf().subplots_adjust(left=0.2)
    plt.savefig(os.path.join(fig_path, 'nonsynonymous.png'), dpi=1200)
    
    fig2 = plt.figure(figsize=[SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH/1.618])
    ax2  = fig2.add_subplot(1,1,1)
    sns.lineplot(np.arange(50), nonsyn_nonlinked[:50], ax=ax2, lw=0.3)
    ax2.grid(b=True, axis='y', linewidth=0.2)
    ax2.set_ylim(0.6, 1)
    #plt.yticks(np.arange(11)/10)
    ax2.set_title("Non-synonymous enrichment", fontsize=AXES_FONTSIZE)
    ax2.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW, length=2)
    for line in ['left', 'right',  'top', 'bottom']:
        ax.spines[line].set_linewidth(SPINE_LW)
    ax2.set_xlabel("Number of largest\ncoefficients", fontsize=AXES_FONTSIZE)
    ax2.set_ylabel("Nonsynonymous (%)", fontsize=AXES_FONTSIZE)
    ax2.axhline(nonsyn_nonlinked[-1], lw=0.3, color='lightcoral', alpha=0.75)
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.gcf().subplots_adjust(left=0.2)
    plt.savefig(os.path.join(fig_path, 'nonsynonymous-top50.png'), dpi=1200)
    
    print(nonsyn_nonlinked[-1], np.sort(np.absolute(inferred))[-200])
    

def finite_sampling_plot(rep_folder, rep_perfect):
    """Compares inference using real or time-varying parameters 
    under the conditions of finite or perfect sampling for various population sizes. """
    color_map = palet.colorbrewer.sequential.Blues_4.get_mpl_colormap()
    n_pops  = len([name for name in os.listdir(rep_folder) if (os.path.isdir(os.path.join(rep_folder, name)) and name!='.ipynb_checkpoints')]) # number of different populations to be tested
    fig = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, n_pops])
    grid = matplotlib.gridspec.GridSpec(n_pops*4+2, 5, figure=fig, wspace=0, hspace=0)
    axes_list = []
    ax_titles0 = fig.add_subplot(grid[0:2, 0])
    ax_titles0.text(0.5, 0.5, "Population\nSize", transform=ax_titles0.transAxes, ha='center', va='center', fontweight='bold', fontsize=AXES_FONTSIZE)
    ax_titles0.spines['right'].set_linewidth(SPINE_LW)
    ax_titles0.set_xticks([])
    ax_titles0.set_yticks([])
    ax_titles1 = fig.add_subplot(grid[0:2, 1])
    ax_titles1.text(0.5, 0.5, "Sampling", transform=ax_titles1.transAxes, ha='center', va='center', fontweight='bold', fontsize=AXES_FONTSIZE)
    axes_list.append(ax_titles1)
    ax_titles2 = fig.add_subplot(grid[0:2, 2])
    ax_titles2.text(0.5, 0.5, "Inference\nParameter (n)", transform=ax_titles2.transAxes, ha='center', va='center', fontweight='bold', fontsize=AXES_FONTSIZE)
    axes_list.append(ax_titles2)
    ax_titles3 = fig.add_subplot(grid[0:2, 3])
    ax_titles3.text(0.5, 0.5, "AUROC\nBeneficial", transform=ax_titles3.transAxes, ha='center', va='center', fontweight='bold', fontsize=AXES_FONTSIZE)
    axes_list.append(ax_titles3)
    ax_titles4 = fig.add_subplot(grid[0:2, 4])
    ax_titles4.text(0.5, 0.5, "AUROC\nDeleterious", transform=ax_titles4.transAxes, ha='center', va='center', fontweight='bold', fontsize=AXES_FONTSIZE)
    ax_titles4.spines['left'].set_linewidth(SPINE_LW)
    ax_titles4.set_xticks([])
    ax_titles4.set_yticks([])
        
    i = 1
    for folder in sorted(os.listdir(os.fsencode(rep_folder))):
        tv_file, const_file, pop_file = None, None, None
        folder_name = os.fsdecode(folder)
        folder_path = os.path.join(rep_folder, folder_name)
        if os.path.isdir(folder_path) and folder_name!='.ipynb_checkpoints':
            #print(folder_name, '(directory found)')
            for file in sorted(os.listdir(folder_path)):
                if os.fsdecode(file)[:10]=='population':
                    pop_file   = np.load(os.path.join(rep_folder, folder_name, os.fsdecode(file)), allow_pickle=True)
                elif os.fsdecode(file)[-6:]=='tv.npz':
                    tv_file    = np.load(os.path.join(rep_folder, folder_name, os.fsdecode(file)), allow_pickle=True)
                elif os.fsdecode(file)[-9:]=='const.npz':
                    const_file = np.load(os.path.join(rep_folder, folder_name, os.fsdecode(file)), allow_pickle=True)
                else:
                    print('unknown file')
                #print(file)
            # calculate AUROC
            folder_path2 = os.path.join(rep_perfect, folder_name)
            tv_perfect, const_perfect = None, None
            for file in sorted(os.listdir(folder_path2)):
                if os.fsdecode(file)[-6:]=='tv.npz':
                    tv_perfect    = np.load(os.path.join(rep_perfect, folder_name, os.fsdecode(file)), allow_pickle=True)
                elif os.fsdecode(file)[-9:]=='const.npz':
                    const_perfect = np.load(os.path.join(rep_perfect, folder_name, os.fsdecode(file)), allow_pickle=True)
                
            AUC_ben_tv,    AUC_del_tv    = calculate_AUROC(tv_file['actual'],    tv_file['inferred'])
            AUC_ben_const, AUC_del_const = calculate_AUROC(const_file['actual'], const_file['inferred'])
                
            AUC_per_ben_tv, AUC_per_del_tv       = calculate_AUROC(tv_perfect['actual'],    tv_perfect['inferred'])
            AUC_per_ben_const, AUC_per_del_const = calculate_AUROC(const_perfect['actual'], const_perfect['inferred'])
                
            # can use the below to color code the squares based on which of tv or const provides the better estimate
            # uncomment the code in the plotting section to instead shade them according to a color map and the AUROC value
            br = ['cornflowerblue', 'lightcoral']
            if AUC_ben_tv > AUC_ben_const:
                c_ben_finite = br
            else:
                c_ben_finite = br[::-1]
            if AUC_del_tv > AUC_del_const:
                c_del_finite = br
            else:
                c_del_finite = br[::-1]
            if AUC_per_ben_tv > AUC_per_ben_const:
                c_ben_perfect = br
            else:
                c_ben_perfect = br[::-1]
            if AUC_per_del_tv > AUC_per_del_const:
                c_del_perfect = br
            else:
                c_del_perfect = br[::-1]
                
            
             # make plots
            ax0 = fig.add_subplot(grid[4*i-2:4*i+2, 0])
            sns.lineplot(np.arange(len(pop_file)), pop_file, lw=0.5, ax=ax0)
            ax0.spines['right'].set_linewidth(SPINE_LW)
            ax0.set_xticks([])
            ax0.set_yticks([])
                
            ax1a = fig.add_subplot(grid[4*i-2:4*i, 1])
            ax1a.text(0.5, 0.5, 'Finite', transform=ax1a.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            axes_list.append(ax1a)
                
            ax1b = fig.add_subplot(grid[4*i:4*i+2, 1])
            ax1b.text(0.5, 0.5, 'Perfect', transform=ax1b.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            axes_list.append(ax1b)
                
            ax2a = fig.add_subplot(grid[4*i-2, 2])
            ax2a.text(0.5, 0.5, 'Time-Varying', transform=ax2a.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            if AUC_ben_tv > AUC_ben_const and AUC_del_tv > AUC_del_const:
                ax2a.patch.set_facecolor(c_ben_finite[1])
            elif AUC_ben_tv < AUC_ben_const and AUC_del_tv < AUC_del_const:
                ax2a.patch.set_facecolor(c_ben_finite[0])
            ax2a.spines['bottom'].set_linewidth(SPINE_LW)
            axes_list.append(ax2a)
                
            ax2b = fig.add_subplot(grid[4*i-1, 2])
            ax2b.text(0.5, 0.5, 'Constant', transform=ax2b.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            if AUC_ben_tv > AUC_ben_const and AUC_del_tv > AUC_del_const:
                ax2b.patch.set_facecolor(c_ben_finite[0])
            elif AUC_ben_tv < AUC_ben_const and AUC_del_tv < AUC_del_const:
                ax2b.patch.set_facecolor(c_ben_finite[1])
            ax2b.spines['top'].set_linewidth(SPINE_LW)
            axes_list.append(ax2b)
                
            ax2c = fig.add_subplot(grid[4*i, 2])
            ax2c.text(0.5, 0.5, 'Time-Varying', transform=ax2c.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            if AUC_per_ben_tv > AUC_per_ben_const and AUC_per_del_tv > AUC_per_del_const:
                ax2c.patch.set_facecolor(c_ben_perfect[0])
            elif AUC_per_ben_tv < AUC_per_ben_const and AUC_per_del_tv < AUC_per_del_const:
                ax2c.patch.set_facecolor(c_ben_perfect[1])
            ax2c.spines['bottom'].set_linewidth(SPINE_LW)
            axes_list.append(ax2c)
                
            ax2d = fig.add_subplot(grid[4*i+1, 2])
            ax2d.text(0.5, 0.5, 'Constant', transform=ax2d.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            if AUC_per_ben_tv > AUC_per_ben_const and AUC_per_del_tv > AUC_per_del_const:
                ax2d.patch.set_facecolor(c_ben_perfect[1])
            elif AUC_per_ben_tv < AUC_per_ben_const and AUC_per_del_tv < AUC_per_del_const:
                ax2d.patch.set_facecolor(c_ben_perfect[0])
            ax2d.spines['top'].set_linewidth(SPINE_LW)
            axes_list.append(ax2d)
                
            ax3a = fig.add_subplot(grid[4*i-2, 3])
            #ax3a.patch.set_facecolor(color_map(AUC_ben_tv))
            ax3a.patch.set_facecolor(c_ben_finite[0])
            ax3a.text(0.5, 0.5, str(AUC_ben_tv)[:5], transform=ax3a.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            ax3a.spines['bottom'].set_linewidth(SPINE_LW)
            axes_list.append(ax3a)
                                         
            ax3b = fig.add_subplot(grid[4*i-1, 3])
            #ax3b.patch.set_facecolor(color_map(AUC_ben_const))
            ax3b.patch.set_facecolor(c_ben_finite[1])
            ax3b.text(0.5, 0.5, str(AUC_ben_const)[:5], transform=ax3b.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE) 
            ax3b.spines['top'].set_linewidth(SPINE_LW)
            axes_list.append(ax3b)
                
            ax3c = fig.add_subplot(grid[4*i, 3])
            #ax3c.patch.set_facecolor(color_map(AUC_per_ben_tv))
            ax3c.patch.set_facecolor(c_ben_perfect[0])
            ax3c.text(0.5, 0.5, str(AUC_per_ben_tv)[:5], transform=ax3c.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            ax3c.spines['bottom'].set_linewidth(SPINE_LW)
            axes_list.append(ax3c)
                
            ax3d = fig.add_subplot(grid[4*i+1, 3])
            #ax3d.patch.set_facecolor(color_map(AUC_per_ben_tv))
            ax3d.patch.set_facecolor(c_ben_perfect[1])
            ax3d.text(0.5, 0.5, str(AUC_per_ben_const)[:5], transform=ax3d.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            ax3d.spines['top'].set_linewidth(SPINE_LW)
            axes_list.append(ax3d)
               
            ax4a = fig.add_subplot(grid[4*i-2, 4])
            #ax4a.patch.set_facecolor(color_map(AUC_del_tv))
            ax4a.patch.set_facecolor(c_del_finite[0])
            ax4a.text(0.5, 0.5, str(AUC_del_tv)[:5], transform=ax4a.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE) 
            ax4a.spines['bottom'].set_linewidth(SPINE_LW)
            ax4a.spines['left'].set_linewidth(SPINE_LW)
            ax4a.set_xticks([])
            ax4a.set_yticks([])
            
            ax4b = fig.add_subplot(grid[4*i-1, 4])
            #ax4b.patch.set_facecolor(color_map(AUC_del_const))
            ax4b.patch.set_facecolor(c_del_finite[1])
            ax4b.text(0.5, 0.5, str(AUC_del_const)[:5], transform=ax4b.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE) 
            ax4b.spines['top'].set_linewidth(SPINE_LW)
            ax4b.spines['left'].set_linewidth(SPINE_LW)
            ax4b.set_xticks([])
            ax4b.set_yticks([])
                
            ax4c = fig.add_subplot(grid[4*i, 4])
            #ax4c.patch.set_facecolor(color_map(AUC_per_del_tv))
            ax4c.patch.set_facecolor(c_del_perfect[0])
            ax4c.text(0.5, 0.5, str(AUC_per_del_tv)[:5], transform=ax4c.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE) 
            ax4c.spines['bottom'].set_linewidth(SPINE_LW)
            ax4c.spines['left'].set_linewidth(SPINE_LW)
            ax4c.set_xticks([])
            ax4c.set_yticks([])
            
            ax4d = fig.add_subplot(grid[4*i+1, 4])
            #ax4d.patch.set_facecolor(color_map(AUC_per_del_const))
            ax4d.patch.set_facecolor(c_del_perfect[1])
            ax4d.text(0.5, 0.5, str(AUC_per_del_const)[:5], transform=ax4d.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE) 
            ax4d.spines['top'].set_linewidth(SPINE_LW)
            ax4d.spines['left'].set_linewidth(SPINE_LW)
            ax4d.set_xticks([])
            ax4d.set_yticks([])
    
            i+=1
        
    for axis in axes_list:
        for line in ['right', 'left']:
            axis.spines[line].set_linewidth(SPINE_LW)
        axis.set_yticks([])
        axis.set_xticks([])
    plt.gcf().subplots_adjust(bottom=0.01)
    plt.gcf().subplots_adjust(top=0.99)
    plt.gcf().subplots_adjust(left=0.01)
    plt.gcf().subplots_adjust(right=0.99)
    plt.savefig(os.path.join(fig_path, 'finite-sampling.png'), dpi=1200)

        
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
    # find sites whose frequency is too small
    allele_number = [] 
    for i in range(len(traj)):
        for j in range(len(traj[i][0])):
            if np.amax(traj[i][:, j]) > cutoff:
                allele_number.append(mutant_sites[i][j])
    allele_number = np.sort(np.unique(np.array(allele_number)))
    return allele_number


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


def sort_color(value):
    # if the actual coefficient is positive, choose blue, if neutral, grey, and if deleterious, red
    if value>0:
        return mcolors.to_rgb('b')
    if value==0:
        #return mcolors.to_rgb(mcolors.CSS4_COLORS['grey'])
        return mcolors.to_rgb('0.5')
    if value<0:
        return mcolors.to_rgb('r')
    

def sort_color2(value):
    # if the actual coefficient is positive, choose blue, if neutral, grey, and if deleterious, red
    if value>0:
        return 'red'
    if value==0:
        #return mcolors.to_rgb(mcolors.CSS4_COLORS['grey'])
        return 'grey'
    if value<0:
        return 'blue'


def main(args):
    """Infer time-varying selection coefficients from the results of a Wright-Fisher simulation"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('--sim_data',    type=str,    default=None,                    help='.npz file containing the results of a simulation')
    parser.add_argument('--inf_data',    type=str,    default=None,                    help='.npz file containing the inference and error bars')
    parser.add_argument('--record',      type=int,    default=1,                       help='number of generations between samples')
    parser.add_argument('--pop_size_a',  type=int,    default=0,                       help='the assumed population size')
    parser.add_argument('--cutoff',      type=float,  default=0.01,                    help='the cutoff frequency. mutations that never rise above this frequency will not be plotted.')
    parser.add_argument('--ss_record',   type=str,    default=None,                    help='.npy file containing the simulation specific record')
    parser.add_argument('--ss_pop_size', type=str,    default=None,                    help='.npy file containing the simulation specific pop size')
    parser.add_argument('--s_ylims',     type=float,  default=None,                    help='limits on y axis of selection plot')
    parser.add_argument('--replicates',  type=str,    default=None,                    help='.npz file containing the replicate simulations')
    parser.add_argument('--replicates2', type=str,    default=None,                    help='.npz file containing different replicate simulations to compare')
    parser.add_argument('--replicates3', type=str,    default=None,                    help='.npz file containing different replicate simulations to compare')
    parser.add_argument('--rep_folder',  type=str,    default=None,                    help='folder containing folder of pairs of the different replicate simulations')
    parser.add_argument('--rep_perfect', type=str,    default=None,                    help='folder containing folder of pairs of replicates with perfect sampling')
    parser.add_argument('--auroc_dir',   type=str,    default=None,                    help='folder containing simulations to compare the AUROC scores for')
    parser.add_argument('--save_file',   type=str,    default='inference1',            help='path to save figure')
    parser.add_argument('--pop_plot_off',  action='store_true',  default=False,  help='whether or not to plot the populations')
    parser.add_argument('--freq_plot_off', action='store_true',  default=False,  help='whether or not to plot the frequencies')
    parser.add_argument('--inf_plot_off',  action='store_true',  default=False,  help='whether or not to plot the inferred coefficients')
    parser.add_argument('--err_plot_off',  action='store_true',  default=False,  help='whether or not to plot the error bars')
    parser.add_argument('--ind_plot',      action='store_true',  default=False,  help='whether or not to plot the coefficients inferred from the individual simulations')
    parser.add_argument('--comb_plot_off', action='store_true',  default=False,  help='whether or not to plot the combined coefficients, trajectories, and histograms on the same plot')
    
    
    arg_list = parser.parse_args(args)
    
    cutoff  = arg_list.cutoff
    pop_plot_off  = arg_list.pop_plot_off
    freq_plot_off = arg_list.freq_plot_off
    inf_plot_off  = arg_list.inf_plot_off
    err_plot_off  = arg_list.err_plot_off
    ind_plot      = arg_list.ind_plot
    comb_plot_off = arg_list.comb_plot_off
    s_ylims       = arg_list.s_ylims
    rep_folder    = arg_list.rep_folder
    rep_perfect   = arg_list.rep_perfect
    auroc_dir     = arg_list.auroc_dir
    save_file     = arg_list.save_file
    if arg_list.sim_data:
        sim_data = np.load(arg_list.sim_data, allow_pickle=True)
        simulations      = sim_data['simulations']
        actual           = sim_data['selection_all']
        pop_size         = sim_data['pop_size']
        traj             = sim_data['traj_record']
        mutant_sites     = sim_data['mutant_sites']
        mutant_sites_all = sim_data['mutant_sites_all']
    else:
        simulations      = 1
        pop_size         = 0
        mutant_sites_all = []
        traj             = []
        mutant_sites     = []
    if arg_list.inf_data:
        inf_data = np.load(arg_list.inf_data, allow_pickle=True)
        inferred = inf_data['selection']
        error    = inf_data['error_bars']
        inferred_ind = inf_data['selection_ind']
        error_ind    = inf_data['errors_ind']
    
    if arg_list.ss_record:
        record = np.load(arg_list.ss_record)
    else:
        record = arg_list.record * np.ones(simulations)
    if arg_list.ss_pop_size:
        if len(np.shape(np.load(arg_list.ss_pop_size))) == 2:
            pop_size_assumed = np.load(arg_list.ss_pop_size) # The guessed population size
        elif len(np.shape(np.load(arg_list.ss_pop_size))) == 1:
            pop_size_assumed = np.swapaxes(np.load(arg_list.ss_pop_size) * np.ones((10000, simulations)), 0, 1)
    elif arg_list.pop_size_a == 0:
        pop_size_assumed = pop_size 
    else:
        pop_size_assumed = [np.ones(len(traj[i])+1) * arg_list.pop_size_a for i in range(simulations)] 
    
    selection = {}
    for i in range(len(mutant_sites_all)):
        selection[mutant_sites_all[i]] = actual[i]

   
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
        for i in range(L_traj):
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
        
    
    # reorganize arrays and filter low frequency trajectories
    traj = trajectory_reshape(traj)
    
    allele_number = find_significant_sites(traj, mutant_sites, cutoff)
    actual = np.array([selection[i] for i in allele_number])
    L_traj = len(allele_number)
    if len(actual) != 0:
        s_value = np.amax(np.array(actual))
    else:
        s_value = 0
    
    traj, inferred, error, mutant_sites = filter_sites(traj, error, inferred, mutant_sites, mutant_sites_all, allele_number)
    mutant_sites_all = np.sort(np.unique([mutant_sites[j][i] for j in range(len(mutant_sites)) for i in range(len(mutant_sites[j]))]))
    
    if ind_plot:
        inferred_ind, error_ind = filter_sites_ind(inferred_ind, error_ind)
    
    # plot the populations
    if not pop_plot_off:
        plt.figure(figsize=[18,3])
        plt.suptitle("population size")
        for i in range(simulations):
            plt.subplot(1, int(simulations), i+1)
            plt.title(f"simulation {i+1}")
            sns.lineplot(np.arange(len(pop_size[i]))*record[i], pop_size[i], label="actual")
            sns.lineplot(np.arange(len(pop_size[i]))*record[i], pop_size_assumed[i][:len(pop_size[i])], label="guessed")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    """
    # plot the selection coefficients
    if not inf_plot_off:
        plt.figure(figsize=[18,5])
        plt.title("selection coefficients")
        plt.grid(b=True, axis='y', linewidth=0.5)
        plt.xticks(np.arange(L_traj), allele_number)
        if s_ylims:
            plt.ylim(-s_ylims, s_ylims)
        sns.scatterplot(np.arange(L_traj), inferred, label="inferred")
        if not err_plot_off:
            plt.errorbar(np.arange(L_traj), inferred, yerr=error, lw=0, elinewidth=1)
        if ind_plot:
            for i in range(simulations):
                sns.scatterplot(np.arange(L_traj), inferred_ind[i], label=f"sim {i}")
                if not err_plot_off:
                    plt.errorbar(np.arange(L_traj), inferred_ind[i], yerr=error_ind[i], lw=0, elinewidth=2)
        sns.scatterplot(np.arange(L_traj), actual, label="actual")
            
    # plot the frequencies
    if not freq_plot_off:
        plt.figure(figsize=[18, L_traj])
        if L_traj % 2 == 0:
            length = int(L_traj/2)
        else:
            length = int(L_traj/2 + 1)
        for i in range(length):
            for j in range(2):
                if 2*i + j in range(int(L_traj)):
                    plt.subplot(int(L_traj/2)+1,2,2*i+j+1)
                    plt.ylim(0,1)
                    for k in range(simulations):
                        if allele_number[2*i+j] in mutant_sites[k]:
                            loc = np.where(np.array(mutant_sites[k])==allele_number[2*i+j])[0][0]
                            sns.lineplot(np.arange(len(traj[k][loc]))*record[k], traj[k][loc], 
                                         label=f"allele {allele_number[2*i+j]}" + f" simulation {k}")
                        else:
                            sns.lineplot(np.arange(len(traj[k][0]))*record[k], np.zeros(len(traj[k][0])))
        plt.tight_layout()
        
    """
 
    # plot combining the frequencies and the selection coefficients
    """
    if not inf_plot_off and not freq_plot_off:
        s_ylims = np.amax(np.absolute(inferred)) + 0.01
        plt.figure(figsize=[18, len(inferred)])
        grid = plt.GridSpec(len(inferred), 2)
        plt.subplot(grid[:, 1])
        plt.title("selection coefficients")
        plt.grid(b=True, axis='x', linewidth=0.5)
        plt.yticks(np.arange(len(inferred)))
        plt.xlim(-s_ylims, s_ylims)
        plt.ylim(-0.5, len(inferred) - 0.75)
        sns.scatterplot(inferred[::-1], np.arange(len(inferred)), label="inferred", color='k')
        if not err_plot_off:
            plt.errorbar(inferred[::-1], np.arange(len(inferred)), xerr=error[::-1], lw=0, elinewidth=1, color='k')
        sns.scatterplot(actual[::-1], np.arange(len(actual)), label="actual", color='r')
        
        for i in np.arange(len(inferred)):
            plt.subplot(grid[i, 0])
            if i == 0:
                plt.title("frequency trajectories")
            plt.ylim(0,1)
            plt.yticks([])
            plt.xticks([])
            for k in range(simulations):
                if allele_number[i] in mutant_sites[k]:
                    loc = np.where(np.array(mutant_sites[k])==allele_number[i])[0][0]
                    sns.lineplot(np.arange(len(traj[k][loc]))*record[k], traj[k][loc])
                else:
                    sns.lineplot(np.arange(len(traj[0]))*record[k], np.zeros(len(traj[0])))
    """
    #def math_formatter(x, pos):
    #    return "${}$".format(x).replace.("-", u"\u2212")    
    cm_to_inch = lambda x: x/2.54    
    # plot combining the frequencies, the selection coefficients, and the histograms
    indices       = np.argsort(actual)[::-1]
    actual        = np.array(actual)[indices]
    inferred      = inferred[indices]
    colors_inf    = np.array([sort_color2(i) for i in actual], dtype=str)
    allele_number = allele_number[indices]
    # Finding the different sets of actual coefficients that are all the same
    actual_unique  = np.unique(actual)
    unique_lengths = [len(np.isin(actual, i).nonzero()[0]) for i in actual_unique]
    colors_unique  = np.array([sort_color2(i) for i in actual_unique], dtype=str)
    if not comb_plot_off:
        """
        # If you want to use these better latex fonts, you will need to wrap all ticks and text with \textbf{}
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        """
        s_ylims = np.amax(np.absolute(inferred)) + 0.01
        fig  = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, len(inferred)/2])
        #grid = matplotlib.gridspec.GridSpec(len(inferred), 4, wspace=0.3, figure = fig)
        grid = matplotlib.gridspec.GridSpec(1, 3, wspace=0.4, figure=fig, left=0.15, right=0.95, bottom=0.15, top=0.95)
        left_grid  = grid[0, 0:2].subgridspec(1, 2, wspace=0.05)
        right_grid = grid[0, 2:]
        fig.text(0.13,   0.985, 'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
        fig.text(0.405,  0.985, 'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
        fig.text(0.705,  0.985, 'c', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
        #fig.text(0.775,  0.825, 'd', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
        
        # plot the selection coefficients
        ax_s = fig.add_subplot(left_grid[0, 1])
        xtick_labels = [''] + [i for i in np.linspace(-int(100*s_ylims), int(100*s_ylims), 5, endpoint=True)[1:]]
        ax_s.set_xticks(np.linspace(-s_ylims, s_ylims, 5, endpoint=True))
        ax_s.set_xticklabels([f'{i}'.replace('-', '\N{MINUS SIGN}') for i in xtick_labels], fontsize=6)
        ax_s.set_yticks([])
        ax_s.set_xlabel('Inferred selection\n coefficients (%)', fontsize=AXES_FONTSIZE)
        ax_s.set_xlim(-s_ylims, s_ylims)
        ax_s.set_ylim(-0.5, len(inferred) - 0.75)
        ax_s.scatter(inferred[::-1], np.arange(len(inferred)), c=colors_inf[::-1], s=5)
        ax_s.tick_params(width=SPINE_LW)
        for line in ['top', 'bottom', 'left', 'right']:
                ax_s.spines[line].set_linewidth(SPINE_LW) 
        if not err_plot_off:
            ax_s.errorbar(inferred[::-1], np.arange(len(inferred)), xerr=error[::-1], lw=0, elinewidth=1, ecolor=colors_inf[::-1])
        for i in range(len(actual_unique)):
            ax_s.axvline(x=actual_unique[i], ymin=np.sum(unique_lengths[:i])/len(actual), ymax=np.sum(unique_lengths[:i+1])/len(actual),
                        lw=1, ls='--', c=colors_unique[i])
        #ax_s.text(0, 1.1, 'b',transform=ax_s.transAxes, ha='center', va='center', fontweight='bold')
        # make the legend
        dashed_line = matplotlib.lines.Line2D([], [], color='k', label='True coefficient', linestyle='dashed')
        red_patch   = matplotlib.patches.Patch(color='red',  label='Beneficial')
        blue_patch  = matplotlib.patches.Patch(color='blue', label='Deleterious')
        grey_patch  = matplotlib.patches.Patch(color='grey', label='Neutral')
        ax_s.legend(handles=[dashed_line, red_patch, grey_patch, blue_patch], fontsize=6, framealpha=0.5)
        
        
        # plot the trajectories
        ax_traj = fig.add_subplot(left_grid[0, 0])
        ax_traj.set_ylabel("Frequency", fontsize=AXES_FONTSIZE)
        ax_traj.set_xticks([])
        ax_traj.set_yticks([])
        ax_traj.spines['top'].set_visible(False)
        ax_traj.spines['right'].set_visible(False)
        ax_traj.spines['bottom'].set_visible(False)
        ax_traj.spines['left'].set_visible(False)
        freq_grid = left_grid[0, 0].subgridspec(len(inferred), 1, hspace=0.1)
        for i in range(len(inferred)):
            ax = fig.add_subplot(freq_grid[i, 0])
            ax.tick_params(width=SPINE_LW)
            for line in ['top', 'bottom', 'left', 'right']:
                ax.spines[line].set_linewidth(SPINE_LW)
            ax.set_ylim(0,1)
            ax.set_yticks([])
            if i == len(inferred) - 1:
                ax.set_xlabel("Generations", fontsize=AXES_FONTSIZE)
                ax.tick_params(axis='x', labelsize=TICK_FONTSIZE, width=SPINE_LW)
            else:
                ax.set_xticks([])
            #if i == 0:
                #ax.text(0, 1.65, 'a',transform=ax.transAxes, ha='center', va='center', fontweight='bold')
            for k in range(simulations):
                if allele_number[i] in mutant_sites[k]:
                    loc = np.where(np.array(mutant_sites[k])==allele_number[i])[0][0]    
                    sns.lineplot(np.arange(len(traj[k][loc]))*record[k], traj[k][loc], color=colors_inf[i], lw=FREQ_LW, ax=ax)
                else:
                    sns.lineplot(np.arange(len(traj[0]))*record[k], np.zeros(len(traj[0])), ax=ax, lw=FREQ_LW)
                    
        # plot histograms            
        rep_data     = np.load(arg_list.replicates, allow_pickle=True)
        positive, negative, zero = [], [], []
        for sim in range(len(rep_data['inferred'])):
            traj     = trajectory_reshape(rep_data['traj'][sim])
            #traj = rep_data['traj'][sim]
            inferred = rep_data['inferred'][sim]
            actual   = rep_data['actual'][sim]
            mutants  = rep_data['mutant_sites'][sim]
            error    = rep_data['errors'][sim]
            mutant_sites_all = np.sort(np.unique(np.array([mutants[i][j] for i in range(len(mutants))
                                                   for j in range(len(mutants[i]))])))
            selection= {}
            for i in range(len(mutant_sites_all)):
                selection[mutant_sites_all[i]] = actual[i]
            allele_number = find_significant_sites(traj, mutants, cutoff)
            L_traj   = len(allele_number)
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
        hist_pos      = {"weights" : np.ones(len(positive)) / len(positive)}
        hist_neg      = {"weights" : np.ones(len(negative)) / len(negative)}
        hist_zero     = {"weights" : np.ones(len(zero)) / len(zero)}
        inf_max       = np.amax([np.amax(i) for i in inferred])
        ax_h          = fig.add_subplot(right_grid)
        #ax_h.set_ylabel('Inferred selection coefficients (%)', fontsize='x-small', labelpad=-35)
        ax_h.text(-0.2, 0.5, "Inferred selection coefficients (%)", ha='center', va='center', fontsize=AXES_FONTSIZE, transform=ax_h.transAxes, rotation='vertical')
        ax_h.set_xlabel('Frequency', fontsize=AXES_FONTSIZE)
        ax_h.axhline(y=-s_value, color='b')
        ax_h.axhline(y=0, color=mcolors.CSS4_COLORS['grey'])
        ax_h.axhline(y=s_value, color='r')
        hist_max = 0.1
        ax_h.set_xlim(0, hist_max)
        ax_h.set_xticks([0, hist_max])
        ax_h.set_xticklabels([0, hist_max], fontsize=6)
        sns.distplot(positive, kde=False, bins=50, color="r", label="beneficial", hist_kws=hist_pos, vertical=True, ax=ax_h)
        sns.distplot(zero, kde=False, bins=50, color=mcolors.CSS4_COLORS['grey'], label="neutral", hist_kws=hist_zero, vertical=True, ax=ax_h)
        sns.distplot(negative, kde=False, bins=50, color='b', label="deleterious", hist_kws=hist_neg, vertical=True, ax=ax_h)
        all_inferred = rep_data['inferred']
        s_max = (2/3)*np.amax([np.amax(np.absolute(i)) for i in all_inferred])
        ax_h.set_ylim(-s_max, s_max)
        ax_h.set_yticks(np.linspace(-s_max, s_max, 5))
        ax_h.set_yticklabels([f'{i}'.replace('-', '\N{MINUS SIGN}') for i in np.linspace(-int(s_max*100), int(s_max*100), 5, dtype=int)], fontsize=6)
        ax_h.yaxis.set_label_coords(-0.225, 0.5)
        ax_h.tick_params(width=SPINE_LW)
        for line in ['top', 'bottom', 'left', 'right']:
                ax_h.spines[line].set_linewidth(SPINE_LW)
        # move histogram tick labels closer
        offset = matplotlib.transforms.ScaledTranslation(3/72, 0, fig.dpi_scale_trans)
        for tick in ax_h.yaxis.get_majorticklabels():
            tick.set_transform(tick.get_transform() + offset)
        #ax_h.text(-0.1, 1.1, 'c',transform=ax_h.transAxes, ha='center', va='center', fontweight='bold')
        
        # plot AUROC's
        #ben_auroc, del_auroc = calculate_AUROC_av(rep_data['actual'], rep_data['inferred'])
        """
        auroc_grid = right_grid[0, 1].subgridspec(2, 1, hspace=0.4)
        ax_ben = fig.add_subplot(auroc_grid[0, 0])
        ax_ben.axis('off')
        #ax_ben.set_xticks([])
        #ax_ben.set_yticks([])
        #ax_ben.set_xlabel("AUROC\nbeneficial")
        ax_ben.text(0.5, 0.5, str(ben_auroc)[:5], transform=ax_ben.transAxes, ha='center', va='center')
        ax_del = fig.add_subplot(auroc_grid[1, 0])
        ax_del.axis('off')
        #ax_del.set_xticks([])
        #ax_del.set_yticks([])
        #ax_del.set_xlabel("AUROC\ndeleterious")
        ax_del.text(0.5, 0.5, str(del_auroc)[:5], transform=ax_del.transAxes, ha='center', va='center')
        """
        #ben_str = str(ben_auroc)[:5]
        #del_str = str(del_auroc)[:5]
        #ax_auc = fig.add_subplot(right_grid[0, 1])
        #ax_auc.axis('off')
        #ax_auc.text(0.1, 0.9, "d", transform=ax_auc.transAxes, ha='center', va='center', fontweight='bold')
        #ax_auc.text(0.5, 0.75, "AUROC", transform=ax_auc.transAxes, ha='center', va='center', fontweight='bold')
        #ax_auc.text(0.5, 0.5, "beneficial\n"  + ben_str, transform=ax_auc.transAxes, ha='center', va='center', fontsize='small')
        #ax_auc.text(0.5, 0.25, "deleterious\n" + del_str, transform=ax_auc.transAxes, ha='center', va='center', fontsize='small')
        
        plt.gcf().subplots_adjust(bottom=0.25)
        plt.savefig(os.path.join(fig_path, save_file+'.png'), dpi=1200)
        
    
        
    n_subplots = 2
    if arg_list.replicates and arg_list.replicates2 and not arg_list.replicates3:
        fig, axs = plt.subplots(n_subplots, 1, figsize=[cm_to_inch(8.8),10])
        plot_histogram_object_oriented(arg_list.replicates, 1, cutoff, n_subplots, axs[0])
        plot_histogram_object_oriented(arg_list.replicates2, 2, cutoff, n_subplots, axs[1])
        
    if (not arg_list.replicates) and arg_list.replicates2 and arg_list.replicates3:
        fig, axs = plt.subplots(n_subplots, 1, figsize=[8,10])
        plot_histogram_object_oriented(arg_list.replicates2, 1, cutoff, n_subplots, axs[0], title='true parameters')
        plot_histogram_object_oriented(arg_list.replicates3, 2, cutoff, n_subplots, axs[1], title='consant parameters')
        
    if arg_list.replicates and arg_list.replicates2 and arg_list.replicates3:
        n_subplots = 3
        fig, axs = plt.subplots(n_subplots, 1, figsize=[8,15])
        plot_histogram_object_oriented(arg_list.replicates, 1, cutoff, n_subplots, axs[0])
        plot_histogram_object_oriented(arg_list.replicates2, 2, cutoff, n_subplots, axs[1])
        plot_histogram_object_oriented(arg_list.replicates3, 3, cutoff, n_subplots, axs[2])
        
    # create a table that shows the inference for different populations given a directory containing folders
    # each of which contains the two simulations (replicated some number of times) and the population size used
    if rep_folder:
        #color_map = plt.get_cmap('Blues')
        color_map = palet.colorbrewer.sequential.Blues_4.get_mpl_colormap()
        n_pops  = len([name for name in os.listdir(rep_folder) if (os.path.isdir(os.path.join(rep_folder, name)) and name!='.ipynb_checkpoints')]) # number of different populations to be tested
        fig = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, n_pops])
        grid = matplotlib.gridspec.GridSpec(n_pops*4+2, 5, figure=fig, wspace=0, hspace=0)
        axes_list = []
        #print(n_pops)
        ax_titles0 = fig.add_subplot(grid[0:2, 0])
        ax_titles0.text(0.5, 0.5, "Population\nSize", transform=ax_titles0.transAxes, ha='center', va='center', fontweight='bold', fontsize=AXES_FONTSIZE)
        ax_titles0.spines['right'].set_linewidth(SPINE_LW)
        ax_titles0.set_xticks([])
        ax_titles0.set_yticks([])
        ax_titles1 = fig.add_subplot(grid[0:2, 1])
        ax_titles1.text(0.5, 0.5, "Sampling", transform=ax_titles1.transAxes, ha='center', va='center', fontweight='bold', fontsize=AXES_FONTSIZE)
        axes_list.append(ax_titles1)
        ax_titles2 = fig.add_subplot(grid[0:2, 2])
        ax_titles2.text(0.5, 0.5, "Inference\nParameter (N)", transform=ax_titles2.transAxes, ha='center', va='center', fontweight='bold', fontsize=AXES_FONTSIZE)
        axes_list.append(ax_titles2)
        ax_titles3 = fig.add_subplot(grid[0:2, 3])
        ax_titles3.text(0.5, 0.5, "AUROC\nBeneficial", transform=ax_titles3.transAxes, ha='center', va='center', fontweight='bold', fontsize=AXES_FONTSIZE)
        axes_list.append(ax_titles3)
        ax_titles4 = fig.add_subplot(grid[0:2, 4])
        ax_titles4.text(0.5, 0.5, "AUROC\nDeleterious", transform=ax_titles4.transAxes, ha='center', va='center', fontweight='bold', fontsize=AXES_FONTSIZE)
        ax_titles4.spines['left'].set_linewidth(SPINE_LW)
        ax_titles4.set_xticks([])
        ax_titles4.set_yticks([])
        
        i = 1
        for folder in sorted(os.listdir(os.fsencode(rep_folder))):
            tv_file, const_file, pop_file = None, None, None
            folder_name = os.fsdecode(folder)
            folder_path = os.path.join(rep_folder, folder_name)
            if os.path.isdir(folder_path) and folder_name!='.ipynb_checkpoints':
                #print(folder_name, '(directory found)')
                for file in sorted(os.listdir(folder_path)):
                    if os.fsdecode(file)[:10]=='population':
                        pop_file   = np.load(os.path.join(rep_folder, folder_name, os.fsdecode(file)), allow_pickle=True)
                    elif os.fsdecode(file)[-6:]=='tv.npz':
                        tv_file    = np.load(os.path.join(rep_folder, folder_name, os.fsdecode(file)), allow_pickle=True)
                    elif os.fsdecode(file)[-9:]=='const.npz':
                        const_file = np.load(os.path.join(rep_folder, folder_name, os.fsdecode(file)), allow_pickle=True)
                    else:
                        print('unknown file')
                    #print(file)
                # calculate AUROC
                folder_path2 = os.path.join(rep_perfect, folder_name)
                tv_perfect, const_perfect = None, None
                for file in sorted(os.listdir(folder_path2)):
                    if os.fsdecode(file)[-6:]=='tv.npz':
                        tv_perfect    = np.load(os.path.join(rep_perfect, folder_name, os.fsdecode(file)), allow_pickle=True)
                    elif os.fsdecode(file)[-9:]=='const.npz':
                        const_perfect = np.load(os.path.join(rep_perfect, folder_name, os.fsdecode(file)), allow_pickle=True)
                
                AUC_ben_tv,    AUC_del_tv    = calculate_AUROC(tv_file['actual'],    tv_file['inferred'])
                AUC_ben_const, AUC_del_const = calculate_AUROC(const_file['actual'], const_file['inferred'])
                
                AUC_per_ben_tv, AUC_per_del_tv       = calculate_AUROC(tv_perfect['actual'],    tv_perfect['inferred'])
                AUC_per_ben_const, AUC_per_del_const = calculate_AUROC(const_perfect['actual'], const_perfect['inferred'])
                
                # can use the below to color code the squares based on which of tv or const provides the better estimate
                # uncomment the code in the plotting section to instead shade them according to a color map and the AUROC value
                br = ['cornflowerblue', 'lightcoral']
                if AUC_ben_tv > AUC_ben_const:
                    c_ben_finite = br
                else:
                    c_ben_finite = br[::-1]
                if AUC_del_tv > AUC_del_const:
                    c_del_finite = br
                else:
                    c_del_finite = br[::-1]
                if AUC_per_ben_tv > AUC_per_ben_const:
                    c_ben_perfect = br
                else:
                    c_ben_perfect = br[::-1]
                if AUC_per_del_tv > AUC_per_del_const:
                    c_del_perfect = br
                else:
                    c_del_perfect = br[::-1]
                
            
                # make plots
                ax0 = fig.add_subplot(grid[4*i-2:4*i+2, 0])
                sns.lineplot(np.arange(len(pop_file)), pop_file, lw=0.5, ax=ax0)
                ax0.spines['right'].set_linewidth(SPINE_LW)
                ax0.set_xticks([])
                ax0.set_yticks([])
                
                ax1a = fig.add_subplot(grid[4*i-2:4*i, 1])
                ax1a.text(0.5, 0.5, 'Finite', transform=ax1a.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
                axes_list.append(ax1a)
                
                ax1b = fig.add_subplot(grid[4*i:4*i+2, 1])
                ax1b.text(0.5, 0.5, 'Perfect', transform=ax1b.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
                axes_list.append(ax1b)
                
                ax2a = fig.add_subplot(grid[4*i-2, 2])
                ax2a.text(0.5, 0.5, 'Time-Varying', transform=ax2a.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
                if AUC_ben_tv > AUC_ben_const and AUC_del_tv > AUC_del_const:
                    ax2a.patch.set_facecolor(c_ben_finite[1])
                elif AUC_ben_tv < AUC_ben_const and AUC_del_tv < AUC_del_const:
                    ax2a.patch.set_facecolor(c_ben_finite[0])
                ax2a.spines['bottom'].set_linewidth(SPINE_LW)
                axes_list.append(ax2a)
                
                ax2b = fig.add_subplot(grid[4*i-1, 2])
                ax2b.text(0.5, 0.5, 'Constant', transform=ax2b.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
                if AUC_ben_tv > AUC_ben_const and AUC_del_tv > AUC_del_const:
                    ax2b.patch.set_facecolor(c_ben_finite[0])
                elif AUC_ben_tv < AUC_ben_const and AUC_del_tv < AUC_del_const:
                    ax2b.patch.set_facecolor(c_ben_finite[1])
                ax2b.spines['top'].set_linewidth(SPINE_LW)
                axes_list.append(ax2b)
                
                ax2c = fig.add_subplot(grid[4*i, 2])
                ax2c.text(0.5, 0.5, 'Time-Varying', transform=ax2c.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
                if AUC_per_ben_tv > AUC_per_ben_const and AUC_per_del_tv > AUC_per_del_const:
                    ax2c.patch.set_facecolor(c_ben_perfect[0])
                elif AUC_per_ben_tv < AUC_per_ben_const and AUC_per_del_tv < AUC_per_del_const:
                    ax2c.patch.set_facecolor(c_ben_perfect[1])
                ax2c.spines['bottom'].set_linewidth(SPINE_LW)
                axes_list.append(ax2c)
                
                ax2d = fig.add_subplot(grid[4*i+1, 2])
                ax2d.text(0.5, 0.5, 'Constant', transform=ax2d.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
                if AUC_per_ben_tv > AUC_per_ben_const and AUC_per_del_tv > AUC_per_del_const:
                    ax2d.patch.set_facecolor(c_ben_perfect[1])
                elif AUC_per_ben_tv < AUC_per_ben_const and AUC_per_del_tv < AUC_per_del_const:
                    ax2d.patch.set_facecolor(c_ben_perfect[0])
                ax2d.spines['top'].set_linewidth(SPINE_LW)
                axes_list.append(ax2d)
                
                ax3a = fig.add_subplot(grid[4*i-2, 3])
                #ax3a.patch.set_facecolor(color_map(AUC_ben_tv))
                ax3a.patch.set_facecolor(c_ben_finite[0])
                ax3a.text(0.5, 0.5, str(AUC_ben_tv)[:5], transform=ax3a.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
                ax3a.spines['bottom'].set_linewidth(SPINE_LW)
                axes_list.append(ax3a)
                                         
                ax3b = fig.add_subplot(grid[4*i-1, 3])
                #ax3b.patch.set_facecolor(color_map(AUC_ben_const))
                ax3b.patch.set_facecolor(c_ben_finite[1])
                ax3b.text(0.5, 0.5, str(AUC_ben_const)[:5], transform=ax3b.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE) 
                ax3b.spines['top'].set_linewidth(SPINE_LW)
                axes_list.append(ax3b)
                
                ax3c = fig.add_subplot(grid[4*i, 3])
                #ax3c.patch.set_facecolor(color_map(AUC_per_ben_tv))
                ax3c.patch.set_facecolor(c_ben_perfect[0])
                ax3c.text(0.5, 0.5, str(AUC_per_ben_tv)[:5], transform=ax3c.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
                ax3c.spines['bottom'].set_linewidth(SPINE_LW)
                axes_list.append(ax3c)
                
                ax3d = fig.add_subplot(grid[4*i+1, 3])
                #ax3d.patch.set_facecolor(color_map(AUC_per_ben_tv))
                ax3d.patch.set_facecolor(c_ben_perfect[1])
                ax3d.text(0.5, 0.5, str(AUC_per_ben_const)[:5], transform=ax3d.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
                ax3d.spines['top'].set_linewidth(SPINE_LW)
                axes_list.append(ax3d)
               
                ax4a = fig.add_subplot(grid[4*i-2, 4])
                #ax4a.patch.set_facecolor(color_map(AUC_del_tv))
                ax4a.patch.set_facecolor(c_del_finite[0])
                ax4a.text(0.5, 0.5, str(AUC_del_tv)[:5], transform=ax4a.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE) 
                ax4a.spines['bottom'].set_linewidth(SPINE_LW)
                ax4a.spines['left'].set_linewidth(SPINE_LW)
                ax4a.set_xticks([])
                ax4a.set_yticks([])
            
                ax4b = fig.add_subplot(grid[4*i-1, 4])
                #ax4b.patch.set_facecolor(color_map(AUC_del_const))
                ax4b.patch.set_facecolor(c_del_finite[1])
                ax4b.text(0.5, 0.5, str(AUC_del_const)[:5], transform=ax4b.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE) 
                ax4b.spines['top'].set_linewidth(SPINE_LW)
                ax4b.spines['left'].set_linewidth(SPINE_LW)
                ax4b.set_xticks([])
                ax4b.set_yticks([])
                
                ax4c = fig.add_subplot(grid[4*i, 4])
                #ax4c.patch.set_facecolor(color_map(AUC_per_del_tv))
                ax4c.patch.set_facecolor(c_del_perfect[0])
                ax4c.text(0.5, 0.5, str(AUC_per_del_tv)[:5], transform=ax4c.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE) 
                ax4c.spines['bottom'].set_linewidth(SPINE_LW)
                ax4c.spines['left'].set_linewidth(SPINE_LW)
                ax4c.set_xticks([])
                ax4c.set_yticks([])
            
                ax4d = fig.add_subplot(grid[4*i+1, 4])
                #ax4d.patch.set_facecolor(color_map(AUC_per_del_const))
                ax4d.patch.set_facecolor(c_del_perfect[1])
                ax4d.text(0.5, 0.5, str(AUC_per_del_const)[:5], transform=ax4d.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE) 
                ax4d.spines['top'].set_linewidth(SPINE_LW)
                ax4d.spines['left'].set_linewidth(SPINE_LW)
                ax4d.set_xticks([])
                ax4d.set_yticks([])
    
                i+=1
        
        for axis in axes_list:
            for line in ['right', 'left']:
                axis.spines[line].set_linewidth(SPINE_LW)
            axis.set_yticks([])
            axis.set_xticks([])
        plt.gcf().subplots_adjust(bottom=0.01)
        plt.gcf().subplots_adjust(top=0.99)
        plt.gcf().subplots_adjust(left=0.01)
        plt.gcf().subplots_adjust(right=0.99)
        plt.savefig(os.path.join(fig_path, 'finite-sampling.png'), dpi=1200)
            
            
    
    # Compare the AUROC scores for simulations with different parameters
    ### Decide on two of the following three: sample size, length of time-series, number of simulations ###
    if auroc_dir != None:
        for file in os.listdir(auroc_dir):
            filename = os.fsdecode(file)
            filpath = os.path.join(auroc_dir, filename)
                    
                    
        
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    

