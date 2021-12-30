#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Check what the above means

import sys
import numpy as np                          # numerical tools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import matplotlib.cm as cm
import seaborn as sns
from tabulate import tabulate
import os
import palettable as palet
import datetime as dt
import colorsys
from colorsys import hls_to_rgb, hsv_to_rgb
import pandas as pd
import mplot as mp
from scipy import stats as spstats


NUC = ['-', 'A', 'C', 'G', 'T']

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
palette1            = sns.hls_palette(2)
MAIN_COLOR          = palette1[0]
COMP_COLOR          = palette1[1]
SIZELINE            = 0.6
AXWIDTH             = 0.4
DPI                 = 1200
SMALLSIZEDOT        = 6.
SPINE_LW            = 0.25
FREQ_LW             = 1
AXES_FONTSIZE       = 6
TICK_FONTSIZE       = 6

fig_path            = 'figures'
image_path          = 'images'


# load data_processing module
cwd = os.getcwd()
import data_processing as dp


def calculate_linked_coefficients(inf_file, link_file, tv=False):
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



def finite_sampling_plot(rep_folder, rep_perfect):
    """Compares inference using real or time-varying parameters 
    under the conditions of finite or perfect sampling for various population sizes. """
    color_map = palet.colorbrewer.sequential.Blues_4.get_mpl_colormap()
    n_pops  = len([name for name in os.listdir(rep_folder) if (os.path.isdir(os.path.join(rep_folder, name)) and name!='.ipynb_checkpoints')]) # number of different populations to be tested
    fig = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, n_pops])
    grid = matplotlib.gridspec.GridSpec(n_pops*4+2, 5, figure=fig, wspace=0, hspace=0)
    axes_list = []
    ax_titles0 = fig.add_subplot(grid[0:2, 0])
    ax_titles0.text(0.5, 0.5, "Population\nSize", transform=ax_titles0.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
    ax_titles0.spines['right'].set_linewidth(SPINE_LW)
    ax_titles0.set_xticks([])
    ax_titles0.set_yticks([])
    ax_titles1 = fig.add_subplot(grid[0:2, 1])
    ax_titles1.text(0.5, 0.5, "Sampling", transform=ax_titles1.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
    axes_list.append(ax_titles1)
    ax_titles2 = fig.add_subplot(grid[0:2, 2])
    ax_titles2.text(0.5, 0.5, "Inference\nParameter (N)", transform=ax_titles2.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
    axes_list.append(ax_titles2)
    ax_titles3 = fig.add_subplot(grid[0:2, 3])
    ax_titles3.text(0.5, 0.5, "AUROC\nBeneficial", transform=ax_titles3.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
    axes_list.append(ax_titles3)
    ax_titles4 = fig.add_subplot(grid[0:2, 4])
    ax_titles4.text(0.5, 0.5, "AUROC\nDeleterious", transform=ax_titles4.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
    ax_titles4.spines['left'].set_linewidth(SPINE_LW)
    ax_titles4.set_xticks([])
    ax_titles4.set_yticks([])
        
    i = 1
    for folder in sorted(os.listdir(os.fsencode(rep_folder))):
        tv_file, const_file, pop_file = None, None, None
        folder_name = os.fsdecode(folder)
        folder_path = os.path.join(rep_folder, folder_name)
        if os.path.isdir(folder_path) and folder_name!='.ipynb_checkpoints':
            for file in sorted(os.listdir(folder_path)):
                if os.fsdecode(file)[:10]=='population':
                    pop_file   = np.load(os.path.join(rep_folder, folder_name, os.fsdecode(file)), allow_pickle=True)
                elif os.fsdecode(file)[-6:]=='tv.npz':
                    tv_file    = np.load(os.path.join(rep_folder, folder_name, os.fsdecode(file)), allow_pickle=True)
                elif os.fsdecode(file)[-9:]=='const.npz':
                    const_file = np.load(os.path.join(rep_folder, folder_name, os.fsdecode(file)), allow_pickle=True)
                else:
                    print('unknown file')
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
            """
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
            """
                
            
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
            #ax2a.text(0.5, 0.5, 'Time-Varying', transform=ax2a.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            if AUC_ben_tv > AUC_ben_const and AUC_del_tv > AUC_del_const:
                #ax2a.patch.set_facecolor(c_ben_finite[1])
                ax2a.text(0.5, 0.5, 'Time-Varying', transform=ax2a.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE, fontweight='bold')
            else:
                ax2a.text(0.5, 0.5, 'Time-Varying', transform=ax2a.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            #elif AUC_ben_tv < AUC_ben_const and AUC_del_tv < AUC_del_const:
            #    ax2a.patch.set_facecolor(c_ben_finite[0])
            ax2a.spines['bottom'].set_linewidth(SPINE_LW)
            axes_list.append(ax2a)
                
            ax2b = fig.add_subplot(grid[4*i-1, 2])
            #ax2b.text(0.5, 0.5, 'Constant', transform=ax2b.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            if AUC_ben_tv > AUC_ben_const and AUC_del_tv > AUC_del_const:
                #ax2b.patch.set_facecolor(c_ben_finite[0])
                ax2b.text(0.5, 0.5, 'Constant', transform=ax2b.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            else:
                ax2b.text(0.5, 0.5, 'Constant', transform=ax2b.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE, fontweight='bold')
            #elif AUC_ben_tv < AUC_ben_const and AUC_del_tv < AUC_del_const:
                #ax2b.patch.set_facecolor(c_ben_finite[1])
            ax2b.spines['top'].set_linewidth(SPINE_LW)
            axes_list.append(ax2b)
                
            ax2c = fig.add_subplot(grid[4*i, 2])
            #ax2c.text(0.5, 0.5, 'Time-Varying', transform=ax2c.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            if AUC_per_ben_tv > AUC_per_ben_const and AUC_per_del_tv > AUC_per_del_const:
                #ax2c.patch.set_facecolor(c_ben_perfect[0])
                ax2c.text(0.5, 0.5, 'Time-Varying', transform=ax2c.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE, fontweight='bold')
            else:
                ax2c.text(0.5, 0.5, 'Time-Varying', transform=ax2c.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            #elif AUC_per_ben_tv < AUC_per_ben_const and AUC_per_del_tv < AUC_per_del_const:
                #ax2c.patch.set_facecolor(c_ben_perfect[1])
            ax2c.spines['bottom'].set_linewidth(SPINE_LW)
            axes_list.append(ax2c)
                
            ax2d = fig.add_subplot(grid[4*i+1, 2])
            #ax2d.text(0.5, 0.5, 'Constant', transform=ax2d.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            if AUC_per_ben_tv > AUC_per_ben_const and AUC_per_del_tv > AUC_per_del_const:
                ax2d.text(0.5, 0.5, 'Constant', transform=ax2d.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            else:
                ax2d.text(0.5, 0.5, 'Constant', transform=ax2d.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE, fontweight='bold')
                #ax2d.patch.set_facecolor(c_ben_perfect[1])
            #elif AUC_per_ben_tv < AUC_per_ben_const and AUC_per_del_tv < AUC_per_del_const:
                #ax2d.patch.set_facecolor(c_ben_perfect[0])
            ax2d.spines['top'].set_linewidth(SPINE_LW)
            axes_list.append(ax2d)
                
            ax3a = fig.add_subplot(grid[4*i-2, 3])
            #ax3a.patch.set_facecolor(c_ben_finite[0])
            if AUC_ben_tv > AUC_ben_const and AUC_del_tv > AUC_del_const:
                ax3a.text(0.5, 0.5, str(AUC_ben_tv)[:5], transform=ax3a.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE, fontweight='bold')
            else:
                ax3a.text(0.5, 0.5, str(AUC_ben_tv)[:5], transform=ax3a.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            ax3a.spines['bottom'].set_linewidth(SPINE_LW)
            axes_list.append(ax3a)
                                         
            ax3b = fig.add_subplot(grid[4*i-1, 3])
            #ax3b.patch.set_facecolor(c_ben_finite[1])
            if AUC_ben_tv > AUC_ben_const and AUC_del_tv > AUC_del_const:
                ax3b.text(0.5, 0.5, str(AUC_ben_const)[:5], transform=ax3b.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE) 
            else:
                ax3b.text(0.5, 0.5, str(AUC_ben_const)[:5], transform=ax3b.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE, fontweight='bold') 
            ax3b.spines['top'].set_linewidth(SPINE_LW)
            axes_list.append(ax3b)
                
            ax3c = fig.add_subplot(grid[4*i, 3])
            #ax3c.patch.set_facecolor(c_ben_perfect[0])
            if AUC_per_ben_tv > AUC_per_ben_const and AUC_per_del_tv > AUC_per_del_const:
                ax3c.text(0.5, 0.5, str(AUC_per_ben_tv)[:5], transform=ax3c.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE, fontweight='bold')
            else:
                ax3c.text(0.5, 0.5, str(AUC_per_ben_tv)[:5], transform=ax3c.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            ax3c.spines['bottom'].set_linewidth(SPINE_LW)
            axes_list.append(ax3c)
                
            ax3d = fig.add_subplot(grid[4*i+1, 3])
            #ax3d.patch.set_facecolor(c_ben_perfect[1])
            if AUC_per_ben_tv > AUC_per_ben_const and AUC_per_del_tv > AUC_per_del_const:
                ax3d.text(0.5, 0.5, str(AUC_per_ben_const)[:5], transform=ax3d.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            else:
                ax3d.text(0.5, 0.5, str(AUC_per_ben_const)[:5], transform=ax3d.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE, fontweight='bold')
            ax3d.spines['top'].set_linewidth(SPINE_LW)
            axes_list.append(ax3d)
               
            ax4a = fig.add_subplot(grid[4*i-2, 4])
            #ax4a.patch.set_facecolor(c_del_finite[0])
            if AUC_ben_tv > AUC_ben_const and AUC_del_tv > AUC_del_const:
                ax4a.text(0.5, 0.5, str(AUC_del_tv)[:5], transform=ax4a.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE, fontweight='bold') 
            else:
                ax4a.text(0.5, 0.5, str(AUC_del_tv)[:5], transform=ax4a.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            ax4a.spines['bottom'].set_linewidth(SPINE_LW)
            ax4a.spines['left'].set_linewidth(SPINE_LW)
            ax4a.set_xticks([])
            ax4a.set_yticks([])
            
            ax4b = fig.add_subplot(grid[4*i-1, 4])
            #ax4b.patch.set_facecolor(c_del_finite[1])
            if AUC_ben_tv > AUC_ben_const and AUC_del_tv > AUC_del_const:
                ax4b.text(0.5, 0.5, str(AUC_del_const)[:5], transform=ax4b.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE) 
            else:
                ax4b.text(0.5, 0.5, str(AUC_del_const)[:5], transform=ax4b.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE, fontweight='bold') 
            ax4b.spines['top'].set_linewidth(SPINE_LW)
            ax4b.spines['left'].set_linewidth(SPINE_LW)
            ax4b.set_xticks([])
            ax4b.set_yticks([])
                
            ax4c = fig.add_subplot(grid[4*i, 4])
            #ax4c.patch.set_facecolor(c_del_perfect[0])
            if AUC_per_ben_tv > AUC_per_ben_const and AUC_per_del_tv > AUC_per_del_const:
                ax4c.text(0.5, 0.5, str(AUC_per_del_tv)[:5], transform=ax4c.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE, fontweight='bold') 
            else: 
                ax4c.text(0.5, 0.5, str(AUC_per_del_tv)[:5], transform=ax4c.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE)
            ax4c.spines['bottom'].set_linewidth(SPINE_LW)
            ax4c.spines['left'].set_linewidth(SPINE_LW)
            ax4c.set_xticks([])
            ax4c.set_yticks([])
            
            ax4d = fig.add_subplot(grid[4*i+1, 4])
            #ax4d.patch.set_facecolor(c_del_perfect[1])
            if AUC_per_ben_tv > AUC_per_ben_const and AUC_per_del_tv > AUC_per_del_const:
                ax4d.text(0.5, 0.5, str(AUC_per_del_const)[:5], transform=ax4d.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE) 
            else:
                ax4d.text(0.5, 0.5, str(AUC_per_del_const)[:5], transform=ax4d.transAxes, ha='center', va='center', fontsize=AXES_FONTSIZE, fontweight='bold') 
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
    plt.savefig(os.path.join(fig_path, 'finite-sampling.pdf'), dpi=1200)


def migration_plot(inf_dir, link_file, days_migrating=100, out_file='migration-plot'):
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
        linked_inf = calculate_linked_coefficients(os.path.join(inf_dir, file), link_file)[1][site_idx]
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
    fig = plt.figure(figsize=[SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH/1.618])
    grid = matplotlib.gridspec.GridSpec(1, 2, wspace=0.2, left=0.05, right=0.95, bottom=0.2, top=0.95)
    
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
    
    df = pd.DataFrame.from_dict({'importations' : num_inflow, 'importations per day' : np.array(num_inflow)/days_migrating, 'inferred coefficient' : inferred})
    
    fig.savefig(os.path.join(fig_path, out_file + '.pdf'), dpi=1200)

        
def plot_sequence_number(file, ax, fixed_axis=False):
    """ Given the output file of count_sequences, plot the number of sequences as a function of time. """
    
    data  = np.load(file, allow_pickle=True)
    title = file.split('/')[-1]
    title = title[:title.find('.')]
    if title.split('-')[-2]=='california' and title.split('-')[-1]=='south':
        title = 'southern california'
    if title.split('-')[-1]=='england_wales_scotland':
        title = 'Great Britain'
    title = title.split('-')[-1]
    title = ' '.join([i.capitalize() for i in title.split(' ')])
    nVec  = data['nVec']
    times = data['times']
    counts = [np.sum(nVec[t]) for t in range(len(nVec))]
    new_times = []
    for i in range(len(times)):
        time = dt.date(2020, 1, 1) + dt.timedelta(int(times[i]))
        new_time = str(time.year) + '/' + str(time.month) + "/" + str(time.day)
        new_times.append(new_time)

    ax.bar(times, counts, width=1, lw=0)
    ax.set_title(title, fontsize=AXES_FONTSIZE - 1, y=0.8)
    for line in ['right', 'top']:
        ax.spines[line].set_visible(False)
    for line in ['left', 'bottom']:
        ax.spines[line].set_linewidth(SPINE_LW)
        
        
def sampling_plots(folder, out_file=None, log=False):
    """ Given the folder containing the data files, plots the number of sampled genomes for each region."""
    
    N = len(os.listdir(folder))
    gen_dir = os.path.join(os.path.split(folder)[0], 'sampling-data-temp')
    if not os.path.exists(gen_dir):
        os.mkdir(gen_dir)
    else:
        for file in os.listdir(gen_dir):
            path = os.path.join(gen_dir, file)
            os.remove(path)
        
    locs_all = []
    for i in range(N):
        filename = os.listdir(folder)[i]
        file_loc = filename.split('-')
        if file_loc[1]=='united kingdom' and file_loc[2]=='england_wales_scotland':
            filename = '-'.join(file_loc[:3]) + '---'
        locs_all.append(filename[:filename.find('---')])
    locs_unique = np.unique(locs_all)
    
    new_files = []
    num_seqs  = []
    all_times = []
    max_seqs  = []    # the maximum number of sequences that appear on any given day
    for k in range(len(locs_unique)):
        nVec, sVec, times = [], [], []
        for j in range(N):
            filename   = os.listdir(folder)[j]
            filepath   = os.path.join(folder, filename)
            loc_temp   = filename[:filename.find('---')]
            if filename.split('-')[1]=='united kingdom' and filename.split('-')[2]=='england_wales_scotland':
                loc_temp = '-'.join(filename.split('-')[:3]) 
            if loc_temp==locs_unique[k]:
                data = np.load(filepath, allow_pickle=True)
                nVec_temp = data['nVec']
                time_temp = data['times']
                if len(times)==0:
                    times = list(time_temp)
                    nVec  = list(nVec_temp)
                    continue
                if times[-1]==time_temp[0]:
                    time_temp = time_temp[1:]
                    nVec_temp = nVec_temp[1:]
                if times[0]==time_temp[-1]:
                    time_temp = time_temp[:-1]
                    nVec_temp = nVec_temp[:-1]
                order = np.argmax([times[0], time_temp[0]])
                if times[0] < time_temp[0] < times[-1]:
                    for i in range(len(time_temp)):
                        nVec[list(times).index(time_temp[i])]  = nVec_temp[i]
                        times[list(times).index(time_temp[i])] = time_temp[i]
                elif order!=0:
                    nVec_new = list(nVec)
                    time_new = list(times)
                    for i in range(times[-1]+1, time_temp[0]):
                        nVec_new.append([])
                        time_new.append(i)
                    for i in range(len(time_temp)):
                        nVec_new.append(nVec_temp[i])
                        time_new.append(time_temp[i])
                    nVec  = nVec_new
                    times = time_new
                else:
                    nVec_temp = list(nVec_temp)
                    time_new  = list(time_temp)
                    for i in range(time_temp[-1]+1, times[0]):
                        nVec_temp.append([])
                        time_new.append(i)
                    for i in range(len(times)):
                        nVec_temp.append(nVec[i])
                        time_new.append(times[i])
                    nVec  = list(nVec_temp)
                    times = list(time_new)
        new_file = os.path.join(gen_dir, locs_unique[k]+'.npz')
        f        = open(new_file, mode='wb')
        np.savez_compressed(f, times=times, nVec=nVec)
        f.close()
        new_files.append(new_file)
        num_seqs.append(np.sum([np.sum(i) for i in nVec]))
        all_times.append(times)
        max_seqs.append(np.amax([np.sum(i) for i in nVec]))
    
    # order regions by total number of samples
    sorter    = np.argsort(num_seqs)[::-1]
    new_files = np.array(new_files)[sorter]
    all_times = np.array(all_times)[sorter]
    max_seqs  = np.array(max_seqs)[sorter]
    
    # Make discrete possibilities for the y-axis scale
    ylim_intervals = [100, 500, 1000, 2500, 8000]
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
    yticks = [np.arange(0, ylims[i] + 1, int(ylims[i] / 5)) for i in range(len(ylims))]
    
    # order regions by their ylimits
    if not log:
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
    fig    = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 1.2])
    #grid   = matplotlib.gridspec.GridSpec(int(len(locs_unique)/plots_per_row)+1, plots_per_row, hspace=0.65, wspace=0.15, bottom=0.05, top=0.98, left=0.05, right=0.99)
    if log:
        grid   = matplotlib.gridspec.GridSpec(int(len(locs_unique)/plots_per_row)+1, plots_per_row, hspace=0.3, wspace=0.05, bottom=0.025, top=0.98, left=0.05, right=0.99)
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
            row_idx  += int(group_lengths[i] / plots_per_row) + 1          
    else:
        for k in range(len(new_files)):
            ax = fig.add_subplot(grid[int(k/plots_per_row), k%plots_per_row])
            ax.set_xlim(times_full[0], times_full[-1])
            ax.set_xticks(xticks)
            if k > len(new_files) - plots_per_row - 1:
                ax.set_xticklabels(xlabels, rotation = 45)
            else:
                ax.set_xticklabels([])
            ax.set_yscale('log')
            ax.tick_params(labelsize=4, length=2, pad=1, width=SPINE_LW)
            plot_sequence_number(new_files[k], ax)
            ax.set_ylim(1, 10000)
            if k % plots_per_row != 0:
                ax.set_yticklabels([])
            else:
                ax.set_yticks([1, 10, 100, 1000, 10000])
        
    if out_file:
        plt.savefig(os.path.join(fig_path, out_file+'.pdf'), dpi=1200)
    else:
        plt.savefig(os.path.join(fig_path, 'sampling-dists.pdf'), dpi=1200)
        
        
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
    
    
def get_label2(i):
    return get_label(i[:-2]) + '-' + i[-1]
    
    
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


def hist_plot_inflow(simulation1, simulation2, num_sims, title=None, x_lims=None):
    inf1  = np.load(simulation1, allow_pickle=True)['inferred']
    inf2 = np.load(simulation2, allow_pickle=True)['inferred']
    inflow1 = [inf1[i][6] for i in range(len(inf1)) if 6 in range(len(inf1[i]))]
    inflow2 = [inf2[i][6] for i in range(len(inf2)) if 6 in range(len(inf2[i]))]
    inflow_weights1 = np.ones(len(inflow1)) / len(inflow1)
    inflow_weights2 = np.ones(len(inflow2)) / len(inflow2)
    hist_dict1 = {"weights" : inflow_weights1, "range" : [-0.01, 0.03]}
    hist_dict2 = {"weights" : inflow_weights2, "range" : [-0.02, 0.02 ]}
    fig = plt.figure(figsize=[SINGLE_COLUMN_WIDTH * 1.618, 2 * 1.618])
    ax  = fig.add_subplot(1,1,1)
    ax.set_ylim(0, 0.1)
    plt.tick_params(axis='y', labelsize=TICK_FONTSIZE, width=SPINE_LW)
    sns.distplot(inflow1, kde=False, bins=50, color="lightcoral", hist_kws=hist_dict1, label='naive')
    sns.distplot(inflow2, kde=False, bins=50, color="cornflowerblue", hist_kws=hist_dict2, label='corrected')
    ax.set_xlabel('Inferred selection\ncoefficients (%)', fontsize=AXES_FONTSIZE)
    ax.set_ylabel('Frequency', fontsize=AXES_FONTSIZE)
    ax.text(0.35, 0.8, 'corrected', fontsize=AXES_FONTSIZE, transform=ax.transAxes, ha='center', va='center', color='cornflowerblue')
    ax.text(0.75, 0.9, 'naive', fontsize=AXES_FONTSIZE, transform=ax.transAxes, ha='center', va='center', color='lightcoral')
    s_max = 0.03
    ax.set_xlim(-s_max, s_max)
    ax.set_xticks(np.linspace(-s_max, s_max, 5))
    ax.set_xticklabels([f'{i}'.replace('-', '\N{MINUS SIGN}') for i in np.linspace(-int(s_max*100), int(s_max*100), 5, dtype=int)], fontsize=TICK_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE, width=SPINE_LW)
    ax.set_position([0.1, 0.15, 0.8, 0.8])
    
    for line in ['bottom', 'left']:
        ax.spines[line].set_linewidth(SPINE_LW)
    for line in ['top', 'right']:
        ax.spines[line].set_visible(False)
    #plt.gcf().subplots_adjust(bottom=0.3)
    #plt.gcf().subplots_adjust(left=0.3)
    plt.savefig(os.path.join(fig_path, 'travel_correction.pdf'), dpi=1200)
    
    
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
        sns.lineplot(param_vals, beni, color=MAIN_COLOR, lw=1, ax=axis, label='Beneficial')
        sns.lineplot(param_vals, deli, color=COMP_COLOR, lw=1, ax=axis, label='Deleterious')
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
        axis.legend(loc='lower right', fontsize=AXES_FONTSIZE)
    
    sim,  ben_sim,  del_sim  = find_auroc(sim_dir,  'sim')
    t,    ben_t,    del_t    = find_auroc(t_dir,    'generations')
    samp, ben_samp, del_samp = find_auroc(samp_dir, 'sample')
    n,    ben_n,    del_n    = find_auroc(n_dir,    'N')
    
    fig  = plt.figure(figsize=[DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH/(1.618)])
    grid = matplotlib.gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.4)
    ax0  = fig.add_subplot(grid[1, 1])
    make_subplot(samp, ben_samp, del_samp, parameter='samples per generation',          axis=ax0)
    ax1  = fig.add_subplot(grid[0, 1])
    make_subplot(t,    ben_t,    del_t,    parameter='generations',                     axis=ax1)
    ax2  = fig.add_subplot(grid[1, 0])
    sim_ticks = [1, 5, 10, 15, 20]
    make_subplot(sim,  ben_sim,  del_sim,  parameter='number of independent outbreaks', axis=ax2, xticks=sim_ticks)
    ax3  = fig.add_subplot(grid[0, 0])
    pop_ticks = [100, 400, 700, 1000]
    make_subplot(n,    ben_n,    del_n,    parameter='population size',                 axis=ax3, xticks=pop_ticks)
    
    fig.text(0.1,   0.95, 'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.525, 0.95, 'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.1,   0.5,  'c', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.525, 0.5,  'd', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    
    plt.savefig(os.path.join(fig_path, 'auroc-comparison.pdf'), dpi=1200)


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
        

def s_compare_regs_cutoff_window(files_reg=[],    labels_reg=[], 
                                 files_cutoff=[], labels_cutoff=[], 
                                 files_wind=[],   labels_wind=[],
                                 out='regularization-comparison'):
    """ Given selection coefficients with different regularizations, make a scatter plot of the first coefficients with each of the others."""

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
    for i in range(len(files_reg) - 1):
        ax = fig.add_subplot(low_grid[0, i])
        if i==0:
            s_compare_scatter(files_reg[i+1], files_reg[0], ax=ax, label1=labels_reg[i+1], label2=labels_reg[0])
        else:
            s_compare_scatter(files_reg[i+1], files_reg[0], ax=ax, label1=labels_reg[i+1])
            ax.set_yticklabels([])
            
    for i in range(len(files_cutoff) - 1):
        ax = fig.add_subplot(mid_grid[0, i])
        if i==0:
            s_compare_scatter(files_cutoff[i+1], files_cutoff[0], ax=ax, label1=labels_cutoff[i+1], label2=labels_cutoff[0])
        else:
            s_compare_scatter(files_cutoff[i+1], files_cutoff[0], ax=ax, label1=labels_cutoff[i+1])
            ax.set_yticklabels([])
            
    for i in range(len(files_wind) - 1):
        ax = fig.add_subplot(top_grid[0, i])
        if i==0:
            s_compare_scatter(files_wind[i+1], files_wind[0], ax=ax, label1=labels_wind[i+1], label2=labels_wind[0])
        else:
            s_compare_scatter(files_wind[i+1], files_wind[0], ax=ax, label1=labels_wind[i+1])
            
    fig.text(0.05,  0.98,  'a', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.285, 0.98,  'b', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.05,  0.68,  'c', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.285, 0.68,  'd', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.05,  0.35,  'e', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.285, 0.35,  'f', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.525, 0.35,  'g', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
    fig.text(0.765, 0.35,  'h', transform=fig.transFigure, ha='center', va='center', fontweight='bold')
            
    plt.savefig(os.path.join(fig_path, out+'.pdf'), dpi=2500)
    
    
    
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
    ax.scatter(s1 * 100, s2 * 100, marker='.', s=4)
    Rsquared = spstats.pearsonr(s1, s2)[0] ** 2
    ax.text(0.1, 0.75, '$R^2$ = %.3f' % Rsquared, fontsize=AXES_FONTSIZE, transform=ax.transAxes)
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(2))

        
def selection_table(file, out=None):
    """ Makes a table of the largest selection coefficients"""
    df_sel = pd.read_csv(file, memory_map=True)

    s_sort       = np.argsort(df_sel['selection coefficient'])[::-1]
    table_data   = []
    table_labels = ['Rank', 'Protein', 'Mutation (nt)', 'Mutation (aa)', 'Selection (%)', 'Location', 'Phenotypic effect']
    for i in range(50):
        row   = []
        entry = df_sel.iloc[list(s_sort)[i]]
        loc   = str(entry['amino acid number in protein'])
        anc   = entry['amino acid mutation'][0]
        mut   = entry['amino acid mutation'][-1]
        nuc_n = entry['nucleotide number'] + 1
        nuc_m = entry['nucleotide']
        nuc_r = entry['reference nucleotide']
        name  = anc + loc + mut
        nuc_name = nuc_r + str(nuc_n) + nuc_m
        row.append('%s' % (i + 1))
        row.append('%s' % entry['protein'])
        row.append('%s' % nuc_name)
        row.append('%s' % name)
        row.append('%.1f' % (100 *entry['selection coefficient']))
        row.append('')
        row.append('')
        table_data.append(row)
    
    print('\\setlength{\\tabcolsep}{12pt}')
    print('\\begin{table}')
    print('\\centering')
    print(tabulate(table_data, headers=table_labels, tablefmt='latex_booktabs', numalign='left'))
    print('\\caption{Table of most highly selected mutations across the SARS-CoV-2 genome.}')
    print('\\label{table:selection}')
    print('\\end{table}')
    
    
def variant_selection_table(file, ind_file):
    """ Makes a table of the selection coefficients for the major variants. 
    file is a .npz file containing information for the variants.
    ind_file is a .csv file containing information for individual mutations."""
    
    def sort_by_contribution(sites, array, ind_file):
        """ Sorts the sites and array in a group in decending order of their selection coefficient."""
        data    = np.load(ind_file, allow_pickle=True)
        s_ind   = data['selection']
        alleles = data['allele_number']
        labels  = [get_label2(i) for i in alleles]
        s_sites = [s_ind[labels.index(i)] for i in sites]
        mask    = np.argsort(s_sites)[::-1]
        sites   = np.array(sites)[mask]
        array   = np.array(array)[mask]
        return sites, array
        
    
    df = pd.read_csv(file)
    
    name_dict = {'Alpha' : 'B.1.1.7', 'Beta' : 'B.1.351', 'Gamma' : 'P.1', 'Delta' : 'B.1.617.2', 
                 'Lambda' : 'C.37', 'Epsilon' : 'B.1.427//B.1.429', '20e_eu1' : 'B.1.177', 'B.1' : 'B.1'}
    
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
    
    table_labels = ['Variant', 'Pango Lineage', 'Selection Coefficient (%)',  'Mutations']
    table_data   = []
    for i in range(len(df_var)):
        row = []
        if var_names[i]=='20e_eu1':
            var_names[i]='20E-EU1'
        row.append(var_names[i])
        row.append(pango_name[i])
        row.append('%.1f' % (100 * selection[i]))
        row.append(muts_full[i])
        table_data.append(row)
    
    print('\\setlength{\\tabcolsep}{12pt}')
    print('\\begin{table}')
    print('\\centering') 
    print(tabulate(table_data, headers=table_labels, tablefmt='latex_booktabs', numalign='left'))
    print('\\caption{Table of selection coefficients for groups of mutations. Mutations that contribute most strongly to selection are listed first.}')
    print('\\label{table:variant_selection}')
    print('\\end{table}') 
