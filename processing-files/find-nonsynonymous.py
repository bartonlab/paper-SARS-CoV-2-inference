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
import datetime as dt
import colorsys
import pandas as pd
from colorsys import hls_to_rgb, hsv_to_rgb

def complementary(color_hex):
    """ Returns a color complementary to the given r,g,b. """
    #hsv = colorsys.rgb_to_hsv(r, g, b)
    rgb = mcolors.to_rgb(color_hex)
    hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    return hsv_to_rgb((hsv[0] + 0.5) % 1, hsv[1], hsv[2])

PROTEINS = ['NSP1', 'NSP2', 'NSP3', 'NSP4', 'NSP5', 'NSP6', 'NSP7', 'NSP8','NSP9', 'NSP10', 'NSP12', 'NSP13', 'NSP14', 
            'NSP15', 'NSP16', 'S', 'ORF3a', 'E', 'M', 'ORF6', 'ORF7a', 'ORF7b', 'ORF8', 'N', 'ORF10']
PROTEIN_LENGTHS = {'ORF3a' : 828, 'E' : 228, 'ORF6' : 186, 'ORF7a' : 365, 'ORF7b' : 132, 'S' : 3822,
                   'N' : 1260, 'M' : 669, 'ORF8' : 336, 'ORF10' : 117, 'NSP1' : 539, 'NSP2' : 1914,
                   'NSP3' : 5834, 'NSP4' : 1500, 'NSP5' : 917, 'NSP6' : 870, 'NSP7' : 249, 'NSP8' : 594,
                   'NSP9' : 339, 'NSP10' : 417, 'NSP12' :2795, 'NSP13' : 1803, 'NSP14' : 1582, 
                   'NSP15' : 1038, 'NSP16' : 894}

cm_to_inch = lambda x: x/2.54   

# load data_processing module
import data_processing as dp


def main(args):
    """Infer time-varying selection coefficients from the results of a Wright-Fisher simulation"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('--data',    type=str,    default=None,                    help='folder containing the data files')    
    parser.add_argument('--multisite', action='store_true', default=False,         help='whether or not the ')
    
    arg_list  = parser.parse_args(args)
    multisite = arg_list.multisite
    data      = arg_list.data
    dp.find_nonsynonymous_all('EPI_ISL_402125', data,
                          os.path.join(data, 'synonymous'),
                          os.path.join(data, 'protein-changes'), 
                          multisite=multisite)
                    
                    
        
if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    

