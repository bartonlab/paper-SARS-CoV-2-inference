#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Check what the above means

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
import os
import pandas as pd


REF_TAG = 'EPI_ISL_402125'


def main(args):
    """Check the number of genomes in each country, and the number in each state in the United States"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',            type=str,    default='./bigdata/SARS-CoV-2-Data/regional', help='output dir')
    parser.add_argument('--alignment',   type=str,    default=None,                                 help='the alignment (and metadata) file')
    parser.add_argument('--laboratory',  default=False,  action='store_true',                       help='whether or not to use the originating_lab information or the virus_name information')
    
    arg_list = parser.parse_args(args)
    
    out_file = arg_list.o
    msa_file = arg_list.alignment
    
    if arg_list.laboratory:
        df = pd.read_csv(msa_file, usecols=['accession', 'location', 'originating_lab'])
    else:
        df = pd.read_csv(msa_file, usecols=['accession', 'location', 'virus_name'])
    locs   = list(df.location)  
    
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
        
    data = dict(continent=continent, country=country, region=region, subregion=subregion)
    df   = pd.DataFrame(data=data)
    df.to_csv(out_file + '.csv', index=False)
    
    

    
if __name__ == '__main__': 
    main(sys.argv[1:])