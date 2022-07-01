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
    
    """
    if not arg_list.laboratory:
        virus_name = list(df.virus_name)
        virus_name = [i.split('/') for i in virus_name]
        locs2      = []
        for i in range(len(virus_name)):
            if len(virus_name[i])>2:
                locs2.append(virus_name[i][2])
            else:
                #print(len(virus_name[i]))
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
    else:
        lab_data   = [str(i).lower() for i in list(df.originating_lab)]
        locs_new   = []
        for i in range(len(locs)):
            l = locs[i].split(' / ')
            m = lab_data[i]
            if len(l) == 2:
                l.append('NA')
                l.append(m)
            elif len(l) == 3:
                l.append(m)
            else:
                print('there is too much location information', len(l), len(m))
            locs_new.append(l)

    df = 0
    continents = []
    countries  = []
    regions    = []
    subregions = []

    for l in locs_new:
        #l = l.split(' / ')
        if len(l)>0:
            continents.append(l[0])
        else:
            continents.append('NA')
        
        if len(l)>1:
            if l[1] == 'houston':
                countries.append('usa')
                regions.append('texas')
                subregions.append('houston')
                continue
            else:
                countries.append(l[1])
        else:
            countries.append('NA')
        
        if len(l)>2:
            regions.append(l[2])
        else:
            regions.append('NA')
        
        if len(l)>3:
            subregions.append('/'.join(l[3:]))
        else:
            subregions.append('NA')
    """
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
    
    """
    countries      = df['country'].value_counts().axes[0]
    country_counts = list(df['country'].value_counts())
    #print(df.country)
    #print(df.head())
    
    df_usa         = df[df.country=='usa']
    states         = df_usa['region'].value_counts().axes[0]
    states_counts  = list(df_usa['region'].value_counts())
    
    usa_sub        = df_usa['subregion'].value_counts().axes[0]
    usa_sub_counts = list(df_usa['subregion'].value_counts())
    
    df_cali        = df_usa[df_usa.region=='california']
    cali_regions   = df_cali['subregion'].value_counts().axes[0]
    cali_counts    = list(df_cali['subregion'].value_counts())
    
    df_india       = df[df.country=='india']
    india_regions  = df_india['region'].value_counts().axes[0]
    india_counts   = list(df_india['region'].value_counts())
    
    df_uk          = df[df.country=='united kingdom']
    uk_subregions  = df_uk['subregion'].value_counts().axes[0]
    uk_counts      = list(df_uk['subregion'].value_counts())
    uk_regions     = df_uk['region'].value_counts().axes[0]
    uk_reg_counts  = list(df_uk['region'].value_counts())
    
    df_england     = df[df.region=='england']
    eng_subregions = df_england['subregion'].value_counts().axes[0]
    eng_counts     = list(df_england['subregion'].value_counts())
    
    df_scot        = df[df.region=='scotland']
    scot_sub       = df_scot['subregion'].value_counts().axes[0]
    scot_counts    = list(df_scot['subregion'].value_counts())
    
    df_wales       = df[df.region=='wales']
    wales_sub      = df_wales['subregion'].value_counts().axes[0]
    wales_counts   = list(df_wales['subregion'].value_counts())
    
    df_brazil      = df[df.country=='brazil']
    brazil_regions = df_brazil['region'].value_counts().axes[0]
    brazil_counts  = list(df_brazil['region'].value_counts())
    
    df_canada      = df[df.country=='canada']
    canada_regions = df_canada['region'].value_counts().axes[0]
    canada_counts  = list(df_canada['region'].value_counts())
    
    df_canada_sub  = df[df.country=='canada']
    canada_sub     = df_canada_sub['subregion'].value_counts().axes[0]
    canada_sub_n   = list(df_canada['subregion'].value_counts())
    
    df_brazil_sub  = df[df.country=='brazil']
    brazil_sub     = df_brazil_sub['subregion'].value_counts().axes[0]
    brazil_sub_n   = list(df_brazil_sub['subregion'].value_counts())
    
    df_turkey      = df[df.country=='turkey']
    turkey_regions = df_turkey['region'].value_counts().axes[0]
    turkey_counts  = list(df_turkey['region'].value_counts())
    
    df_russia      = df[df.country=='russia']
    russia_regions = df_russia['region'].value_counts().axes[0]
    russia_counts  = list(df_russia['region'].value_counts())
    
    df_russia_sub  = df[df.country=='russia']
    russia_sub     = df_russia_sub['subregion'].value_counts().axes[0]
    russia_sub_n   = list(df_russia_sub['subregion'].value_counts())
    """
    
    df.to_csv(out_file + '.csv', index=False)
    
    """
    df_countries   = pd.DataFrame.from_dict({'countries'  : countries,      'counts' : country_counts})
    df_states      = pd.DataFrame.from_dict({'states'     : states,         'counts' : states_counts})
    df_cali        = pd.DataFrame.from_dict({'counties'   : cali_regions,   'counts' : cali_counts})
    df_india       = pd.DataFrame.from_dict({'regions'    : india_regions,  'counts' : india_counts})
    df_uk          = pd.DataFrame.from_dict({'subregions' : uk_subregions,  'counts' : uk_counts})
    df_uk_regions  = pd.DataFrame.from_dict({'regions'    : uk_regions,     'counts' : uk_reg_counts})
    df_usa_sub     = pd.DataFrame.from_dict({'subregions' : usa_sub,        'counts' : usa_sub_counts})
    df_england     = pd.DataFrame.from_dict({'subregions' : eng_subregions, 'counts' : eng_counts})
    df_scotland    = pd.DataFrame.from_dict({'subregions' : scot_sub,       'counts' : scot_counts})
    df_wales       = pd.DataFrame.from_dict({'subregions' : wales_sub,      'counts' : wales_counts})
    df_brazil      = pd.DataFrame.from_dict({'regions'    : brazil_regions, 'counts' : brazil_counts})
    df_canada      = pd.DataFrame.from_dict({'regions'    : canada_regions, 'counts' : canada_counts})
    df_canada_sub  = pd.DataFrame.from_dict({'subregions' : canada_sub,     'counts' : canada_sub_n})
    df_brazil_sub  = pd.DataFrame.from_dict({'subregions' : brazil_sub,     'counts' : brazil_sub_n})
    df_turkey      = pd.DataFrame.from_dict({'regions'    : turkey_regions, 'counts' : turkey_counts})
    df_russia      = pd.DataFrame.from_dict({'regions'    : russia_regions, 'counts' : russia_counts})
    df_russia_sub  = pd.DataFrame.from_dict({'subregions' : russia_sub,     'counts' : russia_sub_n})
    
    
    df_countries.to_csv(out_file+'-countries.csv',   index=False)
    df_states.to_csv(out_file+'-usa.csv',            index=False)
    df_cali.to_csv(out_file+'-usa-california.csv',   index=False)
    df_india.to_csv(out_file+'-india.csv',           index=False)
    df_uk.to_csv(out_file+'-uk.csv',                 index=False)
    df_uk_regions.to_csv(out_file+'-uk-reg.csv',     index=False)
    df_usa_sub.to_csv(out_file+'-usa-sub.csv',       index=False)
    df_england.to_csv(out_file+'-england.csv',       index=False)
    df_scotland.to_csv(out_file+'-scotland.csv',     index=False)
    df_wales.to_csv(out_file+'-wales.csv',           index=False)
    df_brazil.to_csv(out_file+'-brazil.csv',         index=False)
    df_canada.to_csv(out_file+'-canada.csv',         index=False)
    df_canada_sub.to_csv(out_file+'-canada-sub.csv', index=False)
    df_brazil_sub.to_csv(out_file+'-brazil-sub.csv', index=False)
    df_turkey.to_csv(out_file+'-turkey.csv',         index=False)
    df_russia.to_csv(out_file+'-russia.csv',         index=False)
    df_russia_sub.to_csv(out_file+'-russia-sub.csv', index=False)
    """

    
if __name__ == '__main__': 
    main(sys.argv[1:])