#!/usr/bin/env python
# coding: utf-8
# %%

# %%

# Check what the above means

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
import os
import subprocess
import data_processing as dp
import pandas as pd
import lmdb

REF_TAG = 'EPI_ISL_402125'
MAX_AMBIGUOUS = 298
ALPHABET    = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
alph_temp   = list(ALPHABET)
alph_new    = [i for i in alph_temp]
for i in range(len(alph_temp)):
    for j in range(len(alph_temp)):
        alph_new.append(alph_temp[i] + alph_temp[j])
ALPHABET    = alph_new
    
def find_ref_seq(msa_file, ref_tag=REF_TAG):
    """ Get the reference sequence from the MSA"""
    found   = False
    ref_seq = ''
    for line in open(msa_file):
        data = line.split('\n') 
        if len(data)==0 or data[0][0]=='>':
            if found:
                break
            #else:
            #    continue
        if data[0][0]=='>':
            tag = data[0][1:] 
            tag = tag.split('|')[1]
            #print(f'tag {tag}')
            if tag==REF_TAG:
                found=True
        elif found:
            ref_seq += data[0]
    print(f'refernece sequence length {len(ref_seq)}')
    print(f'found {found}')
    print(ref_seq)
    #
    idxs_keep = []
    ref_index = 0
    ref_alpha = 0
    ref_seq   = list(ref_seq)
    for i in range(len(ref_seq)):
        ref_str = 'NA'
        if ref_seq[i]!='-':
            idxs_keep.append(i)
            ref_index += 1
        elif ref_index==22204:
            idxs_keep.append(i)
        else:
            continue
    idxs_keep = np.array(idxs_keep)
    ref_poly  = np.array(ref_seq)[idxs_keep]
    print(idxs_keep)
    idxs_keep = idxs_keep.astype('int32')
    
    return idxs_keep, ref_poly

def index_ref_seq(ref_seq, out_file):
    """Finds an index for the sites on the unfiltered reference sequence so that they can be found after regions are combined"""
    ref_seq   = list(ref_seq)
    #print(ref_seq)
    ref_index = 0
    ref_alpha = 0
    f = open(out_file + '.csv', mode='w')
    f.write('index,ref_index,nucleotide\n')
    for i in range(len(ref_seq)):
        ref_str = 'NA'
        if ref_seq[i]!='-':
            ref_str = '%d' % ref_index
            ref_index += 1
            ref_alpha  = 0
        else:
            ref_str = '%d%s' % (ref_index-1, ALPHABET[ref_alpha])
            ref_alpha += 1
        f.write(f'{i},{ref_str},{ref_seq[i]}\n')
    f.close()
    

def main(args):
    """Merge the metadata with the aligned sequences.
    FUTURE: Check if taking the two lists of accessions, making them sets, 
    finding their intersection, and then looping through the remaining accessions
    and finding their locations in the msa and the metadata (via dictionaries) 
    is faster"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('--meta_file',type=str, default=None,                        help='the file containing the metadata')
    parser.add_argument('--msa_file', type=str, default=None,                        help='the file containing the sequences')
    parser.add_argument('--lmdbDir',  type=str, default=None,                        help='the directory to write the database to')
    parser.add_argument('--metaNew',  type=str, default=None,                        help='the path at which to write the new metadata file')
    parser.add_argument('--dbNumber', type=int, default=None,                        help='the number of the database for this job to write, 1 for each parallel job')
    parser.add_argument('--refFile',  type=str, default='ref-index',                 help='the file to output the reference index to')
    parser.add_argument('--newMSA',       action='store_true',   default=False,      help='if true, the MSA is the MSAunmasked file from GISAID')
    parser.add_argument('--metadataOnly', action='store_true',   default=False,      help='if true, only make the new metadata csv file and do not write sequences to database')
    parser.add_argument('--refOnly',      action='store_true',   default=False,      help='if true, only make the reference index file')
    
    arg_list = parser.parse_args(args)
    
    msa_file  = arg_list.msa_file
    meta_file = arg_list.meta_file
    lmdb_dir  = arg_list.lmdbDir
    meta_out  = arg_list.metaNew
    print2(os.getcwd())
    
    status_name = f'merge-status-lmdb-{arg_list.dbNumber}.csv'
    status_file = open(status_name, 'w')
    status_file.close()
    
    def print2(*args):
        """ Print the status of the processing and save it to a file."""
        stat_file = open(status_name, 'a+')
        line      = [str(i) for i in args]
        string    = '\t'.join(line)
        stat_file.write(string+'\n')
        stat_file.close()
        print(string)
    
    if lmdb_dir is None:
        print('no path for a memory mapped database was passed. Pass the "lmdbDir" flag.')
        sys.exit()
        
    if meta_out is None:
        print('no path for an output metadata file was passed. Pass the "metaNew" flag to save the metadata.')
        save_metadata = False
    else:
        save_metadata = True
    
    df = pd.read_csv('%s' % meta_file, 
                     usecols=['accession', 'virus_name', 'date', 'location', 'submission_date'])
    print2('metadata file read')
    accessions = list(df.accession)
    acc_set    = set(accessions)
    acc_dict   = {v : i for i, v in enumerate(accessions)}
    tag        = []
    print2('length of metadata file:', len(df))
    
    
     # eliminate the "location" column from the metadata and add in columns with the continent, country, region, and subregion
    if save_metadata and arg_list.dbNumber==1:
        idxs_drop = []
        locs      = list(df['location'])
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
            for i in range(len(locs)):
                l = locs[i].split(' / ')
                m = locs2[i].split('-')[:-1]
                m = [j.lower() for j in m]
                if len(l) == 1:
                    l.append('NA')
                    l.append('NA')
                    #l.append('NA')
                    l.append('/'.join(m))
                    if accessions[i] != REF_TAG:
                        idxs_drop.append(i)    # drop sequence if there is no country level location information
                if len(l) == 2:
                    #l.append('NA')
                    l.append('NA')
                    l.append('/'.join(m))
                elif len(l) == 3:
                    #l.append('NA')    # Comment this out to make london sequences location appear in the correct slot. 
                    l.append('/'.join(m))
                else:
                    #l[4] += '/' + '/'.join(m)
                    l.append('/'.join(m))
                #elif len(l) == 4:
                #    l.append('/'.join(m))
                #elif len(l) == 5:
                #    l[4] += '/' + '/'.join(m)
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
    
        df  = df.drop('location', axis=1)
        df2 = pd.DataFrame.from_dict({'continent' : continent, 'country' : country, 
                                      'region' : region, 'subregion' : subregion,
                                      'subsubregion' : subsubregion})
        df = pd.concat([df, df2], axis=1)
        df = df.drop(idxs_drop, axis=0)
        df.to_csv(meta_out)
        
    if arg_list.metadataOnly:
        sys.exit()
    
    # make database directory
    if not os.path.exists(lmdb_dir):
        os.mkdir(lmdb_dir)
    max_seqs = 5E5    # the maximum sequences allowed in each database    
        

    # Add sequences to the MSA if corresponding tags exist in the metadata
    accessions_used    = set()   # accession numbers of sequences that have already been added to the msa
    accession_present  = False
    counter            = 0
    accessions_list    = []
    sequences          = []
    
    if arg_list.dbNumber==1:
        idxs_keep, ref_seq = np.array(find_ref_seq(msa_file))
        ref_file = arg_list.refFile
        index_ref_seq(ref_seq, ref_file)
        idxs_keep = np.array(idxs_keep)
        print(idxs_keep)
        idxs_keep = idxs_keep.astype('int32')
    if arg_list.refOnly:
        sys.exit()
    
    db_counter = 0
    for line in open(msa_file):
        if counter == 0:
            print2('msa file opened')
        data = line.split('\n') 
        if len(data)==0:
            continue
        if data[0][0]=='>':
            counter += 1 
            if accession_present:
                if temp_acc not in accessions_used:
                    #new_seq   = np.array(list(seq))[idxs_keep]
                    #num_ambig = np.count_nonzero(new_seq=='N')
                    #if num_ambig > MAX_AMBIGUOUS:
                    #    continue
                    #new_seq = ''.join(list(new_seq))
                    accessions_used.add(temp_acc)
                    if db_counter==arg_list.dbNumber:
                        accessions_list.append(temp_acc.encode())
                        sequences.append(seq.encode())
            if counter % max_seqs == 0:
                # save partial alignment of max_seqs number of sequences to a database
                if db_counter!=arg_list.dbNumber:
                    db_counter += 1
                    continue
                print2(f'{counter} sequences in msa analyzed')
                db_start_time = timer()
                data   = [(accessions_list[i], sequences[i]) for i in range(len(sequences))]
                db_num = int(counter / max_seqs)
                db_dir = os.path.join(lmdb_dir, f'db-{db_num}')
                env    = lmdb.open(db_dir, map_size=int(1E12), max_dbs=100, max_readers=2000, writemap=True)    # make enviornment
                
                with env.begin(write=True) as txn:    # transaction to clear any existing data stored in the database
                    with txn.cursor() as crs:
                        for key, value in crs.iternext(keys=True):
                            crs.delete()
    
                with env.begin(write=True) as txn:     # transaction to write sequences to database
                    with txn.cursor() as crs:
                        result = crs.putmulti(data)
                        #print2(result)
                        
                        #try:
                        #    result = crs.putmulti(data)    # write data
                        #except TypeError:
                        #    print([type(i) for i in data])
                        #    print(data)
                        #print2(result)   
                db_end_time = timer()
                print2(f'took {(db_end_time - db_start_time) / 60} minutes to write the sequences to the database number {db_num}')
                accessions_list = []    # reset accession list
                sequences       = []    # reset sequence list
                db_counter += 1
            tag = data[0][1:]
            seq = ''
            if isinstance(tag, str):
                temp_acc = tag.split('|')
            else:
                print2('tag', tag)
            if len(temp_acc)<2:
                print2('accession', temp_acc)
            else:
                temp_acc = temp_acc[1]
            if temp_acc in acc_set and temp_acc not in accessions_used:
                accession_present = True
            else:
                accession_present = False
        elif accession_present:
            if len(data)!=0 and db_counter==arg_list.dbNumber:
                seq += data[0]     
                
    # write the final sequences to a final database
    db_start_time = timer()
    data   = [(accessions_list[i], sequences[i]) for i in range(len(sequences))]
    db_num = int(counter / max_seqs) + 1
    db_dir = os.path.join(lmdb_dir, f'db-{db_num}')
    env    = lmdb.open(db_dir, map_size=int(1E12), max_dbs=100, max_readers=2000, writemap=True)    # make enviornment
                
    with env.begin(write=True) as txn:    # transaction to clear any existing data stored in the database
        with txn.cursor() as crs:
            for key, value in crs.iternext(keys=True):
                crs.delete()
    
    with env.begin(write=True) as txn:     # transaction to write sequences to database
        with txn.cursor() as crs:
            result = crs.putmulti(data)
            print2(result)
    db_end_time = timer()
    print2(f'took {(db_end_time - db_start_time) / 60} minutes to write the sequences to the database number {db_num}')
    print2(f'total number of sequences in the database is {counter}')

        
        
        
if __name__ == '__main__': 
    main(sys.argv[1:])
