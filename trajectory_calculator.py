# Save frequency trajectories from variant metadata

import sys
import argparse
import numpy as np                          # numerical tools
import pandas as pd
import datetime as dt
from timeit import default_timer as timer   # timer for performance


# Global variables

VARIANTS = {'alpha': 'Former VOC Alpha GRY (B.1.1.7+Q.*) first detected in the UK',
            'delta': 'Former VOC Delta GK (B.1.617.2+AY.*) first detected in India',
            'omicron': 'Former VOC Omicron GRA (B.1.1.529+BA.*) first detected in Botswana/Hong Kong/South Africa',
            'beta': 'Former VOC Beta GH/501Y.V2 (B.1.351+B.1.351.2+B.1.351.3) first detected in South Africa',
            'gamma': 'Former VOC Gamma GR/501Y.V3 (P.1+P.1.*) first detected in Brazil/Japan',
            'epsilon': 'Former VOI Epsilon GH/452R.V1 (B.1.429+B.1.427) first detected in USA/California',
            'eta': 'Former VOI Eta G/484K.V3 (B.1.525) first detected in UK/Nigeria',
            'iota': 'Former VOI Iota GH/253G.V1 (B.1.526) first detected in USA/New York',
            'kappa': 'Former VOI Kappa G/452R.V3 (B.1.617.1) first detected in India',
            'lambda': 'Former VOI Lambda GR/452Q.V1 (C.37+C.37.1) first detected in Peru',
            'mu': 'Former VOI Mu GH (B.1.621+B.1.621.1) first detected in Colombia',
            'BA.2.86': 'VOI GRA (BA.2.86+BA.2.86.* excluding JN.1, JN.1.*) first detected in Denmark/Israel/USA',
            'XBB': 'VUM GRA (XBB+XBB.* excluding XBB.1.5, XBB.1.16, XBB.1.9.1, XBB.1.9.2, XBB.2.3) first detected in India',
            'XBB.1.5': 'VOI GRA (XBB.1.5+XBB.1.5.*) first detected in Austria/India/Bangladesh',
            'JN.1': 'VOI GRA (JN.1+JN.1.*) first detected in Luxembourg/Iceland',
            'EG.5.1': 'VOI GRA (EG.5+EG.5.*) first detected in Indonesia/France'}
VAR_NAMES = ['alpha', 'delta', 'omicron', 'beta', 'gamma', 'epsilon', 'eta', 'iota', 'kappa', 'lambda', 'mu', 'BA.2.86', 'XBB', 'XBB.1.5', 'JN.1', 'EG.5.1']
PANGO_NAMES = ['BA.1', 'BA.2', 'BA.4', 'BA.5']
PANGO_INCLUDE = {'BA.1': ['B.1.1.529', 'BA.1'], 'BA.2': ['BA.2'], 'BA.4': ['BA.4'], 'BA.5': ['BA.5']}
PANGO_EXCLUDE = {'BA.1': [], 'BA.2': ['BA.2.86', 'BA.2.12.1'], 'BA.4': [], 'BA.5': []} 
T_FIRST = {'alpha': 275,
           'delta': 395,
           'XBB': 960,
           'XBB.1.5': 970 }
T_LAST = {'alpha': 670,
          'gamma': 730,
          'delta': 800 }
T_FIRST_PANGO = {'BA.1': 640,
                 'BA.2': 640,}

COL_LOC = 'Compressed location'
COL_VAR = 'Variant'
COL_PAN = 'Pango lineage'
COL_T   = 'Time'
T_MIN   = 0
T_MAX   = 1500
T_STEP  = 7


###### Main functions ######


def usage():
    print("")


def main(verbose=False):
    """ Simulate Wright-Fisher evolution of a population and save the results. """
    
    # Read in parameters from command line
    
    parser = argparse.ArgumentParser(description='Record variant frequency trajectories from SARS-CoV-2 GISAID metadata')
    parser.add_argument('-i',     type=str,   default='data/metadata.csv', help='input metadata file')
    parser.add_argument('-o',     type=str,   default='data/trajectories', help='output destination')
    parser.add_argument('--nmin', type=int,   default=20,                  help='minimum number of sequences per week')
    parser.add_argument('--tmin', type=int,   default=4,                   help='minimum number of consecutive time points to record')
    arg_list = parser.parse_args(sys.argv[1:])

    n_min = arg_list.nmin
    t_min = arg_list.tmin
    
    # Read in data and loop over locations

    f = open(arg_list.o + '.csv', 'w')
    f.write('location,times,frequencies,variant_names\n')

    df = pd.read_csv(arg_list.i)

    loc_list = np.unique(df[COL_LOC].astype(str))
    for loc in loc_list:
        if loc=='':
            continue

        loc_str = '-'.join(loc.replace('/', '').split())

        df_loc = df[df[COL_LOC]==loc]
    
        # Iterate through sequences and record frequencies if enough data is present

        n_good = 0
        times = []
        frequencies = [[] for i in range(len(VAR_NAMES)+len(PANGO_NAMES))]

        for t in range(T_MIN, T_MAX, T_STEP):
            df_t = df_loc[(t<=df_loc[COL_T]) & (df_loc[COL_T]<(t+T_STEP))]
            n_seqs = len(df_t)

            if n_seqs<n_min:
                if n_good>=t_min:
                    t_str = ' '.join([str(_t) for _t in times])
                    f_str = [' '.join(['%.3f' % _f for _f in freq]) for freq in frequencies]
                    for i_var in range(len(VAR_NAMES)):
                        f.write('%s,%s,%s,%s\n' % (loc, t_str, f_str[i_var], VAR_NAMES[i_var]))
                    for i_var in range(len(PANGO_NAMES)):
                        f.write('%s,%s,%s,%s\n' % (loc, t_str, f_str[i_var+len(VAR_NAMES)], PANGO_NAMES[i_var]))
                n_good = 0
                times = []
                frequencies = [[] for i in range(len(VAR_NAMES)+len(PANGO_NAMES))]

            else:
                n_good += 1
                times.append(t)
                is_nonzero = [True for i_var in range(len(VAR_NAMES))]
                for var_first in T_FIRST.keys():
                    if t<T_FIRST[var_first]:
                        idx = VAR_NAMES.index(var_first)
                        is_nonzero[idx]=0
                        n_seqs -= len(df_t[df_t[COL_VAR]==var_first])
                for var_last in T_LAST.keys():
                    if t>T_LAST[var_last]:
                        idx = VAR_NAMES.index(var_last)
                        is_nonzero[idx]=0
                        n_seqs -= len(df_t[df_t[COL_VAR]==var_last])
                for i_var in range(len(VAR_NAMES)):
                    if is_nonzero[i_var]:
                        frequencies[i_var].append(len(df_t[df_t[COL_VAR]==VARIANTS[VAR_NAMES[i_var]]])/n_seqs)
                    else:
                        frequencies[i_var].append(0)
                
                is_nonzero = [True for i_var in range(len(PANGO_NAMES))]
                for pango_first in T_FIRST_PANGO.keys():
                    if t<T_FIRST_PANGO[pango_first]:
                        idx = PANGO_NAMES.index(pango_first)
                        is_nonzero[idx]=0
                        n_seqs -= np.sum([np.sum(df_t[COL_PAN].str.contains(pango_lineage)) for pango_lineage in PANGO_INCLUDE[pango_first]])
                for i_var in range(len(PANGO_NAMES)):
                    pango_include = PANGO_INCLUDE[PANGO_NAMES[i_var]]
                    pango_exclude = PANGO_EXCLUDE[PANGO_NAMES[i_var]]
                    if is_nonzero[i_var]:
                        n_seqs_pango  = np.sum([np.sum(df_t[COL_PAN].str.contains(pango_lineage)) for pango_lineage in pango_include])
                        n_seqs_pango -= np.sum([np.sum(df_t[COL_PAN].str.contains(pango_lineage)) for pango_lineage in pango_exclude])
                        frequencies[i_var+len(VAR_NAMES)].append(n_seqs_pango/n_seqs)
                    else:
                        frequencies[i_var+len(VAR_NAMES)].append(0)

        if n_good>=t_min:
            t_str = ' '.join([str(_t) for _t in times])
            f_str = [' '.join(['%.4f' % _f for _f in freq]) for freq in frequencies]
            for i_var in range(len(VAR_NAMES)):
                f.write('%s,%s,%s,%s\n' % (loc, t_str, f_str[i_var], VAR_NAMES[i_var]))
            for i_var in range(len(PANGO_NAMES)):
                f.write('%s,%s,%s,%s\n' % (loc, t_str, f_str[i_var+len(VAR_NAMES)], PANGO_NAMES[i_var]))

    # Close file

    f.close()


if __name__ == '__main__': main()

