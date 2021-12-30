#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import argparse
import numpy as np                          # numerical tools
import scipy as sp
from timeit import default_timer as timer   # timer for performance
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
import os

def main():
    """Infer time-varying selection coefficients from the results of a Wright-Fisher simulation"""
    
    def mse(a,b): 
        """Computes mean squared error, a is shape=(L,T) and b is shape=(T,L). Use the inferred for a and actual for b"""
    
        time = len(a[0])
        error_sum = np.zeros(len(a))
        for t in range(time):
            for i in range(len(a)):
                error_sum[i] += np.absolute(a[i,t] - b[t,i])
        return error_sum / time
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',           type=str,        default='analysis',    help='output string')
    parser.add_argument('-N',           type=int,        default=1000,          help='number of individuals in the population')
    parser.add_argument('-g1',          type=float,      default=1,             help='regularization restricting the magnitude of the selection coefficients')
    parser.add_argument('-k',           type=float,      default=0.1,           help='parameter determining how long-tailed the distribution is')
    parser.add_argument('-R',           type=float,      default=2,             help='basic reproduction number')
    parser.add_argument('-T',           type=int,        default=100,           help='number of generations')
    parser.add_argument('--pop_limit',  type=int,        default=1e5,           help='the limit for the population size')
    parser.add_argument('--mu',         type=float,      default=0.001,         help='mutation rate')
    parser.add_argument('--simulations',type=int,        default=1,             help='number of simulations to run and combine in the ')
    parser.add_argument('--timed',      type=int,        default=0,             help='whether or not to time the simulations. default is 0, no timer')
    parser.add_argument('--sample',     type=int,        default=0,             help='number of individuals to sample at each generation')
    parser.add_argument('--record',     type=int,        default=1,             help='record sequence data only every {record} generations')
    parser.add_argument('--n_runs',     type=int,        default=1,             help='number of times to run the simulations')
    parser.add_argument('-out1',        type=str,        default='.\out1', help='output path for the wright fisher simulations')
    parser.add_argument('-out2',        type=str,        default='.\out2', help='output path for the wright fisher simulations')
    parser.add_argument('-i',           type=str,        default=None,                    help='the initial population, including counts and sequences')
    parser.add_argument('--PopSize',    type=str,        default=None,                    help='.npy file containing the time-varying population size')
    parser.add_argument('--InferPop',   type=str,        default=None,                    help='.npy file containing the incorrect population also used in inference')
    parser.add_argument('--in_flow',    type=str,        default=None,                    help='.npz file containing the population to add in at every time')
    parser.add_argument('--script_dir', type=str,        default='simulation-scripts',                    help='directory containing inference and simulation scripts')
    
    arg_list     = parser.parse_args(sys.argv[1:])
    
    N            = arg_list.N
    g1           = arg_list.g1
    mu           = arg_list.mu
    k            = arg_list.k
    R            = arg_list.R
    T            = arg_list.T
    pop_limit    = arg_list.pop_limit
    simulations  = arg_list.simulations
    timed        = arg_list.timed
    out_str      = arg_list.o
    sample       = arg_list.sample
    out1         = arg_list.out1
    out2         = arg_list.out2
    record       = arg_list.record
    initial      = arg_list.i
    pop_size     = arg_list.PopSize
    infer_pop    = arg_list.InferPop
    n_runs       = arg_list.n_runs
    in_flow      = arg_list.in_flow
    script_dir   = arg_list.script_dir
    if not script_dir:
        script_dir = os.cwd()
    if pop_size:
        N = pop_size

    
    # import the Wright-Fisher simulations and time varying inference modules and name them
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("epi", os.path.join(script_dir, "branching.py"))
    epi  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(epi)
    
    spec2 = importlib.util.spec_from_file_location("epi-infer", os.path.join(script_dir, "epi-infer-multiple.py"))
    inference = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(inference)
    
    # initialize arrays that will be used later
    errors, inferred, traj         = [], [], []
    errors_inflow, inferred_inflow = [], []
    actual, mutant_sites           = [], []
    pop_size, R_full               = [], []
        
    # run simulations and record results
    print(out1, out2)
    for i in range(n_runs):
        
        # Do the Wright-Fisher simulation and load results
        if initial is not None and in_flow is not None:
            epi.main(["-o", out1, "--timed", str(timed), "--mu",str(mu),"--sample", str(sample), "--record", str(record), "-i", str(initial),"--pop_limit", str(int(pop_limit)), "-k", str(k), "-R", str(R), "--simulations", str(simulations), "-T", str(T), "--in_flow", in_flow])
        elif initial is not None:
            epi.main(["-o", out1, "--timed", str(timed),"--pop_size",str(pop_size),"--mu",str(mu),"--sample", str(sample), "--record", str(record), "-i", str(initial),"--pop_limit", str(pop_limit), "-k", str(k), "-R", str(R), "--simulations", str(simulations), "-T", str(T)])
        else:
            epi.main(["-o", out1, "--timed", str(timed),"--pop_size",str(pop_size),"--mu",str(mu),"--sample", str(sample), "--record", str(record), "--pop_limit", str(pop_limit), "-k", str(k), "-R", str(R), "--simulations", str(simulations), "-T", str(T)]) 
              
        data = np.load(out1 + ".npz", allow_pickle="True")
        traj.append([list(i) for i in data['traj_record']])
        actual.append([i for i in data['selection_all']])
        mutant_sites.append([list(i) for i in data['mutant_sites']])
        pop_size.append([i for i in data['pop_size']])
        R_full.append([i for i in data['R']])
        
        # Do the inference and load results
        inference.main(["--data", out1+".npz","--g1",str(g1),"-o",out2, "--record", str(record), "--pop_size", str(10000)])
        inference.main(["--data", out1+".npz","--g1",str(g1),"-o",out2+'-corrected', "--record", str(record), "--inflow", in_flow, '-R', str(1)])
        
        inf_data = np.load(out2 + '.npz', allow_pickle="True")
        inf_inflow_data = np.load(out2 + '-corrected.npz', allow_pickle="True")
        errors_inflow.append(inf_inflow_data['error_bars'])
        errors.append(inf_data['error_bars'])
        inferred_inflow.append(inf_inflow_data['selection'])
        inferred.append(inf_data['selection'])
    
    print(out_str)
    # save the MSE array, trajectory array, and averaged selection coefficients array
    f = open(out_str+'.npz', mode='w+b')
    np.savez_compressed(f, traj=traj, inferred=inferred, errors=errors, actual=actual, mutant_sites=mutant_sites, pop_size=pop_size, R=R_full)
    f.close()
    g = open(out_str+'-corrected.npz', mode='w+b')
    np.savez_compressed(g, traj=traj, inferred=inferred_inflow, errors=errors_inflow, actual=actual, mutant_sites=mutant_sites, pop_size=pop_size, R=R_full)
    g.close()
    
if __name__ == '__main__': main()  

