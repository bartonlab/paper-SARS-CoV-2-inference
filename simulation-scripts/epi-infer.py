#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Check what the above means

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance

def main(args):
    """Infer time-varying selection coefficients from the results of a Wright-Fisher simulation"""
    
    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Selection coefficients inference')
    parser.add_argument('-o',            type=str,    default='inferred-coefficients', help='output file')
    parser.add_argument('-R',            type=float,  default=0,                       help='the basic reproduction number')
    parser.add_argument('-k',            type=float,  default=0.1,                     help='parameter determining shape of distribution of new infected')
    parser.add_argument('--mu',          type=float,  default=1e-4,                    help='the mutation rate')
    parser.add_argument('--data',        type=str,    default=None,                    help='.npz file containing the results of a simulation')
    parser.add_argument('--g1',          type=float,  default=1,                       help='regularization restricting the magnitude of the selection coefficients')
    parser.add_argument('--record',      type=int,    default=1,                       help='number of generations between samples')
    parser.add_argument('--pop_size',    type=int,    default=0,                       help='the population size')
    parser.add_argument('--tv_pop_size', type=str,    default=None,                    help='.npy file containing the time-varying population size')
    parser.add_argument('--mutation_off',  action='store_false', default=True,  help='whether or not to include mutational term in inference')
    
    arg_list = parser.parse_args(args)
    
    out_str = arg_list.o
    g1      = arg_list.g1
    mu      = arg_list.mu
    record  = arg_list.record
    k       = arg_list.k
    mut_off = arg_list.mutation_off
    data    = np.load(arg_list.data, allow_pickle=True)
    nVec        = data['nVec']
    sVec        = data['sVec']
    single_freq = data['traj_record'][:-1]
    delta_x     = data['delta_x']
    covar       = data['covar']
    T     = len(nVec)
    times = np.arange(T) * record
    
    if arg_list.tv_pop_size:
        pop_size = np.load(arg_list.tv_pop_size)
    elif arg_list.pop_size == 0:
        pop_size = data['pop_size'][:-1]
    else:
        pop_size = arg_list.pop_size * np.ones(T-1)
    if arg_list.R == 0:
        R = data['R'][times][:-1]
    else:
        R = arg_list.R
        
    g1 *= np.mean(pop_size)    
    coefficient1 = 1 / ((1 / pop_size * k) + ((k / R) / (pop_size * k - 1)))
    coefficient2 = (pop_size * k) / (pop_size * k + 1)
    covariance_int = np.sum(np.swapaxes(covar, 0, 2) * coefficient1 * coefficient2, axis=2) * record
    delta_x_int = np.sum(np.swapaxes(delta_x, 0, 1) * coefficient1, axis=1)
    if mut_off:
        mutation_int = 0
    else:
        mutation_int = mu * np.sum(coefficient1 * coefficient2 * (1 - 2 * np.swapaxes(single_freq, 0, 1)), axis=1) * record
    RHS = delta_x_int - mutation_int
    regularization = np.zeros((len(covariance_int), len(covariance_int)))
    for i in range(len(covariance_int)):
        regularization[i,i] = g1
    LHS = covariance_int + regularization
    error_bars = np.sqrt(np.diag(np.linalg.inv((LHS))))
    selection = np.linalg.solve(LHS, RHS)
        
    # save the solution  
    g = open(out_str+'.npz', mode='w+b') 
    np.savez_compressed(g, error_bars=error_bars, selection=selection)
    g.close()

if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    

