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
    parser.add_argument('-R',            type=float,  default=1,                       help='the basic reproduction number')
    parser.add_argument('-k',            type=float,  default=0.1,                     help='parameter determining shape of distribution of new infected')
    parser.add_argument('--mu',          type=float,  default=1e-4,                    help='the mutation rate')
    parser.add_argument('--data',        type=str,    default=None,                    help='.npz file containing the results of a simulation')
    parser.add_argument('--g1',          type=float,  default=2,                       help='regularization restricting the magnitude of the selection coefficients')
    parser.add_argument('--freq_cutoff', type=float,  default=0,                       help='if a mutant frequency never rises above this number, it will not be used for inference')
    parser.add_argument('--record',      type=int,    default=1,                       help='number of generations between samples')
    parser.add_argument('--pop_size',    type=int,    default=0,                       help='the population size')
    parser.add_argument('--ss_pop_size', type=str,    default=None,                    help='.npy file containing the simulation specific population size')
    parser.add_argument('--ss_k',        type=str,    default=None,                    help='.npy file containing the simulation specific k')
    parser.add_argument('--ss_R',        type=str,    default=None,                    help='.npy file containing the simulation specific R')
    parser.add_argument('--ss_record',   type=str,    default=None,                    help='.npy file containing the simulation specific record')
    parser.add_argument('--outflow',     type=str,    default=None,                    help='.npz file containing the counts and sequences that outflow')
    parser.add_argument('--inflow',      type=str,    default=None,                    help='.npz file containing the counts and sequences that inflow')
    parser.add_argument('--nm_popsize',  type=str,    default=None,                    help='.npy file containing the population sizes used to correct for infection lasting multiple generations')
    parser.add_argument('--decay_rate',  type=float,  default=0,                       help='the exponential decay rate used to correct for infection lasting multiple generations')
    parser.add_argument('--mutation_on',  action='store_true', default=False,  help='whether or not to include mutational term in inference')
    
    arg_list = parser.parse_args(args)
    
    out_str     = arg_list.o
    g1          = arg_list.g1
    mu          = arg_list.mu
    freq_cutoff = arg_list.freq_cutoff
    record      = arg_list.record
    mut_on      = arg_list.mutation_on
    decay_rate  = arg_list.decay_rate * record
    data        = np.load(arg_list.data, allow_pickle=True)
    mutant_sites_tot = data['mutant_sites']
    mutant_sites_all = data['mutant_sites_all']
    simulations      = data['simulations']
    nVec_full        = data['full_nVec']
    sVec_full        = data['full_sVec']
    
    if arg_list.ss_record:
        record = np.load(arg_list.ss_record)
    else:
        record = arg_list.record * np.ones(simulations)
    if arg_list.ss_pop_size:
        pop_size = np.load(arg_list.ss_pop_size, allow_pickle=True)
    elif arg_list.pop_size != 0:
        pop_size = arg_list.pop_size 
    else:
        pop_size = data['pop_size']
    if arg_list.ss_R:
        R = np.load(arg_list.ss_R)
    elif arg_list.R == 0:
        R = data['R'] 
    else:
        R = arg_list.R
    if arg_list.ss_k:
        k = np.load(arg_list.ss_k)
    else:
        k = arg_list.k
        
    if arg_list.nm_popsize:
        nm_popsize = np.load(arg_list.nm_popsize, allow_pickle=True)
    else:
        nm_popsize = pop_size
    
    if arg_list.inflow:
        inflow_data = np.load(arg_list.inflow, allow_pickle=True)
        in_counts = inflow_data['counts']
        in_sequences = inflow_data['sequences']
        
    
    ### NEEDED FOR ELIMINATING SITES WITH SMALL FREQUENCIES ###
    def trajectory_reshape(traj):
        # reshape trajectories    
        traj_reshaped = []
        for i in range(simulations):
            T_temp = len(traj[i])
            L_temp = len(traj[i][0])
            traj_temp = np.zeros((T_temp, L_temp))
            for j in range(T_temp):
                traj_temp[j] = traj[i][j]
            traj_reshaped.append(np.array(traj_temp))
        return traj_reshaped
    
    def find_significant_sites(traj, mutant_sites):
        # find sites whose frequency is too small
        allele_number = [] 
        for i in range(len(traj)):
            for j in range(len(traj[i][0])):
                if np.amax(traj[i][:, j]) > freq_cutoff:
                    allele_number.append(mutant_sites[i][j])
        allele_number = np.sort(np.unique(np.array(allele_number)))
        return allele_number
    
    def eliminate_sites(nVec, sVec, alleles, mutant_sites):
        # eliminates the low frequency sites
        mutant_sites_new = [[] for i in range(simulations)]
        mutant_sites_elim = [[] for i in range(simulations)]
        nVec_new, sVec_new = [], []
        for i in range(len(traj_reshaped)):
            for j in range(len(traj_reshaped[i][0])):
                if mutant_sites[i][j] in alleles:
                    mutant_sites_new[i].append(mutant_sites[i][j])
                else:
                    mutant_sites_elim[i].append(mutant_sites[i][j])
            nVec_temp, sVec_temp = [], []
            for j in range(len(nVec[i])):
                dead_index = []
                for k in range(len(nVec[i][j])):
                    for h in sVec[i][j][k]:
                        if np.array(mutant_sites_elim[i]).size>0:
                            if h in mutant_sites_elim[i]:
                                dead_index.append(k)
                nVec_t = [nVec[i][j][k] for k in range(len(nVec[i][j])) if k not in dead_index]
                sVec_t = [sVec[i][j][k] for k in range(len(nVec[i][j])) if k not in dead_index]
                nVec_temp.append(nVec_t)
                sVec_temp.append(sVec_t)
            nVec_new.append(nVec_temp)   
            sVec_new.append(sVec_temp)
                    
        return mutant_sites_new, nVec_new, sVec_new
    
    ### END OF FUNCTIONS FOR ELIMINATING SMALL FREQUENCY SITES ###
    
    def construct_terms(single, deltax, covariance, k, R, N, record):
        # creates the RHS and the LHS for a given simulation
        if type(k) == np.ndarray:
            times = np.arange(len(single)+1) * record
            times = times.astype("int")
            k = k[times]
        coefficient1 = (1 / ((1 / (N * k)) + ((k / R) / (N * k - 1))))
        coefficient2 = ((N * k) / (N * k + 1))
        """
        if type(coefficient1) != np.float64 and type(coefficient1) != np.float32 and type(coefficient1) != float:
            coefficient1 = coefficient1[:-1]
            coefficient2 = coefficient2[:-1]
        """
        delta_int = np.sum(np.swapaxes(deltax, 0, 1) * coefficient1, axis=1)
        covar_int = np.sum(np.swapaxes(covariance, 0, 2) * coefficient1 * coefficient2, axis=2) * record
        if not mut_on:
            mutation_int = 0
        else:
            mutation_int = mu * np.sum(coefficient1 * coefficient2 * (1 - 2 * np.swapaxes(single, 0, 1)), axis=1) * record
        RHS = delta_int - mutation_int
        return covar_int, RHS
    
    def array_reshape(variable, dim):
        # reshape trajectories    
        var_reshaped = []
        for i in range(simulations):
            T_temp = len(variable[i])
            L_temp = len(variable[i][0])
            if dim == 1:
                var_temp = np.zeros((T_temp, L_temp))
            else:
                var_temp = np.zeros((T_temp, L_temp, L_temp))
            for j in range(T_temp):
                var_temp[j] = variable[i][j]
            var_reshaped.append(np.array(var_temp))
        return var_reshaped

    def allele_counter(sVec, nVec, mutant_sites_samp):
        """ Counts the single-site and double-site frequencies given the subsampled sVec_s and nVec_s. """
        
        Q = np.array([np.sum(nVec[t]) for t in range(len(nVec))])   # array that contains the total number of sampled sequences at each time point
        single_freq_s = np.zeros((len(nVec), len(mutant_sites_samp)))
        double_freq_s = np.zeros((len(nVec), len(mutant_sites_samp), len(mutant_sites_samp)))
        for t in range(len(nVec)):
            for i in range(len(mutant_sites_samp)):
                single_freq_s[t, i] = np.sum([nVec[t][j] for j in range(len(sVec[t])) if (mutant_sites_samp[i] in sVec[t][j])]) / Q[t]
                for j in range(len(mutant_sites_samp)): 
                    if i != j:
                        double_freq_s[t,i,j] = np.sum([nVec[t][k] for k in range(len(sVec[t]))
                                                       if (mutant_sites_samp[i] in sVec[t][k] and mutant_sites_samp[j] in sVec[t][k])]) / Q[t]
        return single_freq_s, double_freq_s
    
    def allele_counter_in(sVec_in, nVec_in, mutant_sites, N, k, R, T, single):
        """ Counts the single-site frequencies of the inflowing sequences"""
        
        if type(N) == int or type(N) == float or type(N) == np.int32 or type(N) == np.int64:
            popsize = N * np.ones(T)
        else:
            popsize = N
        single_freq_in = np.zeros((T, len(mutant_sites)))
        for t in range(T):
            for i in range(len(mutant_sites)):
                single_freq_in[t, i] = np.sum([nVec_in[t][j] for j in range(len(sVec_in[t])) if (mutant_sites[i] in sVec_in[t][j])]) / (popsize[t])
        coefficient = (1 / ((1 / (N * k + 1)) + ((N * k) / (N * k + 1)) * ((k / R) / (N * k - 1)))) * ((N * k) / (N * k - 1)) * (1 / R)
        if type(coefficient) != np.float64 and type(coefficient) != np.float32 and type(coefficient) != float:
            coefficient = coefficient[:T]
        freq_inflow = np.array([np.sum(nVec_in[t]) for t in range(T)]) / popsize
        term1 = np.sum(np.swapaxes(single_freq_in[:T], 0, 1) * coefficient, axis=1)
        term2 = np.sum(np.swapaxes(single[:T], 0, 1) * coefficient * freq_inflow[:T], axis=1)
        integrated_inflow = term1 - term2
        return integrated_inflow

    def covariance_calc(single_frequencies, double_frequencies):
        """ Calculate the covariance matrix at each generation """
        
        len_mut = len(single_frequencies[0])
        covar_temp = np.zeros((len(single_frequencies), len_mut, len_mut))
        for t in range(len(single_frequencies)):
            f1 = single_frequencies[t]
            f2 = double_frequencies[t]
            for i in range(len_mut):
                for j in range(len_mut):
                    if i == j:
                        covar_temp[t,i,i] = f1[i] * (1 - f1[i])
                    else:
                        covar_temp[t,i,j] = f2[i,j] - f1[i] * f1[j]
        return covar_temp
    
    def add_previously_infected(nVec, sVec, popsize, decay_rate):
        """ Adds previously infected individuals to the population at later dates."""
        
        def combine(nVec1, sVec1, nVec2, sVec2):
            """ Combines sequence and count arrays at a specific time."""
            sVec_new = [i for i in sVec1]
            nVec_new = [i for i in nVec1]
            for i in range(len(sVec2)):
                if list(sVec2[i]) in sVec_new:
                    nVec_new[sVec_new.index(list(sVec2[i]))] += nVec2[i]
                else:
                    nVec_new.append(nVec2[i])
                    sVec_new.append(sVec2[i])
            return nVec_new, sVec_new
        
        if type(popsize)!=type(np.arange(5)):
            popsize = np.ones(len(nVec)) * popsize
        #print(popsize)
        new_nVec = []
        new_sVec = []
        for t in range(1, len(nVec)):
            #nVec_t = [i for i in nVec[t]]
            nVec_t = list(np.array([i for i in nVec[t]]) * popsize[t] / np.sum(nVec[t]))
            sVec_t = [i for i in sVec[t]]
            for t_old in range(t):
                probability = np.exp(- decay_rate * (t - t_old)) * (1 - np.exp(- decay_rate))
                #old_nVec    = list(probability * np.array(nVec[t_old]) * popsize[t_old] / popsize[t])
                old_nVec    = list(probability * np.array(nVec[t_old]) * popsize[t_old] / np.sum(nVec[t_old]))
                old_sVec    = sVec[t_old]
                nVec_t, sVec_t = combine(nVec_t, sVec_t, old_nVec, old_sVec)
            new_nVec.append(nVec_t)
            new_sVec.append(sVec_t)
        return new_nVec, new_sVec

    # Eliminating small frequency sites
    traj = [] 
    for sim in range(simulations):
        traj.append(allele_counter(sVec_full[sim], nVec_full[sim], mutant_sites_tot[sim])[0])
    traj_reshaped = trajectory_reshape(traj)
    
    allele_number = find_significant_sites(traj_reshaped, mutant_sites_tot)
    
    mutant_sites_tot, nVec_full, sVec_full = eliminate_sites(nVec_full, sVec_full, allele_number, mutant_sites_tot)
    
    # Constructing the terms in the system of equations
    single_tot, covar_tot, delta_tot = [], [], []
    
    for sim in range(simulations):
        nVec, sVec = nVec_full[sim], sVec_full[sim]
        mutant_sites_samp = mutant_sites_tot[sim]
        if type(pop_size) == int or type(pop_size) == np.int32 or type(pop_size) == np.int64:
            pop_size_sim = pop_size * np.ones(len(nVec))
        elif len(np.shape(pop_size)) == 2:
            pop_size_sim = pop_size[sim]
        elif len(np.shape(pop_size)) == 1:
            pop_size_sim = pop_size[sim] * np.ones(len(nVec))
            
        single_new, double_new = allele_counter(sVec, nVec, mutant_sites_samp)   # the sampled single and double site frequencies of newly infected
        
        if decay_rate!=0:
            #if type(nm_popsize)==type(np.arange(5)):
            if not isinstance(nm_popsize, int) and not isinstance(nm_popsize, float):
                if len(np.shape(nm_popsize))>1:
                    temp_popsize = nm_popsize[sim]
                else:
                    temp_popsize = nm_popsize
            else: 
                temp_popsize = nm_popsize
            nVec_new, sVec_new = add_previously_infected(nVec, sVec, temp_popsize, decay_rate)
            single_freq, double_freq = allele_counter(sVec_new, nVec_new, mutant_sites_samp)    # the frequencies from the total number of infectious at each time
        else:
            single_freq, double_freq = single_new, double_new
    
        """
        delta_x = np.zeros((len(nVec)-1, len(mutant_sites_samp)))
        single_mid = np.zeros((len(nVec)-1, len(mutant_sites_samp)))
        double_mid = np.zeros((len(nVec)-1, len(mutant_sites_samp), len(mutant_sites_samp)))
        pop_mid = np.zeros(len(nVec)-1)
        for i in range(len(nVec)-1):
            delta_x[i] = single_freq[i+1] - single_freq[i]
            single_mid[i] = (single_freq[i] + single_freq[i+1]) / 2
            double_mid[i] = (double_freq[i] + double_freq[i+1]) / 2
            pop_mid[i] = (pop_size_sim[i] + pop_size_sim[i+1]) / 2
        """
        #single_mid, double_mid = single_freq, double_freq
        delta_x = np.zeros((len(nVec), len(mutant_sites_samp)))
        for i in range(1, len(nVec)):
            delta_x[i] = single_new[i] - single_freq[i-1]
        """    
        if decay_rate!=0:
            if type(nm_popsize)==type(np.arange(5)):
                if len(np.shape(nm_popsize))>1:
                    temp_popsize = nm_popsize[sim]
                else:
                    temp_popsize = nm_popsize
            else: 
                temp_popsize = nm_popsize
            nVec, sVec = add_previously_infected(nVec, sVec, temp_popsize, decay_rate)
            single_new, double_new = allele_counter(sVec, nVec, mutant_sites_samp)
        else:
            single_new, double_new = single_mid, double_mid
        """
        covar = covariance_calc(single_freq, double_freq)
            
        #single_tot.append(np.array([np.asanyarray(single_mid[i]) for i in range(len(single_mid))]))
        single_tot.append(list(single_freq))
        covar_tot.append(list(covar))
        delta_tot.append(list(delta_x))
    
    single_tot = array_reshape(single_tot, 1)
    delta_tot  = array_reshape(delta_tot, 1)
    covar_tot  = array_reshape(covar_tot, 2)
    L = len(mutant_sites_all)
    A = np.zeros((L,L))
    b = np.zeros(L)
    selection_individual = np.zeros((simulations, L))
    error_individual     = np.zeros((simulations, L))
    g1 *= np.mean(pop_size) * 2 * np.mean(k)
    
    for sim in range(simulations):
        if type(k)!=float and type(k)!=int:
            if len(np.shape(k)) == 2 or (len(np.shape(k)) == 1 and simulations != 1):
                k_ss = k[sim]
            else:
                k_ss = k
        else:
            k_ss = k
        if type(R)!=float and type(R)!=int:
            R_ss = R[sim]
        else:
            R_ss = R
        if type(record)!=int:
            ss_record = record[sim]
        else:
            ss_record = record
        if type(pop_size)!=int:
            ss_pop_size = pop_size[sim]
        else:
            ss_pop_size = pop_size
        if arg_list.inflow:
            ss_incounts = in_counts[sim]
            ss_inseqs = in_sequences[sim]
        #T = int(record[sim] * len(nVec_full[sim]) - 1) 
        T = int(record[sim] * len(nVec_full[sim])) 
        
        # process the inflowing sequences
        mutant_sites = mutant_sites_tot[sim]
        if arg_list.inflow:
            #mutant_sites_in = np.sort(np.unique(np.array([ss_inseqs[t][i][j] for t in range(len(ss_inseqs))
            #                                         for i in range(len(ss_inseqs[t])) for j in range(len(ss_inseqs[t][i]))] )))
            #nVec = nVec_full[sim]
            inflow_term = allele_counter_in(ss_inseqs, ss_incounts, mutant_sites, ss_pop_size, k_ss, R_ss, T, single_tot[sim])
        else:
            inflow_term = np.zeros(len(mutant_sites))
        
        A_ind = np.zeros((L, L))
        b_ind = np.zeros(L)
        
        covar_int, RHS_temp = construct_terms(single_tot[sim], delta_tot[sim], covar_tot[sim], k_ss, R_ss, ss_pop_size, ss_record)
        for i in range(len(mutant_sites)):
            loc = np.where(mutant_sites_all == mutant_sites[i])
            b[loc] += RHS_temp[i] - inflow_term[i]
            b_ind[loc] = RHS_temp[i]
            for j in range(len(mutant_sites)):
                loc2 = np.where(mutant_sites_all == mutant_sites[j])
                A[loc, loc2] += covar_int[i,j]
                A_ind[loc, loc2] = covar_int[i,j]
        for i in range(L):
            A_ind[i,i] += g1
        selection_individual[sim] = np.linalg.solve(A_ind, b_ind)
        error_individual[sim] = np.sqrt(np.diag(np.linalg.inv(A_ind)))
                
    for i in range(L):
        A[i,i] += g1
    error_bars = np.sqrt(np.diag(np.linalg.inv(A)))
    selection = np.linalg.solve(A, b)
 
    # save the solution  
    g = open(out_str+'.npz', mode='w+b') 
    np.savez_compressed(g, error_bars=error_bars, selection=selection, errors_ind=error_individual, selection_ind=selection_individual, pop_size=pop_size)
    g.close()

if __name__ == '__main__': 
    main(sys.argv[1:])
    

    
    
    
    
    

