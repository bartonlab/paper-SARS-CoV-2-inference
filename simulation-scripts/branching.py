#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# A simple Wright-Fisher simulation with an additive fitness model

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
import copy

def usage():
    print("")

def main(args):
    """ Simulates an epidemiological branching process. """

    # Read in simulation parameters from command line
    parser = argparse.ArgumentParser(description='Wright-Fisher evolutionary simulation.')
    parser.add_argument('-o',   type=str,     default='Epidem',  help='output destination')
    parser.add_argument('-N',   type=int,     default=1000,      help='initial population size')
    parser.add_argument('-L',   type=int,     default=3e4,       help='genome length')
    parser.add_argument('--mu', type=float,   default=1.0e-4,    help='mutation rate')
    parser.add_argument('-i',   type=str,     default=None,      help='file containing initial distribution. Must contain sequences, counts, and selection (which must be the selection coefficients in the order of the numerical order of the sites at which there are mutations). If counts is an array of arrays, then different arrays correspond to different initial distributions for different simulations')
    parser.add_argument('-R',   type=float,   default=2.0,       help='the average reproduction number')
    parser.add_argument('-k',   type=float,   default=0.1,       help='dispersion parameter determining the tail of the distribution')
    parser.add_argument('-T',   type=int,     default=1000,      help='the number of generations to evolve the population for')
    parser.add_argument('--sample',      type=int,   default=0,      help='number of sampled population for the calculation of covariance matrix')
    parser.add_argument('--record',      type=int,   default=1,      help='record sequence data every {record} generations')
    parser.add_argument('--timed',       type=int,   default=0,      help='set to 1 to time the simulation, and set to 2 to be informed of the progress every generation')
    parser.add_argument('--TVsample',    type=str,   default=None,   help='.npy file containing the time-varying sample sizes, length must equal number of recorded times')
    parser.add_argument('--TVR',         type=str,   default=None,   help='.npy file containing the time-varying reproduction number')
    parser.add_argument('--TVk',         type=str,   default=None,   help='.npy file containing the time-varying k-value')
    parser.add_argument('--pop_size',    type=str,   default=None,   help='.npy file containing the desired population size at different times. The actual population size will oscillate around these values if R is large enough, otherwise it will fall below these values. Must have length T+1.')
    parser.add_argument('--simulations', type=int,   default=1,      help='number of simulations with different populations to record')
    parser.add_argument('--Sgaussian',   type=float, default=0.,     help='standard deviation for a normal distribution from which to draw the selection coefficients')
    parser.add_argument('--Sdiscrete',   type=str,   default=None,   help='.npy file containing the support and probabilities for a probability mass function to draw the selection coefficients from')
    parser.add_argument('--options',     type=str,   default=None,   help='.npz file containing options to use for different simulations. Keys can be ss_sample, ss_record, ss_R, ss_k, ss_T, ss_pop_size. The corresponding elements must be the simulation specific values for these options, if they are one-dimensional arrays. If they are two-dimensional arrays, the first corresponds to the simulation number and the second to the time-step, where this is applicable. ss_record and ss_T cannot be time-varying. For the population size, if it is time-varying, the actual population size will oscillate around these values if R is large enough, otherwise it will fall below these values. For time-varying variables shared across simulations, use the TV optional arguments insead.')
    parser.add_argument('--transfer',    type=str,   default=None,   help='.npy file containing the genotypes to be transferred between the populations. if 1D, then the different entries correspond to the amount of sequences to transfer for each population. Either of shape (simulations-1, simulations-1), or (simulations-1, simulations-1, T_max). The [i,j,k] entry is the number of genomes to be transferred from simulation i to simulation i+j+1 at time k.')
    parser.add_argument('--pop_limit',   type=int,   default=1e5,    help='an upper limit for how large to allow the population to get')
    parser.add_argument('--in_flow',     type=str,   default=None,   help='.npz file containing the sequences flowing in to the populations. Must contain counts, sequences, and selection. Selection should be a 1D array containing the selection coefficients for the ordered sites at which there are mutations. Counts should have shape (simulations, time, number of different genomes).')
    parser.add_argument('--CovarianceOff',  action='store_true',  default=False, help='whether or not compute covariance matrix after simulation')
    parser.add_argument('--covar_int_off',  action='store_false', default=True,  help='whether or not to compute integrated covariance matrix')
    parser.add_argument('--DetailOff',      action='store_true',  default=False, help='whether or not to save addititional information at the end of the simulation')
    parser.add_argument('--FullTrajectory', action='store_true',  default=False, help='whether or not to store frequency trajectories at every generation')

    arg_list = parser.parse_args(args)

    out_str     = arg_list.o 
    mu          = arg_list.mu
    timed       = arg_list.timed
    N           = arg_list.N
    L           = arg_list.L
    sample      = arg_list.sample
    s_discrete  = arg_list.Sdiscrete
    simulations = arg_list.simulations
    covariance_off = arg_list.CovarianceOff
    covar_int_off  = arg_list.covar_int_off
    DetailOff = arg_list.DetailOff
    FullTrajectory = arg_list.FullTrajectory

    if arg_list.i:
        initial = np.load(arg_list.i, allow_pickle=True)

    if arg_list.options:
        options = np.load(arg_list.options, allow_pickle=True)
        keys = [i for i in options]
        if 'ss_T' in keys:
            ss_T  = options["ss_T"]
            T_max = np.amax(ss_T)
        else:
            ss_T  = None
            T_max = arg_list.T
            T     = arg_list.T
        if 'ss_record' in keys:
            ss_record = options["ss_record"]
        else:
            ss_record = None
            record    = arg_list.record
        if 'ss_sample' in keys:
            ss_sample = options["ss_sample"]
        else:
            ss_sample = None
            if arg_list.TVsample:
                sample_size = np.load(arg_list.TVsample)
            else:
                sample_size = arg_list.sample * np.ones(T_max+1)
        if 'ss_k' in keys:
            ss_k = options["ss_k"]
        else:
            ss_k = None
            if arg_list.TVk:
                k_full = np.load(arg_list.TVk)
            else:
                k_full = np.ones(T_max+1) * arg_list.k
        if 'ss_R' in keys:
            ss_R = options["ss_R"]
        else:
            ss_R = None
            if arg_list.TVR:
                R_full = np.load(arg_list.TVR)
            else:
                R_full = np.ones(T_max+1) * arg_list.R
        if 'ss_pop_size' in keys:
            ss_pop_size = options["ss_pop_size"]
        else:
            ss_pop_size = None
            if arg_list.pop_size:
                pop_size = np.load(arg_list.pop_size)
            else:
                pop_size = arg_list.pop_limit * np.ones(T_max+1)
    else:
        ss_record, ss_sample, ss_k, ss_T, ss_R, ss_pop_size = None, None, None, None, None, None
        T_max, T = arg_list.T, arg_list.T
        if arg_list.pop_size:
            pop_size = np.load(arg_list.pop_size)
        else:
            pop_size = arg_list.pop_limit * np.ones(T_max+1)
        if arg_list.TVsample:
            sample_size = np.load(arg_list.TVsample)
        else:
            sample_size = arg_list.sample *np.ones(T_max+1)
        if arg_list.TVR:
            R_full = np.load(arg_list.TVR)
        else:
            R_full = np.ones(T_max+1) * arg_list.R
        if arg_list.TVk:
            k_full = np.load(arg_list.TVk)
        else:
            k_full = np.ones(T_max+1) * arg_list.k
        record = arg_list.record
    
    if arg_list.Sgaussian!=0:
        s_gaussian = arg_list.Sgaussian
        s_mean = 0
        s_stddev = s_gaussian
        s_type = "gaussian"
    elif arg_list.Sdiscrete:
        s_discrete = np.load(s_discrete)
        s_type = "discrete"
    else:
        s_discrete = np.array([[-0.03, 0, 0.03], [0.2, 0.6, 0.2]])
        s_type = "discrete"
    
    if arg_list.transfer:
        input_transfer = np.load(arg_list.transfer)
        if len(np.shape(input_transfer)) == 2:
            transfer = np.zeros((len(input_transfer), len(input_transfer), T_max+1))
            for i in range(T_max+1):
                transfer[:, :, i] = input_transfer
        elif len(np.shape(input_transfer)) == 3:
            transfer = input_transfer
        else:
            print("shape of transfer array is incompatible")
        
    selection = {}   # dictionary whose keys are the mutated site numbers and whose values are the selection coefficient at that site
    
    # _ FUNCTIONS _ #
    
    def printUpdate(current, end, bar_length=20):
        """ Print an update of the simulation status. h/t Aravind Voggu on StackOverflow. """
        
        percent = float(current) / end
        dash    = ''.join(['-' for k in range(int(round(percent * bar_length)-1))]) + '>'
        space   = ''.join([' ' for k in range(bar_length - len(dash))])
        sys.stdout.write("\rSimulating: [{0}] {1}%".format(dash + space, int(round(percent * 100))))
        sys.stdout.write("\n")
        sys.stdout.flush()

    def No2Index(nVec_t, no):
        """ Find out what group a sampled individual is in (what sequence does it have) """
        
        tmp_No2Index = 0
        for i in range(len(nVec_t)):
            tmp_No2Index += nVec_t[i]
            if tmp_No2Index > no:
                return i
            
    def draw_samples_discrete(data=[], num_samples=1):
        """Draws samples from a probability mass function where data is a list of shape (2,n),
        data[0] is a list of the support of the function, and data[1] is a list of the corresponding probabilities"""
        
        randoms = np.random.rand(num_samples)
        support = data[0]
        probs = data[1]
        result = []
        total_prob = 0
        for i in randoms:
            j = -1
            while i>total_prob:
                total_prob += probs[j]
                j += 1
            else:
                result.append(support[j-1])
        return np.array(result)
    
    def draw_samples_gaussian(mean=0, sd=0.1, num_samples=1):
        """Draws samples from a continuous probability distribution, where function is the variable name of the distribution"""
        
        return np.random.normal(mean, sd, size=num_samples)
    
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
    
    def fitness(selection, seq):
        """ Calculate fitness for a binarized seq wrt. a given selection matrix at every time."""

        h = 1
        seq = [int(i) for i in seq]
        mutated_sites = [int(i) for i in selection]
        for i in seq:
            if i not in mutated_sites:
                if s_type == "gaussian":
                    new_s = draw_samples_gaussian(mean=s_mean, sd=s_stddev)[0]
                elif s_type == "discrete":
                    new_s = draw_samples_discrete(data=s_discrete)[0]
                selection[i] = new_s
        for i in range(len(seq)):
            h += selection[seq[i]]
        return h
    
    def SampleSequences(sVec, nVec, TV_sample):
        """ Sample a certain number of sequences from the whole population. """

        sVec_sampled = []
        nVec_sampled = []
        for t in range(len(nVec)):
            nVec_tmp = []
            sVec_tmp = []
            if pop_size_record[t] > TV_sample[t] and TV_sample[t] > 0:
                nos_sampled = np.random.choice(int(pop_size_record[t]), int(TV_sample[t]), replace=False)
                indices_sampled = [No2Index(nVec[t], no) for no in nos_sampled]
                indices_sampled_unique = np.unique(indices_sampled)
                for i in range(len(indices_sampled_unique)):
                    nVec_tmp.append(np.sum([index == indices_sampled_unique[i] for index in indices_sampled]))
                    sVec_tmp.append(sVec[t][indices_sampled_unique[i]])
            else:
                for i in range(len(nVec[t])):
                    nVec_tmp.append(nVec[t][i])
                    sVec_tmp.append(sVec[t][i])
            nVec_sampled.append(np.array(nVec_tmp))
            sVec_sampled.append(np.array(sVec_tmp, dtype=object))
        return sVec_sampled, nVec_sampled
    
    def allele_counter(sVec, nVec):
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
    
    def transfer_out(n_transfer, nVec, sVec):
        """ Create sequences and counts to transfer to a different population. """
        
        total_pop = np.sum(nVec)
        nVec_trans_temp, sVec_trans_temp, pop_trans_temp = [], [], []
        num_transfer = np.random.choice(int(total_pop), n_transfer, replace=False)
        ind_sampled = [No2Index(nVec, no) for no in num_transfer]
        ind_sampled_unique = np.unique(ind_sampled)
        for i in range(len(ind_sampled_unique)):
            nVec_trans_temp.append(np.sum([index == ind_sampled_unique[i] for index in ind_sampled]))
            sVec_trans_temp.append(sVec[ind_sampled_unique[i]])
            nVec[ind_sampled_unique[i]] -= np.sum([index == ind_sampled_unique[i] for index in ind_sampled])
        for i in range(len(nVec_trans_temp)):
            pop_trans_temp.append(Species(n = nVec_trans_temp[i], f = fitness(selection, sVec_trans_temp[i]), 
                                          sequence = sVec_trans_temp[i]))
        zero_indices = np.where(nVec == 0)[0]
        nVec = [nVec[i] for i in range(len(nVec)) if i not in zero_indices]
        sVec = [sVec[i] for i in range(len(sVec)) if i not in zero_indices]
        """
        for i in zero_indicies:
            del nVec[i]
            del sVec[i]
        """
        return pop_trans_temp
    
    def transfer_in(pop_transfer, pop):
        """ Combine sequences transfered from another population with the current population"""
        
        for i in range(len(pop_transfer)):
            unique = True
            for j in range(len(pop)):
                if set(pop_transfer[i].sequence) == set(pop[j].sequence):
                    unique = False
                    pop[j].n += pop_transfer[i].n
                    break
            if unique:
                pop.append(pop_transfer[i])

    # ^ FUNCTIONS ^ #
    
    # _ SPECIES CLASS _ #
    
    class Species:

        def __init__(self, n = 1, f = 1, **kwargs):
            """ Initialize clone/provirus-specific variables. """
            self.n = n   # number of members

            if 'sequence' in kwargs:
                self.sequence = kwargs['sequence']  # sequence identifier
                self.f        = fitness(selection, self.sequence)
            else:
                self.sequence = []
                self.f        = 1

        @classmethod
        def clone(cls, s):
            return cls(n = 1, f = s.f, sequence = [k for k in s.sequence]) # Return a new copy of the input Species

        def mutate(self):
            """ Mutate and return self + new sequences.""" 

            newSpecies = []
            if self.n>0:
                nMut    = np.random.binomial(self.n, mu) # get number of individuals that mutate
                self.n -= nMut # subtract number mutated from size of current clone

                # process mutations
                site = np.random.randint(L, size = nMut) # choose mutation sites at random
                for i in site:
                    s = Species.clone(self) # create a new copy sequence
                    s.sequence.append(i)  # mutate the randomly-selected site
                    s.f = fitness(selection, s.sequence)
                    newSpecies.append(s)

            # return the result
            if self.n>0:
                newSpecies.append(self)
            return newSpecies

    # ^ SPECIES CLASS ^ #
    
    # Trial length and recording frequency
    if timed>0:
        start  = timer() # track running time

    # setup inflow population    
    if arg_list.in_flow:
        in_flow_data = np.load(arg_list.in_flow, allow_pickle=True)
        sVec_in = in_flow_data['sequences']
        selection_in = in_flow_data['selection']
        sVec_in_flattened = np.sort(np.unique(np.array([sVec_in[i][j][k][l] for i in range(len(sVec_in)) for j in range(len(sVec_in[i]))
                                                        for k in range(len(sVec_in[i][j])) for l in range(len(sVec_in[i][j][k]))] )))
        for i in range(len(sVec_in_flattened)):
            selection[sVec_in_flattened[i]] = selection_in[i]
        pop_inflow = []
        for sim in range(simulations):
            pop_inflow_sim = []
            for i in range(len(in_flow_data['counts'][sim])):
                nVec_in = in_flow_data['counts'][sim][i]
                sVec_in = in_flow_data['sequences'][sim][i]
                pop_inflow_temp = []
                for j in range(len(nVec_in)):
                    n_seq = nVec_in[j]
                    temp_seq = [int(k) for k in sVec_in[j]]
                    pop_inflow_temp.append(Species(n = n_seq, f = fitness(selection, temp_seq), sequence = temp_seq))
                pop_inflow_sim.append(pop_inflow_temp)
            pop_inflow.append(pop_inflow_sim)
            
    # Create population
    if arg_list.i:
        selection_co = initial['selection']
        # Use the given intial distribution
        if type(initial['counts'][0]) == int or type(initial['counts'][0]) == float or type(initial['counts'][0]) == np.int32 or type(initial['counts'][0]) == np.int64:
            pop_init, sVec_init, nVec_init = [], [], []
            mutated_sites_all = [initial['sequences'][j][i] for j in range(len(initial['sequences'])) for i in range(len(initial['sequences'][j]))]
            mutated_sites = np.sort(np.unique(mutated_sites_all))
            for i in range(len(mutated_sites)):
                selection[mutated_sites[i]] = selection_co[i]
            temp_pop = initial
            temp_sVec = []
            temp_nVec = []
            for i in range(len(temp_pop['counts'])):
                temp_seq = np.array([int(j) for j in temp_pop['sequences'][i]])
                n_seq = temp_pop['counts'][i]
                pop_init.append(Species(n = n_seq, f = fitness(selection, temp_seq), sequence = temp_seq))
                temp_sVec.append(temp_seq)
                temp_nVec.append(n_seq)
            sVec_init.append(np.array(temp_sVec))
            nVec_init.append(np.array(temp_nVec))
            common_initial = True
        else:
            mutated_sites_all = [initial['sequences'][j][k][i] for j in range(len(initial['sequences']))
                                 for k in range(len(initial['sequences'][j])) for i in range(len(initial['sequences'][j][k]))]
            mutated_sites = np.sort(np.unique(mutated_sites_all))
            for i in range(len(mutated_sites)):
                selection[mutated_sites[i]] = selection_co[i]
            common_initial = False
            
    else:
        # Start with all sequences being wild-type
        pop_init  = [Species(n = N)]           # current population
        sVec_init = [np.array([[]])]           # array of sequences at each time point
        nVec_init = [np.array([N])]            # array of sequence counts at each time point
        common_initial = True
    
    single_tot = []
    delta_tot = []
    covar_tot = []
    record_tot = []
    times_tot = []
    pop_size_record_tot = []
    mutant_sites_tot = []
    full_nVec, full_sVec, full_pop, full_true_R = [], [], [], []
    full_pop_trans = []
    # Evolve the populations for the different simulations
    for sim in np.arange(simulations):
        if ss_T is not None:
            T = ss_T[sim]
        if ss_k is not None:
            if len(np.shape(ss_k))==1:
                k_full = ss_k[sim] * np.ones(T+1)
            elif np.len(np.shape(ss_k))==2:
                k_full = ss_k[sim]
            else:
                raise ValueError
                print("shape of k is incompatible")
        if ss_R is not None:
            if len(np.shape(ss_R))==1:
                R_full = ss_R[sim] * np.ones(T+1)
            elif len(np.shape(ss_R))==2:
                R_full = ss_R[sim]
            else: 
                raise ValueError
                print("shape of R is incompatible")
        if ss_pop_size is not None:
            if len(np.shape(ss_pop_size))==1:
                pop_size = ss_pop_size[sim] * np.ones(T+1)
            elif len(np.shape(ss_pop_size))==2:
                pop_size = ss_pop_size[sim]
            else:
                raise ValueError
                print("shape of pop_size is incompatible")
        if ss_sample is not None:
            if type(ss_sample[sim])!=list and type(ss_sample[sim])!=np.ndarray:
                sample_size = ss_sample[sim] * np.ones(T+1)
            else:
                sample_size = ss_sample[sim]
        if ss_record is not None:
            record = ss_record[sim]
       
        R_actual = [R_full[0]]
        if not common_initial:
            pop_init, sVec_init, nVec_init = [], [], []
            temp_sequences = initial['sequences'][sim]
            temp_counts = initial['counts'][sim]
            temp_pop = {}
            temp_pop['sequences'] = temp_sequences
            temp_pop['counts'] = temp_counts
            temp_sVec = []
            temp_nVec = []
            for i in range(len(temp_pop['counts'])):
                temp_seq = np.array([int(j) for j in temp_pop['sequences'][i]])
                n_seq = temp_pop['counts'][i]
                pop_init.append(Species(n = n_seq, f = fitness(selection, temp_seq), sequence = temp_seq))
                temp_sVec.append(temp_seq)
                temp_nVec.append(n_seq)
            sVec_init.append(np.array(temp_sVec))
            nVec_init.append(np.array(temp_nVec)) 
        
        sims_transfer_to = simulations - 1 - sim
        #sims_transfer_from = sim
        pop_trans = []
        
        fnVec, fsVec = [i for i in nVec_init], [i for i in sVec_init]
        fpop = copy.deepcopy(pop_init)
        total_population = np.sum(np.array([s.n for s in fpop]))
        
        pop_max_hit = False
        tStart = 1       # start generation
        tEnd   = T+1       # end generation
        for t in range(tStart, tEnd):
            if timed==2:
                printUpdate(t, tEnd)    # status check

            # Update parameters so that population size stays near desired level.
            average_fitness = np.sum(np.array([s.f * s.n for s in fpop])) / np.sum(np.array([s.n for s in fpop]))
            k = k_full[t]
            if total_population >= pop_size[t]:
                pop_max_hit = True
            """
            if total_population <= pop_size[t]:
                if pop_max_hit:
                    R = (1 / average_fitness) + 0.01
                    #The below is needed if the population size grows quickly
                    #R = (1 / average_fitness) + 0.1   
                else:
                    R = R_full[t]
            else:
                pop_max_hit = True
                R = (1 / average_fitness) - 0.01
            """
            if pop_max_hit:
                if t+1 < tEnd:
                    ### Consider instead multiplying by pop_size[t+1] / pop_size[t+1]
                    R = (1 / average_fitness) * (pop_size[t] / total_population)
                else:
                    R = 1
            else:
                R = R_full[t]
                
            R_actual.append(R)
        
            # Select species to replicate
            r = np.array([s.n * k for s in fpop])  # negative binomial rate paramter
            p = np.array([(k / (k + R * s.f)) for s in fpop])  # negative binomial success probability 
            r[r==0] = 0.00001     # r=0 is not allowed in the negative biniomial distribution
            total_population = 0
            while total_population < 10:   # makes sure that the population doesn't die off by resampling if there are too few infected
                n = np.random.negative_binomial(r, p) # selected number of each species
                total_population = np.sum(n)

            # Update population size and mutate
            newPop = []
            for i in range(len(fpop)):
                fpop[i].n = n[i] # set new number of each species
                # include mutations, then add mutants to the population array
                p = fpop[i].mutate()
                for j in range(len(p)):
                    unique = True
                    for k in range(len(newPop)):
                        #if np.array_equal(np.sort(p[j].sequence), np.sort(newPop[k].sequence)):
                        if set(p[j].sequence) == set(newPop[k].sequence):
                            unique       = False
                            newPop[k].n += p[j].n
                            break
                    if unique:
                        newPop.append(p[j])
            fpop = newPop
            
            if arg_list.transfer:
                
                # transfer populations to next simulations
                pop_trans_t = []
                temp_sVec = [s.sequence for s in fpop]
                temp_nVec = [s.n for s in fpop]
                for i in range(sims_transfer_to):
                    n_trans = int(transfer[sim, i, t])
                    if n_trans > 0:
                        pop_trans_temp = transfer_out(n_trans, temp_nVec, temp_sVec)
                        pop_trans_t.append([s for s in pop_trans_temp])    
                temp_pop = []
                for i in range(len(temp_nVec)):
                    temp_pop.append(Species(n = temp_nVec[i], f = fitness(selection, temp_sVec[i]), sequence = temp_sVec[i]))
                
                # receive population transfered from earlier simulations
                for sim_from in range(sim):
                    if t in range(len(full_pop_trans[sim_from])):
                        pop_trans_to = full_pop_trans[sim_from][t][sim-sim_from-1]
                        transfer_in(pop_trans_to, temp_pop)
                    fpop = [i for i in temp_pop] 
            
            if arg_list.in_flow:
                # add inflow population
                if t in range(len(pop_inflow[sim])):
                    pop_into = pop_inflow[sim][t]
                    transfer_in(pop_into, fpop)
          
            if fpop==[]:
                print("population died off")
                break
    
             # Update measurements
     
            if record == 1:
                fnVec.append(np.array([s.n        for s in fpop]))
                fsVec.append(np.array([s.sequence for s in fpop], dtype=object))            
            elif t % record == 0 and t != 0:       # avoids the problem of recording the initial population, then recording again at t=0
                fnVec.append(np.array([s.n        for s in fpop]))
                fsVec.append(np.array([s.sequence for s in fpop], dtype=object))
            
            if arg_list.transfer:
                pop_trans.append([i for i in pop_trans_t])
        
        if arg_list.transfer:
            full_pop_trans.append([i for i in pop_trans])
        
        pop, nVec, sVec, R_true = fpop, fnVec, fsVec, R_actual
        
        times = np.array(np.arange(len(nVec))) * record
        pop_size_record = np.array([np.sum(nVec[t]) for t in range(len(nVec))])
        mutant_sites = np.array([int(i) for i in selection])
        s_coefficients = np.array([selection[i] for i in mutant_sites])
        
        # calculate the covariance matrix and the single and double site allele frequencies, the single site are the sampled trajectories
        if covariance_off==False:
            if sample != 0 or arg_list.TVsample != None:
                sVec, nVec = SampleSequences(sVec, nVec, sample_size)
            full_nVec.append([i for i in nVec])
            full_sVec.append([i for i in sVec])
            full_pop.append(np.array(pop))
            full_true_R.append(np.array(R_true))
            
            # calculating allele frequencies and covariance matrix using the Stratonovich convention
            sVec_flatten = [sVec[i][j][k] for i in range(len(sVec)) for j in range(len(sVec[i])) for k in range(len(sVec[i][j]))]
            mutant_sites_samp = np.sort(np.unique(np.array(sVec_flatten)))
            selection_samp = np.array([selection[i] for i in mutant_sites_samp])
            single_freq, double_freq = allele_counter(sVec, nVec)   # the sampled single and double site frequencies
            delta_x = np.zeros((len(nVec)-1, len(selection_samp)))
            single_mid = np.zeros((len(nVec)-1, len(selection_samp)))
            double_mid = np.zeros((len(nVec)-1, len(selection_samp), len(selection_samp)))
            k_mid = np.zeros(len(nVec)-1)
            R_mid = np.zeros(len(nVec)-1)
            pop_mid = np.zeros(len(nVec)-1)
            for i in range(len(nVec)-1):
                delta_x[i] = single_freq[i+1] - single_freq[i]
                single_mid[i] = (single_freq[i] + single_freq[i+1]) / 2
                double_mid[i] = (double_freq[i] + double_freq[i+1]) / 2
                k_mid[i] = (k_full[i] + k_full[i+1]) / 2
                R_mid[i] = (R_true[i] + R_true[i+1]) / 2
                pop_mid[i] = (pop_size_record[i] + pop_size_record[i+1]) / 2
            covar_mid = covariance_calc(single_mid, double_mid)
            
            single_tot.append(list(single_mid))
            covar_tot.append(list(covar_mid))
            record_tot.append(record)
            delta_tot.append(list(delta_x))
            pop_size_record_tot.append(pop_size_record)
            mutant_sites_tot.append(mutant_sites_samp)
            times_tot.append(times)
                
    mutant_sites_all = np.sort(np.unique(np.array([mutant_sites_tot[i][j] for i in range(len(mutant_sites_tot))
                                                   for j in range(len(mutant_sites_tot[i]))])))
    num_mutated_sites = len([i for i in selection])
    selection_all = np.array([selection[i] for i in mutant_sites_all])
    #single_tot = trajectory_reshape(single_tot)
    
    # end and output total time
    if timed>0:
        end = timer()
        print('\nTotal time: %lfs, average per generation %lfs' % ((end - start),(end - start)/float(tEnd)))
        script_time = end - start
    else:
        script_time = 0
    
    # save arrays
    f = open(out_str+'.npz', mode='w+b')
    np.savez_compressed(f, script_time=script_time, pop_size=pop_size_record_tot, selection_all=selection_all, mutant_sites_all=mutant_sites_all, 
                        mutant_sites=mutant_sites_tot, traj_record=single_tot, R=full_true_R, covar=covar_tot, delta_x=delta_tot, 
                        record=record_tot, simulations=simulations, full_nVec=full_nVec, full_sVec=full_sVec, n_mutations=num_mutated_sites, times=times_tot)
    f.close()

if __name__ == '__main__': 
    main(sys.argv[1:])  
        
         
        
