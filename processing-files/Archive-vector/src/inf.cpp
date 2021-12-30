#include <stdio.h>
#include <stdlib.h>
#include <string>
//#include <gsl/gsl_vector.h>
//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_linalg.h>
#include <chrono>

#include "inf.h"    // inference declarations
#include "io.h"     // input/output


typedef std::chrono::high_resolution_clock Clock;


// GLOBAL VARIABLES

bool useDebug = false;
Vector TOTALCOV;
std::vector<double> DX;


// Compute single and pair allele frequencies from binary sequences and counts

void computeAlleleFrequencies(const IntVector &sequences, // vector of sequence vectors
                              const std::vector<double> &counts, // vector of sequence counts
                              int q, // number of states (e.g., number of nucleotides or amino acids)
                              std::vector<double> &p1, // single allele frequencies
                              Vector &p2 // pair allele frequencies
                              ) {
    
    // Set frequencies to zero
    
    for (int a=0;a<p1.size();a++) p1[a] = 0;
    for (int a=0;a<p2.size();a++) {
	   
	for (int b=0;b<p2[a].size();b++) p2[a][b] = 0;

    }

    int L = (int) p1.size();

    // Iterate through sequences and count the frequency of each state at each site,
    // and the frequency of each pair of states at each pair of sites
    
    for (int k=0;k<sequences.size();k++) {
        
        for (int i=0;i<sequences[k].size();i++) {

            int a = (i * q) + sequences[k][i];

            p1[a] += counts[k];

            for (int j=i+1;j<sequences[k].size();j++) {

                int b = (j * q) + sequences[k][j];

                p2[a][b] += counts[k];
                p2[b][a] += counts[k];

            }

        }
        
    }

}


// Interpolate single and pair allele frequencies between two points, weighted by (1-x) and x, respectively

void interpolateFrequencies(const std::vector<double> &p1_0, // single allele frequencies, 1st time point
                            const std::vector<double> &p2_0, // pair allele frequencies, 1st time point
                            const std::vector<double> &p1_1, // single allele frequencies, 2nd time point
                            const std::vector<double> &p2_1, // pair allele frequencies, 2nd time point
                            double x, // interpolation weight (0.5 = even split between 1st and 2nd times)
                            std::vector<double> &p1, // interpolated single allele frequencies
                            std::vector<double> &p2 // interpolated pair allele frequencies
                            ) {

    int L = (int) p1_0.size();

    // Iterate through states and perform simple linear interpolation of frequencies
    
    for (int a=0;a<L;a++) {
        
        p1[a] = ((1 - x) * p1_0[a]) + (x * p1_1[a]);
            
        for (int b=a+1;b<L;b++) {
        
            double xp2 = ((1 - x) * p2_0[(a * L) + b]) + (x * p2_1[(a * L) + b]);
            
            p2[(a * L) + b] = xp2;
            p2[(b * L) + a] = xp2;
            
        }
            
    }

}


// Update the summed covariance matrix

void updateCovarianceIntegrate(double dg, // time step
                               const std::vector<double> &p1_0, // single allele frequencies
                               const Vector &p2_0, // pair allele frequencies
                               const std::vector<double> &p1_1, // single allele frequencies
                               const Vector &p2_1, // pair allele frequencies
                               Vector &totalCov // integrated covariance matrix
                               ) {

    int L = (int) p1_0.size();
                          
    // Iterate through states and add contributions to covariance matrix

    for (int a=0;a<L;a++) {
        
        totalCov[a][a] += dg * ( ((3 - (2 * p1_1[a])) * (p1_0[a] + p1_1[a])) - (2 * p1_0[a] * p1_0[a]) ) / 6;
        
        for (int b=a+1;b<L;b++) {
        
            double dCov1 = -dg * ((2 * p1_0[a] * p1_0[b]) + (2 * p1_1[a] * p1_1[b]) + (p1_0[a] * p1_1[b]) + (p1_1[a] * p1_0[b])) / 6;
            double dCov2 = dg * 0.5 * (p2_0[a][b] + p2_1[a][b]);

            totalCov[a][b] += dCov1 + dCov2;
            totalCov[b][a] += dCov1 + dCov2;
            
        }
        
    }

}


// Update the summed mutation vector (flux out minus flux in)
// Note: since first row of mutation matrix is the reference, the mutation matrix is SHIFTED wrt frequencies,
// because the reference frequency is not explicitly tracked

void updateMuIntegrate(double dg, // time step
                       const Vector &muMatrix, // mutation matrix
                       const std::vector<double> &p1_0, // single allele frequencies
                       const std::vector<double> &p1_1, // single allele frequencies
                       std::vector<double> &totalMu // contribution to selection estimate from mutation
                       ) {

    int q = (int) muMatrix.size(); // number of tracked alleles (states)
    int L = (int) p1_0.size() / q;

    for (int i=0;i<L;i++) {

        for (int a=0;a<q;a++) {

            double fluxIn  = 0;
            double fluxOut = 0;

            for (int b=0;b<a;b++) {

                fluxIn  += 0.5 * (p1_0[(i * q) + b] + p1_1[(i * q) + b]) * muMatrix[b][a];
                fluxOut += 0.5 * (p1_0[(i * q) + a] + p1_1[(i * q) + a]) * muMatrix[a][b];

            }
            for (int b=a+1;b<q;b++) {

                fluxIn  += 0.5 * (p1_0[(i * q) + b] + p1_1[(i * q) + b]) * muMatrix[b][a];
                fluxOut += 0.5 * (p1_0[(i * q) + a] + p1_1[(i * q) + a]) * muMatrix[a][b];

            }

            totalMu[(i * q) + a] += dg * (fluxOut - fluxIn);

        }

    }

}


// Process asymptotic sequences (long time limit)

void processAsymptotic(const IntVVector &sequences, const Vector &counts, const Vector &muMatrix, int q, Vector &totalCov, std::vector<double> &dx) {

    int    L     = ((int) sequences[0][0].size()) * (q-1);  // sequence length (i.e. number of tracked alleles)
    double cNorm = (double) counts.size();                  // total number of time points, needed to normalize overall frequencies
    
    IntVector cSequences;               // sequence vector collapsed in time
    std::vector<double> cCounts;        // count vector collapsed in time
    std::vector<double> totalMu(L,0);   // accumulated mutation term
    std::vector<double> p1(L,0);        // total allele frequency vector
    //std::vector<double> p2(L*L,0);      // total allele pair frequencies
    Vector p2;
    p2.resize(L);
    for (int i=0;i<p2.size();i++) {p2[i].resize(L,0); }

    
    // collapse sequence and count vectors
    
    for (int k=0;k<sequences.size();k++) {
    
        for (int i=0;i<sequences[k].size();i++) {
        
            cSequences.push_back(sequences[k][i]);
            cCounts.push_back(counts[k][i]/cNorm);
            
        }
        
    }
    
    // compute allele frequencies and covariance
    
    computeAlleleFrequencies(cSequences, cCounts, q, p1, p2);
//    updateCovariance(1, p1, p2, totalCov);
//    updateMu(1, muMatrix, p1, totalMu);
    
    // gather dx and totalMu terms
    
    for (int a=0;a<L;a++) dx[a] += totalMu[a];

}


// Process standard sequences (time series)

void processStandard(const IntVVector &sequences, // vector of sequence vectors
                     const Vector &counts, // vector of sequence counts
                     const std::vector<double> &times, // sequence sampling times
                     const Vector &muMatrix, // matrix of mutation rates
                     int q, // number of states (e.g., number of nucleotides or amino acids)
                     Vector &totalCov, // integrated covariance matrix
                     std::vector<double> &dx // selection estimate numerator
                     ) {

    int L = ((int) sequences[0][0].size()) * q;     // sequence length (i.e. number of tracked alleles)
    std::vector<double> totalMu(L,0);               // accumulated mutation term
    std::vector<double> p1(L,0);                    // current allele frequency vector
    //std::vector<double> p2(L*L,0);                  // current allele pair frequencies
    Vector              p2;                         // current allele pair frequencies
    std::vector<double> lastp1(L,0);                // previous allele frequency vector
    //std::vector<double> lastp2(L*L,0);              // previous allele pair frequencies
    Vector              lastp2;                     // previous allele pair frequencies
    std::vector<double> xp1(L,0);                   // processed (midpoint, etc) allele frequency vector
    //std::vector<double> xp2(L*L,0);                 // processed (midpoint, etc) allele pair frequencies
    Vector              xp2;                        // processed (midpoint, etc) allele pair frequencies

    lastp2.resize(L);
    xp2.resize(L);
    p2.resize(L);
    for (int i=0;i<lastp2.size();i++) { 
	    
	lastp2[i].resize(L, 0);
	xp2[i].resize(L, 0);
	p2[i].resize(L,0);

    }
    
    // set initial allele frequency and covariance then loop
    
    computeAlleleFrequencies(sequences[0], counts[0], q, lastp1, lastp2);
    for (int a=0;a<L;a++) dx[a] -= lastp1[a];
    
    //--- debug
    if (useDebug) {
        int lwidth = 5;
        printf("dx = {\t");
        for (int a=0;a<L;a++) { if (a%lwidth==0 && a>0) printf("\n\t"); printf("%.4e\t",dx[a]); }
        printf("}\n\n");
    }
    //---
    
    // Iterate through sets of sequences collected at all time points,
    // computing allele frequencies, interpolating, and
    // adding contributions to selection coefficient estimates
    
    //printf("t1\tt2\tt2-t1\titer\tnSteps\tdt\tweight\tdg\n");
    printf("Total times is %d\n",(int) sequences.size());
    
    for (int k=1;k<sequences.size();k++) {
    
        computeAlleleFrequencies(sequences[k], counts[k], q, p1, p2);
        updateCovarianceIntegrate(times[k] - times[k-1], lastp1, lastp2, p1, p2, totalCov);
        //updateMuIntegrate(times[k] - times[k-1], muMatrix, lastp1, p1, totalMu);
        
        printf("Working on time point %d",k);	
        if (k==sequences.size()-1) { for (int a=0;a<L;a++) dx[a] += p1[a]; }
        else { 

            lastp1 = p1;
	    lastp2 = p2;

	    //for (int a=0;a<L;a++) {lastp2[a] = p2[a]; }

	}
    
    }
    
    //--- debug
    if (useDebug) {
        int lwidth = 5;
        printf("dx = {\t");
        for (int a=0;a<L;a++) { if (a%lwidth==0 && a>0) printf("\n\t"); printf("%.4e\t",dx[a]); }
        printf("}\n\n");
    }
    //---
    
    // Gather dx and totalMu terms
    
    for (int a=0;a<L;a++) dx[a] += totalMu[a];

}


// Add Gaussian regularization for selection coefficients (modifies integrated covariance)

void regularizeCovariance(const IntVVector &sequences, // vector of sequence vectors
                          int q, // number of states (e.g., number of nucleotides or amino acids)
                          double gammaN, // normalized regularization strength
                          bool useCovariance, // if false, don't consider off-diagonal terms
                          Vector &totalCov // integrated covariance matrix
                          ) {
    
    int L = ((int) sequences[0][0].size()) * q;

    for (int a=0;a<L;a++) totalCov[a][a] += gammaN; // standard regularization (no gauged state)

}


// MAIN PROGRAM

int run(RunParameters &r) {
    
    // READ IN SEQUENCES FROM DATA
    
    IntVVector sequences;       // set of integer sequences at each time point
    Vector counts;              // counts for each sequence at each time point
    std::vector<double> times;  // list of times of measurements
    
    if (FILE *datain = fopen(r.getSequenceInfile().c_str(),"r")) { getSequences(datain, sequences, counts, times); fclose(datain); }
    else { printf("Problem retrieving data from file %s! File may not exist or cannot be opened.\n",r.getSequenceInfile().c_str()); return EXIT_FAILURE; }
    
    if (r.useVerbose) {
        if (times.size()>1) printf("Got %d time points, %.2f difference between first two times.\n",(int)sequences.size(),times[1]-times[0]);
        printf("Total number of loci (sites) is %d, number of alleles (states) is %d.\n",(int)sequences[0][0].size(),(int)r.q);
        if (counts.size()>1) {
            double sum = 0;
            for (int i=0;i<counts[1].size();i++) sum += counts[1][i];
            printf("Sum of all counts at the second time point is %.2f (1.00 expected).\n\n",sum);
        }
        printf("Parameters: N = %.2e, mu = %.2e, gamma = %.2e.\n\n",r.N,r.mu,r.gamma);
        if (counts.size()>0) {
            printf("First sequence at the first time point:\n%.2f\t",counts[0][0]);
            for (int a=0;a<sequences[0][0].size();a++) printf("%d ",sequences[0][0][a]);
            printf("\n\n");
        }
    }
    
    Vector muMatrix;    // matrix of mutation rates
    
    if (r.useMatrix) {
    
        if (FILE *muin = fopen(r.getMuInfile().c_str(),"r")) { getMu(muin, muMatrix); fclose(muin); }
        else { printf("Problem retrieving data from file %s! File may not exist or cannot be opened.\n",r.getMuInfile().c_str()); return EXIT_FAILURE; }
        
        r.q = (int) muMatrix.size();
        
    }
    else {
        
        muMatrix.resize(r.q, std::vector<double>(r.q, r.mu));
        for (int i=0;i<r.q;i++) muMatrix[i][i] = 0;
        
    }
    
    // PROCESS SEQUENCES
    
    int    L      = ((int) sequences[0][0].size()) * r.q;    // sequence length (i.e. number of tracked alleles)
    double tol    = r.tol;                                   // tolerance for changes in covariance between time points
    double gammaN = r.gamma/r.N;                             // regularization strength divided by population size
    printf("Before resizing anything\n");
    DX.resize(L, 0);                                         // difference between start and end allele frequencies
    printf("DX resized successfully\n");
    TOTALCOV.resize(L);                                      // accumulated allele covariance matrix
    printf("TOTALCOV resized successfully\n");
    
    for (int i=0;i<TOTALCOV.size();i++) TOTALCOV[i].resize(L, 0);
    printf("TOTALCOV subvectors resized successfully\n");
    
    // _ START TIMER
    auto t_start = Clock::now();
    
    if (r.useAsymptotic) processAsymptotic(sequences, counts, muMatrix, r.q, TOTALCOV, DX);
    else                 processStandard(sequences, counts, times, muMatrix, r.q, TOTALCOV, DX);
    
    // If there is more than one input trajectory, loop through all of them and add contributions
    // NOTE: CURRENT CODE ASSUMES THAT ALL VALUES OF N ARE EQUAL
    
    if (r.infiles.size()>1) { for (int k=1;k<r.infiles.size();k++) {
    
        // Reset trajectory variables and reload them with new data
        
        sequences.clear();
        counts.clear();
        times.clear();
        
        if (FILE *datain = fopen(r.getSequenceInfile(k).c_str(),"r")) { getSequences(datain, sequences, counts, times); fclose(datain); }
        else { printf("Problem retrieving data from file %s! File may not exist or cannot be opened.\n",r.getSequenceInfile().c_str()); return EXIT_FAILURE; }
        
        // Add contributions to dx and totalCov
        
        if (r.useAsymptotic) processAsymptotic(sequences, counts, muMatrix, r.q, TOTALCOV, DX);
        else                 processStandard(sequences, counts, times, muMatrix, r.q, TOTALCOV, DX);
    
    } }
    
    // REGULARIZE
    
    regularizeCovariance(sequences, r.q, gammaN, r.useCovariance, TOTALCOV);
    
    // RECORD COVARIANCE (optional)
    
    if (r.saveCovariance) {
        if (FILE *dataout = fopen(r.getCovarianceOutfile().c_str(),"w")) { printCovariance(dataout, TOTALCOV, L); fclose(dataout); }
        else { printf("Problem writing data to file %s! File may not be created or cannot be opened.\n",r.getCovarianceOutfile().c_str()); return EXIT_FAILURE; }
    }
    
    // RECORD NUMERATOR (optional)
    
    if (r.saveNumerator) {
        if (FILE *dataout = fopen(r.getNumeratorOutfile().c_str(),"w")) { printNumerator(dataout, DX, L); fclose(dataout); }
        else { printf("Problem writing data to file %s! File may not be created or cannot be opened.\n",r.getCovarianceOutfile().c_str()); return EXIT_FAILURE; }
    }
    
//    // INFER THE SELECTION COEFFICIENTS -- solve Cov . sMAP = dx
//
//    std::vector<double> sMAP(L,0);
//
//    if (r.useCovariance) {
//
//        int status;
//
//        gsl_matrix_view _cov = gsl_matrix_view_array(totalCov, L, L);   // gsl covariance + Gaussian regularization
//        gsl_vector_view  _dx = gsl_vector_view_array(dx, L);            // gsl dx vector
//        gsl_vector    *_sMAP = gsl_vector_alloc(L);                     // maximum a posteriori selection coefficients for each allele
//        gsl_permutation  *_p = gsl_permutation_alloc(L);
//
//        gsl_linalg_LU_decomp(&_cov.matrix, _p, &status);
//        gsl_linalg_LU_solve(&_cov.matrix, _p, &_dx.vector, _sMAP);
//
//        for (int a=0;a<L;a++) sMAP[a] = gsl_vector_get(_sMAP, a);
//
//        gsl_permutation_free(_p);
//        gsl_vector_free(_sMAP);
//
//        delete[] dx;
//        delete[] totalCov;
//
//    }
//
//    else {
//
//        for (int a=0;a<L;a++) sMAP[a] = dx[a] / totalCov[(a * L) + a];
//
//    }
    
    auto t_end = Clock::now();
    // ^ END TIMER
    
    printf("%lld\n",std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count());
    
//    // WRITE TO FILE
//
//    if (FILE *dataout = fopen(r.getSelectionCoefficientOutfile().c_str(),"w")) { printSelectionCoefficients(dataout, sMAP); fclose(dataout); }
//    else { printf("Problem writing data to file %s! File may not be created or cannot be opened.\n",r.getSelectionCoefficientOutfile().c_str()); return EXIT_FAILURE; }
//
//    if (r.useVerbose) {
//        int lwidth = 5;
//        printf("s = {\t");
//        for (int a=0;a<L;a++) { if (a%lwidth==0 && a>0) printf("\n\t"); printf("%.4e\t",sMAP[a]); }
//        printf("}\n");
//    }
    
    return EXIT_SUCCESS;
 
}
