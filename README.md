
# Overveiw

This repository contains and data and scripts for reproducing the results accompanying the manuscript  

### Inferring effects of mutations on SARS-CoV-2 transmission from genomic surveillance data
Brian Lee<sup>1</sup>, Muhammad Saqib Sohail<sup>2</sup>, Elizabeth Finney<sup>1</sup>, Syed Faraz Ahmed<sup>2</sup>, Ahmed Abdul Quadeer<sup>2</sup>, Matthew R. McKay<sup>2,3,4,5</sup> and John P. Barton<sup>1,#</sup>

<sup>1</sup> Department of Physics and Astronomy, University of California, Riverside  
<sup>2</sup> Department of Electronic and Computer Engineering, Hong Kong University of Science and Technology  
<sup>3</sup> Department of Chemical and Biological Engineering, Hong Kong University of Science and Technology  
<sup>4</sup> Department of Electrical and Electronic Engineering, University of Melbourne  
<sup>5</sup> Department of Microbiology and Immunology, University of Melbourne, at The Peter Doherty Institute for Infection and Immunity  
<sup>#</sup> correspondence to [john.barton@ucr.edu](mailto:john.barton@ucr.edu)  

# Contents

1. __Branching process simulations__: A notebook to generate and analyze simulation is given in `simulations.ipynb`, and the scripts for generating and analyzing simulations is given in the `simulation-scripts/` directory.
2. __SIR simulations__: The folder `SIR` contains MATLAB files for running and analyzing two different multi-variant SIR simulations. This folder contains its own readme file with instructions on how to use the files.
3. __Data processing__: A notebook for processing and analyzing SARS-CoV-2 sequence data is given in `data-paper.ipynb`. Scripts for analyzing and processing the data are given in the `data_processing.py` module and the `processing-files/` directory. Due to the number of SARS-CoV-2 genomes, much of the data analysis is best run on a computer cluster. We have provided code for producing the necessary job files in the `data-paper.ipynb` notebook. The original sequence data and metadata can be downloaded from [GISAID](https://gisaid.org).
4. __Figures__: A notebook for generating the figures found in the paper or the supplementary material is given in `figures.ipynb`. Modules for generating the figures in the paper are given in `figs.py` and `mplot.py`, while figures in the supplementary data can be produced using the `epi_figs.py` module.

### Software dependencies

Parts of the analysis are implemented in C++11 and the [GNU Scientific Library](https://www.gnu.org/software/gsl/).

# License

This repository is dual licensed as [GPL-3.0](LICENSE-GPL) (source code) and [CC0 1.0](LICENSE-CC0) (figures, documentation, and our presentation of the data).
