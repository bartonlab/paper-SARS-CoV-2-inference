% README.TXT
% These codes were run on MATLAB R2017b an require Statistics and Machine Learning Toolbox' v11.2 and Bioinformatics Toolbox v4.9
%
% 1. This code plots the SIR supplementary figures. Choose 'Set 105 or 205'
%    by specifying on Line 9
%    Plot_figus_A_E_Supplement_new.m
% 2. This code is the first preprocessing step. It prepares SIR time series
%    data for analysis. Choose 'Set 105 or 205' by specifying on Line 31
%    PreprocessingStep_0_SIR_SARSCoV2.m
%    *Note: to run this code, delete the analyses folder SIR_EvoAnalysis
% 3. This code is the second preprocessing step. It prepares SIR time series
%    data for analysis. Choose 'Set 105 or 205' by specifying on Line 25
%    PreprocessingStep_1_SIR_SARSCoV2.m
% 4. This code analyzes the SIR data and estimates selection coefficients.
%    Choose 'Set 105 or 205' by specifying on Line 24
%    AnalysisMPL_SIR_SARSCoV2.m
% 5. This code runs an SIR model under a scenario where the number of
%    newly-infected individuals continues to increase and then remains
%    fixed (this is referred to as Set 105)
%    Run_multiSIR_FD_new_timeVaryingBeta_SARSCOV2_const_infec.m
%    *Note: to run this code, delete the analyses folders
%    SIR_EvoData/Set105...
% 6. This code runs an SIR model under a scenario where we fix $r_a=1$ and
%    adapt the transmission rate over time such that the system follows the
%    typical SIR dynamics (this is referred to as Set 205)
%    Run_multiSIR_FD_new_timeVaryingBeta_SARSCOV2_varyGamma.m
%    *Note: to run this code, delete the analyses folders
%    SIR_EvoData/Set205...