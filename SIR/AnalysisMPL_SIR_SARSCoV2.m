clc
clear all
close all
warning off


%========================== INITIALIZATION ================================

% -------------------------- User specified -------------------------------
thisProfileStrCellAll{1} = '60080';
thisProfileStrCellAll{2} = '60010';
thisProfileStrCellAll{3} = '60020';
thisProfileStrCellAll{4} = '60050';
thisProfileStrCellAll{5} = '60100';
thisProfileStrCellAll{6} = '60200';
% thisProfileStrCellAll{1} = '60004';
% thisProfileStrCellAll{2} = '60005';
% thisProfileStrCellAll{3} = '60010';
% thisProfileStrCellAll{4} = '60015';
% thisProfileStrCellAll{5} = '60020';

for s = 1:5

thisSetStr = '105';%'205';%'105'; % '51';%'13';%'5';%'12';%'1';%755;
thisProfileStr = thisProfileStrCellAll{s}; %'60100';%'1';
numStrains = 6%6;%2;
samplingOption = 2; % 1: regular, 2: high, 3:regular with 0.5
dayGroup = 1;%5;


if(dayGroup > 0)
     thisSetStr = [thisSetStr '_Profile' thisProfileStr '_DayGroup' num2str(dayGroup)];
end

if(samplingOption == 1)
    samplingStr = 'SampReg';
    fractionOfCasesSequnced = 0.01*10000;
    textCell{1} = ['dirNamesSet' thisSetStr '_' samplingStr '_ng_Multinomial' num2str(fractionOfCasesSequnced) '_by_10000_initStr' num2str(numStrains) '_'];
elseif(samplingOption == 2)
    samplingStr = 'SampHigh';
    fractionOfCasesSequnced = 10000;%0.5*10000;
    textCell{1} = ['dirNamesSet' thisSetStr '_' samplingStr '_ng_Multinomial' num2str(fractionOfCasesSequnced) '_by_10000_initStr' num2str(numStrains) '_'];
elseif(samplingOption == 3)
    samplingStr = 'SampReg';
    fractionOfCasesSequnced = 0.5*10000;
    textCell{1} = ['dirNamesSet' thisSetStr '_' samplingStr '_ng_Multinomial' num2str(fractionOfCasesSequnced) '_by_10000_initStr' num2str(numStrains) '_'];
end

% chose convention 1: Ito, 2: Stratonovich, 3: Linear interpolation
% Stratonovich has increased robustness to sampling effects than Ito
% Linear interpolation has most increased robustness to sampling effects
setConvention = 1;

priorConstSC = 0; % this is the strength of the SC regularization term

thisGenomicSegStartInd = 1; % this is the starting location of the first NT of the protein in the whole genome
thisGenomicSegStopInd = 5; % this is the ending location of the last NT of the protein in the whole genome

% for GAG use these 
% thisGenomicSegStartInd = 790;
% thisGenomicSegStopInd = 2289;

% this file contains the NT-to-NT mutation probability. It must be located
% in the folder .../MPL Pipeline/Data_Misc/MutationProbabilities/
fileNameContainingMutProb = 'MutProb_SyntheticData_0.txt';
recombProb = 0;%1e-4; % recombination probability

FLAG_UserProvidedRefSeq = false; % SET: user provides reference sequence in ACGT- form
referenceSequence = repmat('A', 1, 10);
FLAG_binaryApprox = true; % SET: use binary approximation (only binary approximation wroks currently)
numNT = 5; % specify the number of NT that 'can' occur in the provided fasta files. Default value is 5 (ACGT-)

FLAG_MarkAccessibility = false; % KEEP this FALSE for the time being, Accessibility code needs to be checked

FLAG_SaveIntCovMtx = true;%false; % SET: will save Integrated Covariance matrix (for debugging only)
FLAG_useFreqEntry = true;
FLAG_troubleShoot = true; % SET: saves SelEstNoMu and SelEstSLNoMu
FLAG_Epi = false; % SET: use MPL with epistasis, UNSET: MPL with epistasis not used

%--------------------------------------------------------------------------


% ------------------------- AUTO INITIALIZATION ---------------------------
% NO USER INPUT REQUIRED
if(setConvention == 1)
    FLAG_stratonovich = false;%true;%true; % SET: stratonovich convention, UNSET: Ito convention
    FLAG_linearInt = false;
elseif(setConvention == 2)
    FLAG_stratonovich = true;%true;%true; % SET: stratonovich convention, UNSET: Ito convention
    FLAG_linearInt = false;
elseif(setConvention == 3)
    FLAG_stratonovich = false;%true;%true; % SET: stratonovich convention, UNSET: Ito convention
    FLAG_linearInt = true;
end

% this file will contain the names of .fasta files to analyze suing the
% AnalysisMPL_shortRead code. This file will be generated autotomatically
% in preprocessingStep1. Here we just need to specify the name. 
fileNameFastaFilesWithHU = 'fastaFilesHU.txt'; 
meFastaFileToAnalyze = 'fastaFilesToAnalyze.txt'; % right now, these fasta files need to be generated on laptop

FLAG_firstSeqIsRef = true; % set: 1st sequence of every fasta file is reference sequence
mainDir = pwd;
if(ispc)
    chosenSlash = '\';
elseif(isunix)
    chosenSlash = '/';
else
    disp('Error: system is not unix and not PC...')
    pause
end
dirNameTemp123 = 'dirNameFiles';
dirNameStr1Files = [mainDir chosenSlash 'Data_Misc' chosenSlash dirNameTemp123 chosenSlash];


fileNamesListThisDir = findFileNamesWithGivenText(dirNameStr1Files, textCell);
numPat = length(fileNamesListThisDir);
%--------------------------------------------------------------------------
if(numPat == 0)
    disp('NumPat = 0. Check initialization settings and run again.')
end


% ========================== BEGIN PROCESSING =============================

for pat = 1:numPat
    
    fileNameContainingDirPath = [dirNameStr1Files fileNamesListThisDir{pat}];
    indOfDash = strfind(fileNameContainingDirPath, '_');
    indOfDot = strfind(fileNameContainingDirPath, '.');
    patID = fileNameContainingDirPath(indOfDash(end-1)+1:indOfDash(end)-1);
    thisProt = fileNameContainingDirPath(indOfDash(end)+1:indOfDot(end)-1);

    disp('-----------------------------------------------------------------')
    disp(' ')
    disp(['Patient: ' patID])
    disp(['Protein: ' thisProt])
    FLAG_Skip = false;
    if(strcmp(patID, 'MC1') == false)
        FLAG_Skip = true;
    end
    if(FLAG_Skip == false)
        %analysisStep1_v2(fileNameContainingDirPath, priorConstSC, FLAG_stratonovich, FLAG_MarkAccessibility, FLAG_UserProvidedRefSeq, FLAG_SaveIntCovMtx, FLAG_useFreqEntry, FLAG_troubleShoot, FLAG_linearInt);
        priorConst = priorConstSC;
        FLAG_vector = [FLAG_stratonovich;
                       FLAG_MarkAccessibility;
                       FLAG_UserProvidedRefSeq;
                       FLAG_SaveIntCovMtx;
                       FLAG_useFreqEntry;
                       FLAG_troubleShoot;
                       FLAG_linearInt];
        % no need to specify  FLAG_Epi for MPL without epistasis analyss
        %analysisStep1_v2_SIR(fileNameContainingDirPath, priorConst, FLAG_vector, referenceSequence);
        %analysisStep1_v2_SIR_betaN(fileNameContainingDirPath, priorConst, FLAG_vector, referenceSequence);
        analysisStep1_v2_SIR_betaN_qI_new(fileNameContainingDirPath, priorConst, FLAG_vector, referenceSequence);
    else
        disp('...skipping this patient-protien combination...')
    end
end

end
