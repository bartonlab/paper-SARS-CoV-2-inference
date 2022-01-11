
% prepares  dirNames_X_Y.txt files needed in PreProcessingStep1

% usage: data folder should have the following structure
%        Data dir:  /.../dir1/[Patient_ID]/[Protein_name]/[FileNames].fasta
%        Analysis Dir should be different from data directiry


clc
clear all
close all

%-------------------- USER CONTROLLED INITIALIZATION ----------------------
FLAG_SaveFile = 1;

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

thisSetStr = '105';%'205';%'105';%'51';%'5';%'12';%'1';%755;
thisProfileStr = thisProfileStrCellAll{s}; %'60100';%'1';
numStrains = 6;%6;%2;
samplingOption = 2; % 1: regular, 2: high, 3:regular with 0.5
dayGroup = 1;

if(dayGroup > 0)
    thisSetStr = [thisSetStr '_Profile' thisProfileStr '_DayGroup' num2str(dayGroup)];
end
if(samplingOption == 1)
    samplingStr = 'SampReg';
    fractionOfCasesSequnced = 0.01*10000;
elseif(samplingOption == 2)
    samplingStr = 'SampHigh';
    fractionOfCasesSequnced = 10000;%0.5*10000;
elseif(samplingOption == 3)
    samplingStr = 'SampReg';
    fractionOfCasesSequnced = 0.5*10000;
end

if(samplingOption == 1 || samplingOption == 3)
    % These variables need to be set for given dataset
    dataDirNameMain = [pwd '/SIR_EvoData/Set' thisSetStr '/ng_Multinomial' num2str(fractionOfCasesSequnced) '_by_10000_initStr' num2str(numStrains) '/'];                                                                                                        
    analysisDirNameMain = [pwd '/SIR_EvoAnalysis/Set' thisSetStr '/ng_Multinomial' num2str(fractionOfCasesSequnced) '_by_10000_initStr' num2str(numStrains) '/'];
    str1 = ['dirNamesSet' thisSetStr '_' samplingStr '_ng_Multinomial' num2str(fractionOfCasesSequnced) '_by_10000_initStr' num2str(numStrains) '_'];
elseif(samplingOption == 2)    
    % These variables need to be set for given dataset
    dataDirNameMain = [pwd '/SIR_EvoData/Set' thisSetStr '/ng_High_Multinomial' num2str(fractionOfCasesSequnced) '_by_10000_initStr' num2str(numStrains) '/'];                                                                                                        
    analysisDirNameMain = [pwd '/SIR_EvoAnalysis/Set' thisSetStr '/ng_High_Multinomial' num2str(fractionOfCasesSequnced) '_by_10000_initStr' num2str(numStrains) '/'];
    str1 = ['dirNamesSet' thisSetStr '_' samplingStr '_ng_Multinomial' num2str(fractionOfCasesSequnced) '_by_10000_initStr' num2str(numStrains) '_'];
end



%--------------------------------------------------------------------------

% ------------------------- AUTO INITIALIZATION ---------------------------
% NO USER INPUT REQUIRED
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
if(exist(dirNameStr1Files, 'dir') == 0)
    mkdir(dirNameStr1Files)
end

allProts{1} = 'synth';

numProts = length(allProts);

for prot = 1:numProts
    thisProt = allProts{prot};

    dirNamePatList = getFolderContent(dataDirNameMain, 'dir');
    
    numPat = length(dirNamePatList);
    if(numPat == 0)
        disp('NumPat = 0. Check initialization settings and run again.')
    end

    %%
    for pat = 1:numPat
        disp(['Patient: ' dirNamePatList{pat}])
        dirNameDataThisPat = [dataDirNameMain dirNamePatList{pat} chosenSlash thisProt chosenSlash];
        dirNameAnalysisThisPat = [analysisDirNameMain dirNamePatList{pat} chosenSlash thisProt chosenSlash];

        if(exist(dirNameAnalysisThisPat, 'dir') == 0)
            mkdir(dirNameAnalysisThisPat)
        end
        fileNameContainingDirPath = [str1 dirNamePatList{pat} '_' thisProt '.txt'];

        if(FLAG_SaveFile == 1)
            fprintf('Saving .txt file containing names of data and analysis folders...')

            if(exist([dirNameStr1Files fileNameContainingDirPath], 'file') == 2)
                delete([dirNameStr1Files fileNameContainingDirPath])
            end
            fileID = fopen([dirNameStr1Files fileNameContainingDirPath],'w');
            for f = 1:2
               if(f == 1)
                  fprintf(fileID,'%s\n', '% This file specifies directory paths to Data, Analysis');
                  fprintf(fileID,'%s\n', '% the syntax is /dir1/dir2/dir3/ ');
                  fprintf(fileID,'%s\n', '% for windows based system, the code will automatically reformat the path');
                  fprintf(fileID,'%s\n', '%');
                  fprintf(fileID,'%s\n', '% -------------------------------------------------------------------------');
               end
               if(f == 1)
                   fprintf(fileID,'%s\n',['dirNameData=' dirNameDataThisPat]);
               elseif(f == 2)
                   fprintf(fileID,'%s\n',['dirAnalysisName=' dirNameAnalysisThisPat]);
               end
            end
            fclose(fileID);
            disp('done.')
        elseif(FLAG_SaveFile == 0)
            disp('Warning: .txt data file not saved as FLAG_SaveFile flag not set.')
        else
            disp('Error: case undefined')
            pause
        end
    end
end



end