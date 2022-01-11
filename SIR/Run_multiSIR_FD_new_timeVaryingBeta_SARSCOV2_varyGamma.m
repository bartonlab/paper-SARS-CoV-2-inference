% call basic_sir functions


% this version makes the profile of number of new infected in population
% same as that in Run_multiSIR_FD_new_timeVaryingBeta_SARSCOV2.m while
% changing gamma

% citySetStr = 20X, the 20 indicates the data set is one where we make the
% profile of the number of infected ppl same as that of
% Run_multiSIR_FD_new_timeVaryingBeta_SARSCOV2.m while changing gamma
%%
clc
clear all
%close all

dirMain = pwd;
% =========================================================================
% initialization
saveFigs = 1;
SIRSet = 5;%51;
RtSet = 6;%2;
tmax = 185;
if(SIRSet == 1)
    citySetStr = '1';
    perSiteSelectionThis = 15*[0.03 0.02 0.03 -0.015 -0.015 0 0 0 0 0];
    genotypesThis = [0 0 0 0 0 0 0 0 0 0;
                     0 0 0 0 0 0 0 1 0 0;
                     1 0 1 0 0 0 0 0 0 0;
                     0 1 0 0 0 0 0 0 0 0;
                     0 0 0 1 0 0 0 0 0 0;
                     0 0 0 0 1 0 0 0 0 0];
    I_init = [400 400 10 400 400 400]'*1;
    maxSamplingTime = 80;
    daysToRecover = 25/3; % 8.3333; gamma = 1/daysToRecover
elseif(SIRSet == 2)
    citySetStr = '2';
    perSiteSelectionThis = 15*[0.03 0.02 0.03 -0.015 -0.045 0 0 0 0 0];
    genotypesThis = [0 0 0 0 0 0 0 0 0 0;
                     1 0 1 0 0 0 0 0 0 0;
                     0 1 0 1 0 0 0 0 0 1;
                     0 0 1 0 1 0 0 0 0 0;
                     0 0 0 1 0 0 0 0 0 0;
                     0 0 0 0 1 0 0 0 0 1];
    I_init = [400 10 400 400 400 400]'*10;
    maxSamplingTime = 120;
    daysToRecover = 25/3; % 8.3333; gamma = 1/daysToRecover
elseif(SIRSet == 3) % same as 2, but lower SC
    citySetStr = '3';
    perSiteSelectionThis = 3*[0.03 0.02 0.03 -0.015 -0.045 0 0 0 0 0];
    genotypesThis = [0 0 0 0 0 0 0 0 0 0;
                     1 0 1 0 0 0 0 0 0 0;
                     0 1 0 1 0 0 0 0 0 1;
                     0 0 1 0 1 0 0 0 0 0;
                     0 0 0 1 0 0 0 0 0 0;
                     0 0 0 0 1 0 0 0 0 1];
    I_init = [400 10 400 400 400 400]'*10;
    maxSamplingTime = 120;
    daysToRecover = 25/3; % 8.3333; gamma = 1/daysToRecover
elseif(SIRSet == 4)
    citySetStr = '4';
    perSiteSelectionThis = 15*[0.03 0.02 0.03 -0.015 -0.045 0 0 0 0 0];
    genotypesThis = [0 0 0 0 0 0 0 0 0 0;
                     1 0 1 0 0 0 0 0 1 0;
                     0 1 0 0 0 0 0 1 0 0;
                     1 0 1 0 0 0 0 0 0 0;
                     0 0 0 0 1 0 1 0 0 0;
                     0 0 1 1 0 0 0 0 0 1];
    I_init = [400 10 400 10 100 150]';
    daysToRecover = 25/3; % 8.3333; gamma = 1/daysToRecover
elseif(SIRSet == 5) % for varyGamma, diff value of gamma then base code
    citySetStr = '5';
    perSiteSelectionThis = 3*[0.03 0.01 -0.045 0 0];
    genotypesThis = [0 0 0 0 0;
                     1 0 0 0 0;
                     0 1 0 0 0;
                     0 0 0 1 0;
                     0 0 0 0 1;
                     0 0 1 0 0];
    I_init = [400 200 400 400 400 400]'*10;
    maxSamplingTime = 120;
    daysToRecover = 1; % 8.3333; gamma = 1/daysToRecover
elseif(SIRSet == 51) % only diff is that gamma = 1
    citySetStr = '51';
    perSiteSelectionThis = 3*[0.03 0.01 0.03 -0.015 -0.045 0 0 0 0 0];
    genotypesThis = [0 0 0 0 0 0 0 0 0 0;
                     1 0 0 0 0 0 0 0 0 0;
                     0 1 0 0 0 0 0 0 0 0;
                     0 0 0 0 0 0 0 0 1 0;
                     0 0 0 0 0 0 0 0 0 1;
                     0 0 0 0 1 0 0 0 0 0];
    I_init = [400 200 400 400 400 400]'*10;
    maxSamplingTime = 20;
    daysToRecover = 1; %  gamma = 1/daysToRecoverend
end
numNewInfcProfileFileName = ['Set' citySetStr '_Profile' num2str(RtSet) '.mat'];
load(numNewInfcProfileFileName, 'newInfected_Interp')
newInfected_Base = newInfected_Interp'; % remove 'Interp' as this term would be used within the sir function

citySetStr = ['20' citySetStr];
% genetic parameters
%--------------------------------------------------------------------------
perSiteSelection = perSiteSelectionThis;
genotypes = genotypesThis;
s_a = genotypes*perSiteSelection';


numStrains = size(genotypes, 1);% computed automatically
L = size(genotypes, 2);% computed automatically
% SIR parameters
%--------------------------------------------------------------------------
N = 10000000; % population size NY
mud = 0*ones(numStrains, 1);%1/80; % birth/death rate
mub = 0*ones(numStrains, 1);%1/80;
gamma = 1/daysToRecover*ones(numStrains, 1);%365/7; % rate of recovery

loadRtCurveThisState;
SetStrCell{1} = citySetStr;
R0 = thisStateMeanR0;%2;
R0_all = R0*(1+s_a);%3*ones(numStrains, 1);
%beta = R0.*(gamma + mub)/N;%1.3*(gamma + mub); % rate of disease transmission
beta0 = R0_all.*(gamma + mub)/N;%1.3*(gamma + mub); % rate of disease transmission

%I_init = 0.006*N/numStrains*ones(numStrains, 1); % equal number of I _Init
% for each strain

R_init = zeros(numStrains, 1);
S_init = (N - sum(I_init))*ones(numStrains, 1);
time = 184%200; % days
timeVec = 0:1:time;

% sampling parameters
%--------------------------------------------------------------------------
fractionOfCasesSequnced = 0.01;%0.001;
fractionOfCasesSequncedHigh = 1;%0.5;
samplingWindowSize = 1;%4; % 4: today + last 3 days
samplingTime = 1:1:maxSamplingTime;


% initialization to save MSA for inference of selection
%--------------------------------------------------------------------------

TusedOverWrite = samplingTime(end);

% SET: ng and dT need to be specified as vectors for each time point
ngSamplingSchemeStr = ['Multinomial' num2str(fractionOfCasesSequnced*10000) '_by_' num2str(10000)];
ngSamplingSchemeStrHigh = ['High_Multinomial' num2str(fractionOfCasesSequncedHigh*10000) '_by_' num2str(10000)];
numItr = 1;
% =========================================================================


% -------------------------------------------------------------------------
% 1 Simulate
% -------------------------------------------------------------------------
param(1,:) = [mud; gamma; mub];


beta_t = linspace(0, time, time+1);

% in varyGamma code, beta is computed adaptively inside the SIR function
% and comes as output
% betaTimeVarying = zeros(numStrains, time+1);
for i = 1:numStrains
    %betaTimeVarying(i,:) = beta0(i)*exp(-0.02*beta_t);
    betaTimeVarying(i,:) = beta0(i)*rtThisStateNorm;
end


betaForConstInfcFileName = [dirMain '\' 'Set' citySetStr '_betaValues.txt'];

if(exist(betaForConstInfcFileName, 'file') ~= 0)
    delete(betaForConstInfcFileName)
end
% 1.1 Run deterministic SIR model
% -------------------------------------------------------------------------
% for simplifying calculations, the /N part is included in beta, and not
% with I
%[t, Y] = ode45(@sir_multi_FD, [0 time], [S_init; I_init; R_init], [],param);

[t, Y] = ode45(@(t,y) sir_multi_timeVaryingBeta_FD_varyGamma(t,y, beta_t, betaTimeVarying, newInfected_Base, param, N, betaForConstInfcFileName), [0 time], [S_init; I_init; R_init]);

inputTemp = dlmread(betaForConstInfcFileName);
tempTime = inputTemp(:,1);
betaTimeVarying = inputTemp(:,[2:7])';

[a1, a2] = unique(round(100000*tempTime')/100000);
tempTime = tempTime(a2);
betaTimeVarying = betaTimeVarying(:, a2);

dim = numStrains;
S = Y(:,(0*dim+1):1*dim);
I = Y(:,(1*dim+1):2*dim);
R = Y(:,(2*dim+1):3*dim);

% 1.2 interpolate S, I, R
% -------------------------------------------------------------------------
t_InterpAllT = unique([t; timeVec']);
I_InterpAllT = zeros(length(t_InterpAllT), dim); % extra long vec, contains I values output of ode45 and also at tp 1 2 3 ....
R_InterpAllT = zeros(length(t_InterpAllT), dim);
%betaTimeVarying_InterpAllT = zeros(length(t_InterpAllT), dim);
for i = 1:dim
    I_InterpAllT(:,i) = interp1(t, I(:,i), t_InterpAllT);
    R_InterpAllT(:,i) = interp1(t, R(:,i), t_InterpAllT);
    %betaTimeVarying_InterpAllT(:,i) = interp1(timeVec, betaTimeVarying(i,:), t_InterpAllT);
end
rtThisState_InterpAllT = interp1(timeVec, rtThisState, t_InterpAllT);

SAll_InterpAllT = interp1(t, S(:,1), t_InterpAllT); % because all S_i's are the same
IAll_InterpAllT = sum(I_InterpAllT, 2);
RAll_InterpAllT = sum(R_InterpAllT, 2);

t_InterpAllT2 = unique([tempTime; t_InterpAllT]);
betaTimeVarying_InterpAllT = zeros(length(t_InterpAllT2), dim);
for i = 1:dim
    betaTimeVarying_InterpAllT(:,i) = interp1(tempTime', betaTimeVarying(i,:), t_InterpAllT2)/N;
end

% 1.3 find the differential of I
% -------------------------------------------------------------------------
%betaRow = beta';

indTemp1 = (abs(floor(t_InterpAllT) - t_InterpAllT) == 0);
t_Interp = t_InterpAllT(indTemp1);
I_Interp = I_InterpAllT(indTemp1,:); % contains I values ONLY at tp 1 2 3 ....
R_Interp = R_InterpAllT(indTemp1,:);
SAll_Interp = SAll_InterpAllT(indTemp1);
IAll_Interp = IAll_InterpAllT(indTemp1);
RAll_Interp = RAll_InterpAllT(indTemp1);


indTemp2 = (abs(floor(t_InterpAllT2) - t_InterpAllT2) == 0);
betaTimeVarying_Interp = betaTimeVarying_InterpAllT(indTemp2,:);

gammaRow = gamma';
%newInfected_InterpAllT = repmat(betaRow, length(t_InterpAllT), 1).*I_InterpAllT.*repmat(SAll_InterpAllT, 1, numStrains);
newInfected_Interp = betaTimeVarying_Interp.*I_Interp.*repmat(SAll_Interp, 1, numStrains);
Idiff_Interp = newInfected_Interp - repmat(gammaRow, length(t_Interp), 1).*I_Interp;



rtThisState_Interp = rtThisState_InterpAllT(indTemp1);
% 1.4 find indices of time vector where seqeuncing is done
% -------------------------------------------------------------------------
numSampledSeqsPerStrain = -1*ones(length(samplingTime), numStrains);
numSampledSeqsPerStrainHigh = -1*ones(length(samplingTime), numStrains);

% % Sequences sampled from I(t)
% for i = 1:length(samplingTime)
%     indTemp = t_InterpAllT == samplingTime(i);
%     temp2 = I_InterpAllT(indTemp,:);
%     numSampledSeqsPerStrain(i,:) = mnrnd(round(sum(temp2)*fractionOfCasesSequnced),(temp2/sum(temp2)));
%     numSampledSeqsPerStrainHigh(i,:) = mnrnd(round(sum(temp2)*fractionOfCasesSequncedHigh),(temp2/sum(temp2)));
% end

% % Sequences sampled from only those individuals who become infectious in
% the last x days
generationsUsedForSequencing = zeros((length(samplingTime)-1), 2);
for i = 1:(length(samplingTime)) 
    windowLowLim = i - samplingWindowSize + 1;
    if(windowLowLim < 0)
        windowLowLim = 0;
    end
    windowUpLim = i;
    generationsUsedForSequencing(i,:) = [windowLowLim windowUpLim];
    
    indTemp = (timeVec >= windowLowLim & timeVec <= windowUpLim);
    %infectPopToSeqFrom = sum(newInfected_Interp(indTemp,:));
    if(samplingWindowSize > 1)
        infectPopToSeqFrom = sum(newInfected_Interp(indTemp,:));
    else
        infectPopToSeqFrom = (newInfected_Interp(indTemp,:));
    end
%     if(i == 20)
%         asdf
%     end
   if(fractionOfCasesSequnced ~= 1)
        numSampledSeqsPerStrain(i,:) = mnrnd(round(sum(infectPopToSeqFrom)*fractionOfCasesSequnced),(infectPopToSeqFrom/sum(infectPopToSeqFrom)));
    else
        numSampledSeqsPerStrain(i,:) = round(infectPopToSeqFrom);
    end
    if(fractionOfCasesSequncedHigh ~= 1)
        numSampledSeqsPerStrainHigh(i,:) = mnrnd(round(sum(infectPopToSeqFrom)*fractionOfCasesSequncedHigh),(infectPopToSeqFrom/sum(infectPopToSeqFrom)));
    else
        numSampledSeqsPerStrainHigh(i,:) = round(infectPopToSeqFrom);
    end
end

% 1.5 Sample sequences from population and make MSA for selection analysis
% -------------------------------------------------------------------------
alleleFreq = zeros(length(samplingTime), size(genotypes, 2));
alleleFreqHigh = zeros(length(samplingTime), size(genotypes, 2));
% make msa per time point and store the allele freqs
for i = 1:length(samplingTime)
    numSeqsThisTP = numSampledSeqsPerStrain(i,:);
    thisMSA = zeros(sum(numSeqsThisTP), size(genotypes, 2));
    count = 0;
    for j = 1:size(genotypes, 1) % do for each genotype
        if(numSeqsThisTP(j) > 0)
            thisMSA((count+1:(count+numSeqsThisTP(j))), :) = repmat(genotypes(j,:), numSeqsThisTP(j), 1);
            count = count + numSeqsThisTP(j);
        end
    end
    
    alleleFreq(i,:) = sum(thisMSA/sum(numSeqsThisTP)); 
    % for High sampling freq
    numSeqsThisTPHigh = numSampledSeqsPerStrainHigh(i,:);
    thisMSAHigh = zeros(sum(numSeqsThisTPHigh), size(genotypes, 2));
    countHigh = 0;
    for j = 1:size(genotypes, 1) % do for each genotype
        if(numSeqsThisTPHigh(j) > 0)
            thisMSAHigh((countHigh+1:(countHigh+numSeqsThisTPHigh(j))), :) = repmat(genotypes(j,:), numSeqsThisTPHigh(j), 1);
            countHigh = countHigh + numSeqsThisTPHigh(j);
        end
    end
    
    alleleFreqHigh(i,:) = sum(thisMSAHigh/sum(numSeqsThisTPHigh));     
end

genFreq = numSampledSeqsPerStrain./sum(numSampledSeqsPerStrain,2);
genFreqHigh = numSampledSeqsPerStrainHigh./sum(numSampledSeqsPerStrainHigh,2);

samplesPerDay = sum(numSampledSeqsPerStrain,2);
samplesPerDayHigh = sum(numSampledSeqsPerStrainHigh,2);
fileNameToSave = ['Set' SetStrCell{1} '.mat'];
if(exist(fileNameToSave, 'file') == 2)
    delete(fileNameToSave)
end

% save sum_newInfected_Interp for sim where gama needs to be changed for
% same sum_newInfected_Interp profile
sum_newInfected_Interp = sum(newInfected_Interp');


[numSampledSeqsPerStrainHigh_I] = get_numSampSeq(I_Interp, samplingTime, samplingWindowSize, timeVec, fractionOfCasesSequnced, fractionOfCasesSequncedHigh, genotypes);




% call plotting codes
Plot_multiStrain_timeVaryingBeta_SARSCOV2;
TUsedForInfVec = [10 20 50 80 100]% [4 5 10 15 20]% 50 80 100];

%%
% make FASTA files for further analysis
%makeFASTAfilesMSA(SetStrCell, numItr, numSampledSeqsPerStrain , genotypes, ngSamplingSchemeStr, samplingTime, perSiteSelection)
%makeFASTAfilesMSA(SetStrCell, numItr, numSampledSeqsPerStrainHigh, genotypes, ngSamplingSchemeStrHigh, samplingTime, perSiteSelection)
%%
dayGroup = 1;

betaTimeVarying_Interp_samp = betaTimeVarying_Interp(samplingTime,:);
%%%%% BETA effective based on freq of all numStrain genotypes %%%%%%
%betaTimeVarying_Interp_samp_EFFECTIVE  = sum((betaTimeVarying_Interp_samp.*genFreqHigh), 2);

%%%% Beta effective = beta_WT
betaTimeVarying_Interp_samp_EFFECTIVE  = betaTimeVarying_Interp_samp(:,1);
%rtThisState_Interp = (betaTimeVarying_Interp*N./repmat(gamma', size(betaTimeVarying_Interp,1),1));
rtThisState_Interp = betaTimeVarying_Interp_samp_EFFECTIVE*N./gamma(1);
exp_factor = (SAll_Interp(samplingTime)).*rtThisState_Interp(samplingTime,1)/N; 
betaTimeVarying_Interp_Mul_N = betaTimeVarying_Interp_samp_EFFECTIVE*N;
%q = makeFASTAfilesMSADayGroup_multiGenOutput(SetStrCell, numItr, numSampledSeqsPerStrainHigh, genotypes, ngSamplingSchemeStrHigh, samplingTime, perSiteSelection, dayGroup, TUsedForInfVec, exp_factor);
q = makeFASTAfilesMSADayGroup_multiGenOutput_betaIncluded_I(SetStrCell, numItr, numSampledSeqsPerStrainHigh, genotypes, ngSamplingSchemeStrHigh, samplingTime, perSiteSelection, dayGroup, TUsedForInfVec, exp_factor, betaTimeVarying_Interp_Mul_N);
save(fileNameToSave)
%%
popSizePerDayGroup = zeros(1, (TusedOverWrite - dayGroup));
for i = 1:(TusedOverWrite - dayGroup)
    popSizePerDayGroup(i) = sum(samplesPerDay(i:(i+dayGroup-1)));  
end
