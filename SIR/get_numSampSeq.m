
function [numSampledSeqsPerStrainHigh] = get_numSampSeq(newInfected_Interp, samplingTime, samplingWindowSize, ...
    timeVec, fractionOfCasesSequnced, fractionOfCasesSequncedHigh, genotypes)

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
