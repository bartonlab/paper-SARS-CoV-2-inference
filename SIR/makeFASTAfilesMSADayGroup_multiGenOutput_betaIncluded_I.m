

%% Sampler: Converts GT test data to sampled .DAT file
% based on DataGen_Samples_unEvenMATFile_or_Random_FASTA_Loop.m%%

%% this code :
%      1. loads data in .mat
%      2. calculate one and two point frequencies, sample, and store in
%      .dat format 


% THIS one also saves Beta_timevaryig (*N) so that scaling with beta isnt
% required manually

% modified: 3-May-2021
% Author: M Saqib Sohail


%%
function q = makeFASTAfilesMSADayGroup_multiGenOutput_betaIncluded_I(SetStrCell, numItr, numSampledSeqsPerStrainIn, genotypes, ngSamplingSchemeStr, samplingTime, perSiteSelection, dayGroup, TUsedForInfVec,exp_factor_Interp, betaTimeVarying_Interp_Mul_N)

exp_factor_Interp = exp_factor_Interp';
fprintf('Making FASTA files for further analysis...')

repLoopLen = length(TUsedForInfVec);

for kkk = 1:1:repLoopLen
    thisTUsed = TUsedForInfVec(kkk);
    samplingTime = 1:1:thisTUsed;
    numSampledSeqsPerStrain = numSampledSeqsPerStrainIn(1:thisTUsed,:);
    singleSumSampledSeqsPerStrainIn = sum(numSampledSeqsPerStrainIn,2)';
    singleSumSampledSeqsPerStrainIn_ThisTUsed = singleSumSampledSeqsPerStrainIn(1:thisTUsed);
    exp_factor_Interp_ThisTUsed = exp_factor_Interp(1:thisTUsed);
    betaTimeVarying_Interp_Mul_N_ThisTUsed = betaTimeVarying_Interp_Mul_N(1:thisTUsed);
    numStrains = size(genotypes, 1);% computed automatically
    L = size(genotypes, 2);% computed automatically
    %%
    for asets = 1:length(SetStrCell)
        thisSetStr = SetStrCell{asets};
        if(dayGroup > 0)
            temp1 = num2str(thisTUsed);
            temp1Len = length(temp1);

            temp2 = [repmat('0', 1, (4-temp1Len)) temp1];
            thisSetStr = [thisSetStr temp2 '_DayGroup' num2str(dayGroup)];
        end
        initStrains = numStrains(asets);%%1; % strains in starting population

        if(ispc)
            chosenSlash = '\';
        elseif(isunix)
            chosenSlash = '/';
        else
            disp('Error: system si not unix and not PC...')
            pause
        end
        mainDir = pwd;
        newDataDir = mainDir;
        dirNameFASTATemp = [newDataDir chosenSlash 'SIR_EvoData' chosenSlash 'Set' thisSetStr chosenSlash 'ng_' ngSamplingSchemeStr '_initStr' num2str(initStrains) chosenSlash];

        protName = 'synth';
        for thisItr = 1:numItr
            tic
            
            patID = ['MC' num2str(thisItr)];
            dirNameFASTA = [dirNameFASTATemp patID chosenSlash protName chosenSlash];

%             if(exist(dirNameFASTA , 'dir') == 0)
%                 mkdir(dirNameFASTA )
%             end

        %% 2. calculate one and two point frequencies, sample, and store in .dat format
        %--------------------------------------------------------------------------

            % Reconstruct MSA from raw data files, perform finite sampling by 
            % selecting ng individuals from the MSA and find sampled 1 and 2 point
            % frequencies
            %numGen = (Tend - Tstart + 1)/dT; % number of generation in the whole msa
            %numGen = (Tend - Tstart)/dT; % number of generation in the whole msa

            % later can make a switch to chose less than N samples too to simulate what
            % happens in FLU samples 

            %numSamplesPerGenSelected = ng;

            %samplingTimePoints = Tstart:dT:Tend;
            samplingTimePoints = samplingTime(sum(numSampledSeqsPerStrain,2)' ~= 0);%dTAll;
            numSamplingPoints = length(samplingTimePoints);


            if(dayGroup == 1)
                dayGroupVec = [1:numSamplingPoints]';
                q = -1*ones(numSamplingPoints, L);
            else
                dayGroupVec = zeros((dayGroup - ceil(dayGroup/2)), dayGroup);
                count40 = 1;
                for j = 1:(numSamplingPoints - (dayGroup - 1))%(numSamplingPoints - (dayGroup - floor(dayGroup/2)))
                    dayGroupVec(j,:) = count40:(count40+dayGroup-1);
                    count40 = count40 + 1;
                end
                q = -1*ones(numSamplingPoints  - (dayGroup - 1), L);
            end    



            for tp = 1:size(dayGroupVec, 1)%samplingTimePoints
                %thisSamplingTP = samplingTimePoints(tp);
                thisSamplingTP = dayGroupVec(tp,1);

                if(dayGroup == 1)
                    countAllStrainsThisTP = numSampledSeqsPerStrain(thisSamplingTP,:)';
                else
                    countAllStrainsThisTP = numSampledSeqsPerStrain(dayGroupVec(tp,:),:)';
                    countAllStrainsThisTP = sum(countAllStrainsThisTP, 2);
                end

                thisMSA = genotypes;

                numSamplesPerGenSelected = sum(countAllStrainsThisTP);
                q(tp,:) = countAllStrainsThisTP'*thisMSA/numSamplesPerGenSelected; % normalize frequencies;


                %[strainsThisTimePoint freqStrainThisTimePoint]
                count2 = 0;
                %strainID_and_Freq = zeros(1000,2);
                temp40 = repmat(' ', 1, 100);
                header = repmat({temp40}, 1, 1000);
                temp50 = repmat(' ', 1, L);
                seqNT_All = repmat({temp50}, 1, 1000);
                freqValTempVec = 0;
                for k = 1:size(thisMSA,1)
                    thisStrain = thisMSA(k,:);
                    freqThisStrinInPop_ng = countAllStrainsThisTP(k)/numSamplesPerGenSelected;
                    if(freqThisStrinInPop_ng ~= 0)
                        count2 = count2 + 1;
                        %strainID_and_Freq(count2,:) = [thisStrain freqThisStrinInPop_ng];
                        % count2+1 cuz 1st entry of header will be the ref seq
                        % freqThisStrinInPop_ng here is actually the number o
                        % haplotypes in population

                        freqValTemp = round(100000*freqThisStrinInPop_ng)/100000;
                        freqValTempVec(k) = freqValTemp;
                        header{count2+1} = ['>haplotype, days: ' num2str(thisSamplingTP) ', freq: ' num2str(freqValTemp) ', reads: ' num2str(countAllStrainsThisTP(k)) ','];
                        seqNT_All{count2+1} = int2nt(thisMSA(k,:) + 1);
                    end
                end


                header{1} = ['>reference_sequence, all zeros'];
                seqNT_All{1} = int2nt(zeros(1, L)+1);

                fileNameFASTA = [patID '_' protName '_bsample1of1_t' num2str(thisSamplingTP) '.fasta'];

                if(~(exist(dirNameFASTA, 'dir') == 7))
                    mkdir(dirNameFASTA)
                end

                if(exist([dirNameFASTA fileNameFASTA], 'file') == 2)
                    delete([dirNameFASTA fileNameFASTA])
                end

                fastawrite([dirNameFASTA fileNameFASTA], header(1:count2+1), seqNT_All(1:count2+1));




                clear qijAtTimeTk;
                clear masterStrainFreq;




                dirnamePerSiteSelction = [dirNameFASTA 'perSiteSelection' chosenSlash];

                if(exist(dirnamePerSiteSelction, 'dir') == 0)
                    mkdir(dirnamePerSiteSelction)
                end

                if(exist([dirNameFASTA 'perSiteSelction.txt'], 'file') == 2)
                    delete([dirNameFASTA 'perSiteSelction.txt'])
                end

                dlmwrite([dirnamePerSiteSelction 'perSiteSelction.txt'], perSiteSelection);
                
                
            end
        end
    end
    lim1 = (dayGroup - 1);
           
%     size(temp2020)
%     size(exp_factor_Interp_temp2020)
%     for dg = 1:dayGroup
%         %[(kkk) length(temp2020)-lim1+(kkk-1)]
%         %length(temp2020)
%         temp3030(dg, :) = temp2020((dg):end-lim1+(dg-1));
%         exp_factor_Interp_temp2020_dayGroup(dg,:) = exp_factor_Interp_temp2020((dg):end-lim1+(dg-1));
%     end
         
    
    populationN_ThisTused = exp_factor_Interp_ThisTUsed.*singleSumSampledSeqsPerStrainIn_ThisTUsed;
    
    populationN_ThisTused_dayGroup = zeros(1, (thisTUsed-dayGroup));
    for t_dg = 1:(thisTUsed-dayGroup)
%       populationN_ThisTused(t_dg:(t_dg+(dayGroup-1)))
%        pause
        populationN_ThisTused_dayGroup(t_dg) = sum(populationN_ThisTused(t_dg:(t_dg+(dayGroup-1))));
    end
        
    
    
    populationN = populationN_ThisTused_dayGroup;
    
   populationN
    %dlmwrite([dirnamePerSiteSelction 'Set' thisSetStr '_Tused.txt'], populationN);
    dlmwrite([dirnamePerSiteSelction 'populationN.txt'], populationN);
    dlmwrite([dirnamePerSiteSelction 'betaN.txt'], betaTimeVarying_Interp_Mul_N_ThisTUsed);
    
    dlmwrite([dirnamePerSiteSelction 'q_I.txt'], q);
    clear temp3030
end
disp('done.')
xDim = 20;
yDim = 15;
fig5 = figure('Units','centimeters', ...
                'Position', [20 2 xDim yDim]);
% row 1 :  2 + 1.81 + 0.2 + 1.81 + 0.2 + 1.81 + 1.5  + 4 + 1.5 + 4 + + 0.5
% row 2:   1 + 4.25 + 0.5 + 4.25 + 0.5 + 4.25 + 0.5 + 4.25 + 0.5
leftMargin = 1.5;
rightMargin = 0.5;
bottomMargin = 1;
topMargin = 0.5;

vgap = 0.6;
hgap = 1.5;
%width = 10 - leftMargin - rightMargin;
width = (xDim - leftMargin - rightMargin - 1*hgap)/2;
height1 = (yDim - bottomMargin - topMargin - 4*vgap)/5;%2.65;

ha(1) = axes('Units','centimeters', ...
                'Position',[leftMargin bottomMargin+4*vgap+4*height1 width height1], ...
                'XTickLabel','', ...
                'YTickLabel','');
ha(2) = axes('Units','centimeters', ...
                'Position',[leftMargin+hgap+width bottomMargin+4*vgap+4*height1 width height1], ...
                'XTickLabel','', ...
                'YTickLabel','');
ha(3) = axes('Units','centimeters', ...
                'Position',[leftMargin bottomMargin+3*vgap+3*height1 width height1], ...
                'XTickLabel','', ...
                'YTickLabel','');
ha(4) = axes('Units','centimeters', ...
                'Position',[leftMargin+hgap+width bottomMargin+3*vgap+3*height1 width height1], ...
                'XTickLabel','', ...
                'YTickLabel','');            
ha(5) = axes('Units','centimeters', ...
                'Position',[leftMargin bottomMargin+2*vgap+2*height1 width height1], ...
                'XTickLabel','', ...
                'YTickLabel','');
ha(6) = axes('Units','centimeters', ...
                'Position',[leftMargin+hgap+width bottomMargin+2*vgap+2*height1 width height1], ...
                'XTickLabel','', ...
                'YTickLabel','');             
ha(7) = axes('Units','centimeters', ...
                'Position',[leftMargin bottomMargin+1*vgap+1*height1 width height1], ...
                'XTickLabel','', ...
                'YTickLabel','');
ha(8) = axes('Units','centimeters', ...
                'Position',[leftMargin+hgap+width bottomMargin+1*vgap+1*height1 width height1], ...
                'XTickLabel','', ...
                'YTickLabel',''); 
ha(9) = axes('Units','centimeters', ...
                'Position',[leftMargin bottomMargin+0*vgap+0*height1 width height1], ...
                'XTickLabel','', ...
                'YTickLabel','');
ha(10) = axes('Units','centimeters', ...
                'Position',[leftMargin+hgap+width bottomMargin+0*vgap+0*height1 width height1], ...
                'XTickLabel','', ...
                'YTickLabel','');   
 
for i = 1:size(genotypes, 2)
    axes(ha(i))
    %plot(alleleFreq(:,i), 'color', color_scheme31(70,:), 'linewidth', 1)
    %hold on
    plot(q(:,i), 'LineStyle', '-', 'linewidth', 0.5)
    
    set(gca, ...
      'Box'         , 'on'     , ...
      'TickDir'     , 'in'     , ...
      'TickLength'  , [.01 .01] , ...
      'XMinorTick'  , 'off'      , ...
      'YMinorTick'  , 'off'      , ...
      'YGrid'       , 'off'      , ...
      'XGrid'       , 'off'      , ...
      'XColor'      , [.1 .1 .1], ...
      'YColor'      , [.1 .1 .1], ...
      'YTick', [0:0.1:1], ... 'YLim', [0.5 1.001], ...'XLim', [0.5 sum(indSelected_logical) + 0.5], ...'YTick'       , 0:0.1:1, ...'XTick'       , 1.5:0.5:2.5, ...
      'Fontsize', 8, ...
      'LineWidth', 0.5)
     axis([samplingTime(1) samplingTime(end) 0 0.35])
     ylabel('Frequency')
     title(['I, Day group, Site: ' num2str(i)])
     if(i <= 8)
        set(gca, ...
          'XTickLabel', ' ')
     else
         xlabel('Time (t)')
     end
     
end