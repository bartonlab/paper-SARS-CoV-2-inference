%% same as Plot_figus_A_E_Supplement.m . Display is different based on discussion with John
% NOTE: estimated SC are hard coded as those are obtained 
clc
clear all
close all

figABCDEFG_FLAG = [1 1 1 0 1 1 1]; % 'Set105_Profile6';
%figABCDEFG_FLAG = [1 1 1 0 0 1 1]; % 'Set205_Profile6';
thisSet = 205;%105%205;
samplingOption = 2; % 1: regular, 2: high, 3:regular with 0.5
dayGroup = 1;%5;


SetProfileStr = ['Set' num2str(thisSet) '_Profile6'];

if(thisSet == 5)
    load([SetProfileStr '.mat'], 'perSiteSelection', 'I_init', 'genotypes', 'numStrains', 't_Interp', 'I_Interp', 'dim', 'numSampledSeqsPerStrain', 'betaTimeVarying_Interp', 'N', 'gamma', 'q', 'samplingTime', 'newInfected_Interp', 'q_z')
else
    load([SetProfileStr '.mat'], 'perSiteSelection', 'I_init', 'genotypes', 'numStrains', 't_Interp', 'I_Interp', 'dim', 'numSampledSeqsPerStrain', 'betaTimeVarying_Interp', 'N', 'gamma', 'q', 'samplingTime', 'newInfected_Interp', 'numSampledSeqsPerStrainHigh')
end

siteFilter = sum(genotypes) ~= 0;
perSiteSelection = perSiteSelection(siteFilter);
genotypes = genotypes(:,siteFilter);
q = q(:,siteFilter);

% popAtEachTime = sum(numSampledSeqsPerStrainHigh,2);
% upTotime = 100;
% sum(q(1:upTotime,:).*(1 - q(1:upTotime,:))./popAtEachTime(1:upTotime))

if(dayGroup > 0)
     thisSetStr = [num2str(thisSet) '_Profile' '60100' '_DayGroup' num2str(dayGroup)];
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

dirNameTemp123 = 'dirNameFiles';
if(ispc)
    chosenSlash = '\';
elseif(isunix)
    chosenSlash = '/';
else
    disp('Error: system is not unix and not PC...')
    pause
end
dirNameStr1Files = [pwd chosenSlash 'Data_Misc' chosenSlash dirNameTemp123 chosenSlash];
fileNamesListThisDir = findFileNamesWithGivenText(dirNameStr1Files, textCell);

fileNameContainingDirPath = [dirNameStr1Files fileNamesListThisDir{1}];
[dirNameData, dirNameAnalysis] = loadDirNames(fileNameContainingDirPath);

filenameEstimates = [dirNameAnalysis chosenSlash 'Estimates' chosenSlash 'SelEst_Z_MC1_synth_Ito_gamma0.txt'];

% -------------------------------------------------------------------------
% 2. plot figures A to E
% -------------------------------------------------------------------------
set(0,'DefaultAxesFontName','Arial')
set(0,'DefaultTextFontName','Arial')
set(0,'DefaultAxesFontSize',6)
set(0,'DefaultTextFontSize',6)
fontSize = 7;
mapBlues = brewermap(8,'Blues');            
color_scheme1 = brewermap(100,'Reds');
color_scheme2 = brewermap(100,'Greys');
color_scheme3 = brewermap(100,'Blues');
white_scheme = repmat([ 1 1 1],100,1);
alpha1 = 0.8;
alpha2 = 0.7;
color_scheme11 = (255*color_scheme1*(alpha1) + 255*white_scheme*(1-alpha1))/255;
color_scheme21 = (255*color_scheme2*(alpha2) + 255*white_scheme*(1-alpha2))/255;
color_scheme31 = (255*color_scheme3*(alpha2) + 255*white_scheme*(1-alpha2))/255;


colorCell{1} = color_scheme11(90,:);
colorCell{2} = color_scheme11(80,:);
colorCell{3} = color_scheme11(70,:);
colorCell{4} = color_scheme11(60,:);
colorCell{5} = color_scheme31(90,:);%color_scheme11(50,:);
colorCell{6} = color_scheme31(70,:);%color_scheme11(40,:);


% plot selection strengths
xDim = 18.5;
yDim = 18; %(7.5) + vgap1 + (F n G)
fig2 = figure('Units','centimeters', ...
                'Position', [2 2 xDim yDim]);
% row 1 :  2 + 1.81 + 0.2 + 1.81 + 0.2 + 1.81 + 1.5  + 4 + 1.5 + 4 + + 0.5
% row 2:   1 + 4.25 + 0.5 + 4.25 + 0.5 + 4.25 + 0.5 + 4.25 + 0.5
leftMargin = 1.5;
rightMargin = 0.5;
bottomMargin = 1;
topMargin = 0.5;

vgap1 = 1.5;
hgap1 = 2.2;
hgap2 = 2.2;
%width = 10 - leftMargin - rightMargin;
width = (xDim - leftMargin - rightMargin - hgap2);
width1 = width*0.33;
width2 = width*0.67/2;
width3 = (xDim - leftMargin - rightMargin - hgap2)/2;
%height1 = (yDim - bottomMargin - topMargin - vgap1)/3;
height_Main = (yDim - bottomMargin - topMargin - vgap1*2);
height_A_E = height_Main*0.45;
height_F = height_Main*0.55;
height1 = height_A_E/2;
vgap_F1 = 0.6;
hgap_F1 = hgap2;

height_F1 = (height_F - 3*vgap_F1)/3;
width_F1 = width/2;

if(figABCDEFG_FLAG(1) == 1)  
    ha(102) = axes('Units','centimeters', ...
                    'Position',[leftMargin bottomMargin+vgap1+height1*1+vgap1+height_F width1 height1*1], ...
                    'XTickLabel','', ...
                    'YTickLabel','');
end

if(figABCDEFG_FLAG(2) == 1)  
    ha(103) = axes('Units','centimeters', ...
                    'Position',[leftMargin bottomMargin+vgap1+height_F width1 height1*1], ...
                    'XTickLabel','', ...
                    'YTickLabel','');
end
if(figABCDEFG_FLAG(3) >= 1)  
    ha(104) = axes('Units','centimeters', ...
                    'Position',[leftMargin+hgap_F1+width_F1 bottomMargin+vgap1+height_F width_F1 height1*1], ...
                    'XTickLabel','', ...
                    'YTickLabel','');
end

if(figABCDEFG_FLAG(5) == 1)
    ha(106) = axes('Units','centimeters', ...
                    'Position',[leftMargin+hgap_F1+width_F1 bottomMargin+vgap1+height1*1+vgap1+height_F width_F1 height1], ...
                    'XTickLabel','', ...
                    'YTickLabel','');
end

if(figABCDEFG_FLAG(6) == 1) 
%     ha(1) = axes('Units','centimeters', ...
%                     'Position',[leftMargin bottomMargin+4*vgap_F1+4*height_F1 width_F1 height_F1], ...
%                     'XTickLabel','', ...
%                     'YTickLabel','');
%     ha(2) = axes('Units','centimeters', ...
%                     'Position',[leftMargin+hgap_F1+width_F1 bottomMargin+4*vgap_F1+4*height_F1 width_F1 height_F1], ...
%                     'XTickLabel','', ...
%                     'YTickLabel','');
%     ha(3) = axes('Units','centimeters', ...
%                     'Position',[leftMargin bottomMargin+3*vgap_F1+3*height_F1 width_F1 height_F1], ...
%                     'XTickLabel','', ...
%                     'YTickLabel','');
%     ha(4) = axes('Units','centimeters', ...
%                     'Position',[leftMargin+hgap_F1+width_F1 bottomMargin+3*vgap_F1+3*height_F1 width_F1 height_F1], ...
%                     'XTickLabel','', ...
%                     'YTickLabel','');            
    ha(1) = axes('Units','centimeters', ...
                    'Position',[leftMargin bottomMargin+2*vgap_F1+2*height_F1 width_F1 height_F1], ...
                    'XTickLabel','', ...
                    'YTickLabel','');
    ha(2) = axes('Units','centimeters', ...
                    'Position',[leftMargin+hgap_F1+width_F1 bottomMargin+2*vgap_F1+2*height_F1 width_F1 height_F1], ...
                    'XTickLabel','', ...
                    'YTickLabel','');             
    ha(3) = axes('Units','centimeters', ...
                    'Position',[leftMargin bottomMargin+1*vgap_F1+1*height_F1 width_F1 height_F1], ...
                    'XTickLabel','', ...
                    'YTickLabel','');
    ha(4) = axes('Units','centimeters', ...
                    'Position',[leftMargin+hgap_F1+width_F1 bottomMargin+1*vgap_F1+1*height_F1 width_F1 height_F1], ...
                    'XTickLabel','', ...
                    'YTickLabel',''); 
    ha(5) = axes('Units','centimeters', ...
                    'Position',[leftMargin bottomMargin+0*vgap_F1+0*height_F1 width_F1 height_F1], ...
                    'XTickLabel','', ...
                    'YTickLabel','');
%     ha(10) = axes('Units','centimeters', ...
%                     'Position',[leftMargin+hgap_F1+width_F1 bottomMargin+0*vgap_F1+0*height_F1 width_F1 height_F1], ...
%                     'XTickLabel','', ...
%                     'YTickLabel','');   
end

if(figABCDEFG_FLAG(1) == 1)          
    axes(ha(102))
    if(thisSet == 105)
        perSiteSelectionEstimate = dlmread(filenameEstimates);% [0.081 0.027 -0.127 0 0];
    elseif(thisSet == 205)
        perSiteSelectionEstimate = dlmread(filenameEstimates);% [0.07 0.027 -0.158 0 0];
    end
    
    %---------------------------------------------------------------------
    % BARPLOT of SC
    %bar([perSiteSelection; perSiteSelectionEstimate]')
    
    % Line and circle plot of SC
    for i = 1:5
        plot((0.65:0.1:1.43) + (i-1), perSiteSelection(i)*ones(1, 8), 'k', 'LineWidth', 1)
        hold on
    end
    plot(1:5, perSiteSelectionEstimate, 'LineStyle', 'none', 'Marker', 'o', 'MarkerEdgeColor', color_scheme11(80,:), 'LineWidth', 1)
    %---------------------------------------------------------------------
    maxSelVal = max(abs(perSiteSelection));
    axis([0.5 10.5 -1.1*maxSelVal 1.1*maxSelVal])
      set(gca, ...
          'Box'         , 'on'     , ...
          'TickDir'     , 'in'     , ...
          'TickLength'  , [.01 .01] , ...
          'XMinorTick'  , 'off'      , ...
          'YMinorTick'  , 'off'      , ...
          'YGrid'       , 'off'      , ...
          'XGrid'       , 'off'      , ...
          'XColor'      , [.1 .1 .1], ...
          'YColor'      , [.1 .1 .1], ...'YTick', [0:0.2:1], ... 'YLim', [0.5 1.001], ...'XLim', [0.5 sum(indSelected_logical) + 0.5], ...'YTick'       , 0:0.1:1, ...
          'YTick'       , [-0.15 -0.1  -0.05 0 0.05 0.1 0.15], ...'YTick'       , [-maxSelVal -0.5*maxSelVal 0 0.5*maxSelVal maxSelVal], ...'YTick'       , [-0.1  -0.05 0 0.05 0.1], ...
          'Fontsize', fontSize, ...
          'LineWidth', 0.5)
    xlabel('Locus')
    ylabel({'Selection coefficient'} )
    axis([0.5 5.5 -0.175  0.175])
    %grid on
    %leg = legend('True', 'Estimated', 'Location','EastOutside')
    %set(leg,'color','none');
    %set(leg, 'Edgecolor','none');
end

plot(0:0.1:6, zeros(1,61), 'k:', 'linewidth', 0.5) 

dimDummy = [.1 .1 .1 .1];
markera = annotation('line',[.1 .1], [.2 .2]);
markera.Units = 'centimeter';
markera.Position = [leftMargin+width1+0.5 yDim-topMargin-height1/2+0.5 0.5 0];%[0.55 4.1 0.7 0.5];
markera.LineStyle = '-';
markera.LineWidth = 1;

markerb = annotation('ellipse',dimDummy);
markerb.Units = 'centimeter';
markerb.Position = [leftMargin+width1+0.65 yDim-topMargin-height1/2 0.21 0.21];%[0.55 4.1 0.7 0.5];
markerb.EdgeColor = color_scheme11(80,:);
markerb.LineWidth = 1;
%markera.LineStyle = '-';

texta = annotation('textbox',dimDummy,'String','True','FitBoxToText','on')
texta.Units = 'centimeter';
texta.Position = [leftMargin+width1+1 yDim-topMargin-height1/2+0.75 0.5 0]
texta.LineStyle = 'none';
texta.FontSize = fontSize;

textb = annotation('textbox',dimDummy,'String','Estimated','FitBoxToText','on')
textb.Units = 'centimeter';
textb.Position = [leftMargin+width1+1 yDim-topMargin-height1/2+0.35 0.5 0]
textb.LineStyle = 'none';
textb.FontSize = fontSize;


if(figABCDEFG_FLAG(2) == 1)          
    axes(ha(103))
    heatmapMtx = genotypes;
    for i = 1:5
        xValCell{i} = i;
    end
    for j = 1:6
        yvalCell{j} = ['Variant ' num2str(j)];
    end
    heatmap(xValCell, yvalCell, heatmapMtx,'ColorbarVisible','off')
    set(gca, ...
          'Fontsize', fontSize)
    xlabel('Locus')

    for i =1:numStrains
        dimDummy = [0.1 0.1 0.1 0.1]; % dummy position (in normalized units)
        textNA = annotation('textbox',dimDummy,'String',['I_{0,var ' num2str(num2str(i)) '}: ' num2str(I_init(i))],'FitBoxToText','on');
        textNA.Units = 'centimeter';
        textNA.Position = [leftMargin+width1+0.2 height1*1+bottomMargin+0.05-(0.5*i + 0.01*(i-1))-0.05+vgap1+height_F 4 0.5];
        textNA.LineStyle = 'none';
        textNA.FontSize = fontSize;
    end
end

if(figABCDEFG_FLAG(3) == 1)          
    axes(ha(104))
    %plot(t_Interp, sum(I_Interp,2), 'color', color_scheme11(99,:), 'LineWidth',1)
    plot(t_Interp, sum(newInfected_Interp,2), 'color', color_scheme11(99,:), 'LineWidth',1)
    hold on
    lineStyleCell{1} = '-';
    lineStyleCell{2} = '--';
    lineStyleCell{3} = ':';
    lineStyleCell{4} = '-.';
    lineStyleCell{5} = '--';
    lineStyleCell{6} = '-';
    for i = 1:dim
        %plot(t_Interp, I_Interp(:,i), 'LineStyle', lineStyleCell{i}, 'color', colorCell{i}, 'LineWidth',0.5)
        plot(t_Interp, newInfected_Interp(:,i), 'LineStyle', lineStyleCell{i}, 'color', colorCell{i}, 'LineWidth',0.5)
    end
      set(gca, ...
          'Box'         , 'on'     , ...
          'TickDir'     , 'in'     , ...
          'TickLength'  , [.01 .01] , ...
          'XMinorTick'  , 'off'      , ...
          'YMinorTick'  , 'off'      , ...
          'YGrid'       , 'off'      , ...
          'XGrid'       , 'off'      , ...
          'XColor'      , [.1 .1 .1], ...
          'YColor'      , [.1 .1 .1], ...'YTick', [0:0.2:1], ... 'YLim', [0.5 1.001], ...'XLim', [0.5 sum(indSelected_logical) + 0.5], ...'YTick'       , 0:0.1:1, ...'XTick'       , 1.5:0.5:2.5, ...
          'Fontsize', fontSize, ...
          'LineWidth', 0.5)
    xlabel('Time (generations)')
    ylabel('Newly infected individuals (n)')
    %leg = legend('n', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5', 'n_6')
    leg = legend('n', 'n_{var 1}', 'n_{var 2}', 'n_{var 3}', 'n_{var 4}', 'n_{var 5}', 'n_{var 6}', 'Location', 'EastOutSide')
    set(leg,'color','none');
    set(leg, 'Edgecolor','none');


    %legend('I', ['I_1, R_{0,1} = ' num2str(R0_all(1))], ['I_2, R_{0,2} = ' num2str(R0_all(2))], ['I_3, R_{0,3} = ' num2str(R0_all(3))], ['I_4, R_{0,4} = ' num2str(R0_all(4))], ['I_5, R_{0,5} = ' num2str(R0_all(5))], ['I_6, R_{0,6} = ' num2str(R0_all(6))])
    %title(['Strain number :' num2str(drawStrain1)])
    grid off
    if(thisSet == 205)
        axis([0 100 0 1.25*max(sum(newInfected_Interp,2))])
    elseif(thisSet == 105)
        axis([0 100 0 1.25*max(sum(newInfected_Interp,2))])
    end
    %axis([0 50 0 1.5*1e5])
elseif(figABCDEFG_FLAG(3) == 2)
    axes(ha(104))
    plot(t_Interp, sum(newInfected_Interp,2), 'color', color_scheme11(99,:), 'LineWidth',1)
    hold on
    lineStyleCell{1} = '-';
    lineStyleCell{2} = '--';
    lineStyleCell{3} = ':';
    lineStyleCell{4} = '-.';
    lineStyleCell{5} = '--';
    lineStyleCell{6} = '-';
    for i = 1:dim
        plot(t_Interp, newInfected_Interp(:,i), 'LineStyle', lineStyleCell{i}, 'color', colorCell{i}, 'LineWidth',0.5)
    end
      set(gca, ...
          'Box'         , 'on'     , ...
          'TickDir'     , 'in'     , ...
          'TickLength'  , [.01 .01] , ...
          'XMinorTick'  , 'off'      , ...
          'YMinorTick'  , 'off'      , ...
          'YGrid'       , 'off'      , ...
          'XGrid'       , 'off'      , ...
          'XColor'      , [.1 .1 .1], ...
          'YColor'      , [.1 .1 .1], ...'YTick', [0:0.2:1], ... 'YLim', [0.5 1.001], ...'XLim', [0.5 sum(indSelected_logical) + 0.5], ...'YTick'       , 0:0.1:1, ...'XTick'       , 1.5:0.5:2.5, ...
          'Fontsize', fontSize, ...
          'LineWidth', 0.5)
    xlabel('Time (t)')
    ylabel('Newly infected individuals (n)')
    leg = legend('n', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5', 'n_6')
    set(leg,'color','none');
    set(leg, 'Edgecolor','none');


    %legend('I', ['I_1, R_{0,1} = ' num2str(R0_all(1))], ['I_2, R_{0,2} = ' num2str(R0_all(2))], ['I_3, R_{0,3} = ' num2str(R0_all(3))], ['I_4, R_{0,4} = ' num2str(R0_all(4))], ['I_5, R_{0,5} = ' num2str(R0_all(5))], ['I_6, R_{0,6} = ' num2str(R0_all(6))])
    %title(['Strain number :' num2str(drawStrain1)])
    grid off
    if(thisSet == 205)
        axis([0 100 0 1.5*max(sum(newInfected_Interp,2))])
    elseif(thisSet == 105)
        axis([0 100 0 1.5*max(sum(newInfected_Interp,2))])
    end
    %axis([0 50 0 1.5*1e5])
end

if(figABCDEFG_FLAG(4) == 1)          
    barwidthIn = 1;
    axes(ha(105))
    %figure
    b1 = bar((numSampledSeqsPerStrain), 'stacked')
    b1(1).BarWidth = barwidthIn;
    b1(2).BarWidth = barwidthIn;
    b1(3).BarWidth = barwidthIn;
    b1(4).BarWidth = barwidthIn;
    b1(5).BarWidth = barwidthIn;
    b1(6).BarWidth = barwidthIn;
    b1(1).FaceColor = color_scheme11(90,:);
    b1(2).FaceColor = color_scheme11(80,:);
    b1(3).FaceColor = color_scheme11(70,:);
    b1(4).FaceColor = color_scheme11(60,:);
    b1(5).FaceColor = color_scheme31(90,:);
    b1(6).FaceColor = color_scheme31(70,:);

    set(gca, ...
          'Box'         , 'on'     , ...
          'TickDir'     , 'in'     , ...
          'TickLength'  , [.01 .01] , ...
          'XMinorTick'  , 'off'      , ...
          'YMinorTick'  , 'off'      , ...
          'YGrid'       , 'off'      , ...
          'XGrid'       , 'off'      , ...
          'XColor'      , [.1 .1 .1], ...
          'YColor'      , [.1 .1 .1], ...'YTick', [0:0.2:1], ... 'YLim', [0.5 1.001], ...'XLim', [0.5 sum(indSelected_logical) + 0.5], ...'YTick'       , 0:0.1:1, ...'XTick'       , 1.5:0.5:2.5, ...
          'Fontsize', fontSize, ...
          'LineWidth', 0.5, ...
          'YScale', 'log')
    xlabel('Time (generations)')
    ylabel('Sequences sampled per day')
    axis([0 5 -0.2 0.2])
    %title(['Total sequences sampled: ' num2str(sum(sum(numSampledSeqsPerStrain)))])
end

if(figABCDEFG_FLAG(5) == 1)          
    axes(ha(106))
    for i = 1:size(genotypes, 1)

        plot(t_Interp, betaTimeVarying_Interp(:,i)*N/gamma(i), 'LineStyle', lineStyleCell{i}, 'color', colorCell{i}, 'linewidth', 1)
        hold on
    end
    set(gca, ...
      'Box'         , 'on'     , ...
      'TickDir'     , 'in'     , ...
      'TickLength'  , [.01 .01] , ...
      'XMinorTick'  , 'off'      , ...
      'YMinorTick'  , 'on'      , ...
      'YGrid'       , 'off'      , ...
      'XGrid'       , 'off'      , ...
      'XColor'      , [.1 .1 .1], ...
      'YColor'      , [.1 .1 .1], ...
      'YTick', [0:1:6], ... 'YLim', [0.5 1.001], ...'XLim', [0.5 sum(indSelected_logical) + 0.5], ...'YTick'       , 0:0.1:1, ...'XTick'       , 1.5:0.5:2.5, ...
      'Fontsize', fontSize, ...
      'LineWidth', 0.5)
     %axis([samplingTime(1) samplingTime(end) 0 1])
     ylabel({'Effective reproduction', 'number (R_t)'})
     %title(['Locus ' num2str(i) ', s_' num2str(i) ' = ' num2str(perSiteSelection(i))])
     %title('R_t estimates for NY from https://rt.live/')



    leg = legend('Var 1', 'Var 2', 'Var 3', 'Var 4', 'Var 5', 'Var 6', 'Location', 'EastOutSide')
    set(leg,'color','none');
    set(leg, 'Edgecolor','none');
    %plot(0:1:(200-1), ones(1,200), 'k:', 'linewidth', 0.5) 
    %plot(0:1:(200-1), ones(1,200), ':', 'color', color_scheme21(50,:), 'linewidth', 0.5) 
    xlabel('Time (generations)')
    
    if(thisSet == 205)
        axis([0 100 0 6])
    else(thisSet == 105)
        axis([0 100 0.5 2.5])
    end
end



if(thisSet == 105 || thisSet == 205)
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
          'YTick', [0:0.2:1], ... 'YLim', [0.5 1.001], ...'XLim', [0.5 sum(indSelected_logical) + 0.5], ...'YTick'       , 0:0.1:1, ...'XTick'       , 1.5:0.5:2.5, ...
          'Fontsize', fontSize, ...
          'LineWidth', 0.5)

         if(thisSet == 205)
            %axis([samplingTime(1) samplingTime(end) 0 1])
            axis([samplingTime(1) 100 0 1])
         elseif(thisSet == 105)
            %axis([samplingTime(1) samplingTime(end) 0 0.6])
            axis([samplingTime(1) 100 0 0.6])
         end
         ylabel('Frequency')
         title(['Locus: ' num2str(i)])
         if(i <= 3)
            set(gca, ...
              'XTickLabel', ' ')
         else
             xlabel('Time (generations)')
         end

    end

elseif(thisSet == 5)
    for i = 1:size(genotypes, 2)
        axes(ha(i))
        %plot(alleleFreq(:,i), 'color', color_scheme31(70,:), 'linewidth', 1)
        %hold on
        plot(q(1:end-1,i), 'LineStyle', '-', 'linewidth', 0.5)
        hold on
        plot(q_z(2:end,i), 'LineStyle', '--', 'linewidth', 0.5)
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
          'YTick', [0:0.2:1], ... 'YLim', [0.5 1.001], ...'XLim', [0.5 sum(indSelected_logical) + 0.5], ...'YTick'       , 0:0.1:1, ...'XTick'       , 1.5:0.5:2.5, ...
          'Fontsize', fontSize, ...
          'LineWidth', 0.5)

         axis([samplingTime(1) samplingTime(end) 0 0.4])

         ylabel('Frequency')
         title(['Locus: ' num2str(i)])
         if( i == 1)
             leg = legend('x_i(t)', 'z_i(t+1)', 'Location', 'SouthEast');
             set(leg,'color','none');
             set(leg, 'Edgecolor','none');
         end
         if(i <= 3)
            set(gca, ...
              'XTickLabel', ' ')
         else
             xlabel('Time (generations)')
         end

    end
 
end



ta = annotation('textbox',dimDummy,'String','a','FitBoxToText','on')
ta.Units = 'centimeter';
ta.Position = [0 yDim-0.4 0.7 0.5];%[0.55 4.1 0.7 0.5];
ta.LineStyle = 'none';
ta.FontWeight = 'bold';
ta.FontSize = 12;

ta2 = annotation('textbox',dimDummy,'String','b','FitBoxToText','on')
ta2.Units = 'centimeter';
ta2.Position = [0 yDim-0.4-height1-vgap1 0.7 0.5];%[0.55 4.1 0.7 0.5];
ta2.LineStyle = 'none';
ta2.FontWeight = 'bold';
ta2.FontSize = 12;

tb = annotation('textbox',dimDummy,'String','c','FitBoxToText','on')
tb.Units = 'centimeter';
tb.Position = [0+width3+hgap2 yDim-0.4 0.7 0.5];%[0.55 4.1 0.7 0.5];
tb.LineStyle = 'none';
tb.FontWeight = 'bold';
tb.FontSize = 12;

tc = annotation('textbox',dimDummy,'String','d','FitBoxToText','on')
tc.Units = 'centimeter';
tc.Position = [0+width3+hgap2 yDim-0.4-height1-vgap1 0.7 0.5];%[0.55 4.1 0.7 0.5];
tc.LineStyle = 'none';
tc.FontWeight = 'bold';
tc.FontSize = 12;


tf = annotation('textbox',dimDummy,'String','e','FitBoxToText','on')
tf.Units = 'centimeter';
tf.Position = [0 yDim-0.8-height_A_E-vgap1*2 0.7 0.5];%[0.55 4.1 0.7 0.5];
tf.LineStyle = 'none';
tf.FontWeight = 'bold';
tf.FontSize = 12;

% tg = annotation('textbox',dimDummy,'String','F','FitBoxToText','on')
% tg.Units = 'centimeter';
% tg.Position = [0+width3+hgap2 yDim-0.4-height_A_E-vgap1*2 0.7 0.5];%[0.55 4.1 0.7 0.5];
% tg.LineStyle = 'none';
% tg.FontWeight = 'bold';
% tg.FontSize = 12;
%%
saveFigs = 1
if(saveFigs == 1)
    figname = ['SIR_Suppfig_' SetProfileStr];
    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 xDim yDim])%,[0 0 8 6.45])% ,[0 0 8 6])
    %set(gcf, 'renderer', 'painters');
    print(figname, '-dpng','-r400')
    set(gcf, 'PaperSize', [xDim yDim])
    print(figname, '-dpdf', '-fillpage')
end
