%%
% -------------------------------------------------------------------------
% 2. plot results
% -------------------------------------------------------------------------
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

xDim = 10;
yDim = 8;
fig1 = figure('Units','centimeters', ...
                'Position', [2 2 xDim yDim]);
% row 1 :  2 + 1.81 + 0.2 + 1.81 + 0.2 + 1.81 + 1.5  + 4 + 1.5 + 4 + + 0.5
% row 2:   1 + 4.25 + 0.5 + 4.25 + 0.5 + 4.25 + 0.5 + 4.25 + 0.5
leftMargin = 1.5;
rightMargin = 0.5;
bottomMargin = 1;
topMargin = 0.5;

vgap = 0.6;
hgap = 1.5;
%width = 10 - leftMargin - rightMargin;
width = (xDim - leftMargin - rightMargin);
height1 = (yDim - bottomMargin - topMargin);%2.65;

ha(101) = axes('Units','centimeters', ...
                'Position',[leftMargin bottomMargin width height1], ...
                'XTickLabel','', ...
                'YTickLabel','');
axes(ha(101))            
plot(t, S(:,1),'b','LineWidth',1)%, axis([0 1000 0 0.2])
hold on
plot(t, sum(I,2), 'r','LineWidth',2)
plot(t, sum(R,2), 'k','LineWidth',1)
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
      'Fontsize', 8, ...
      'LineWidth', 0.5)
xlabel('Time (t)')
ylabel('Population (N)')
legend('S', 'I', 'R')
%title(['Strain number :' num2str(drawStrain1)])
grid on
axis([0 time+20 0 1e7])

% plot SIR for all t steps calculated by ode45
% figure
% plot(t, sum(I,2), 'color', color_scheme11(99,:), 'LineWidth',2)
% hold on
% for i = 1:dim
%     plot(t, I(:,i),  'color', colorCell{i}, 'LineWidth',1)
% end
% xlabel('Time (t)')
% ylabel('Population (N)')
% legend('I', 'I_1', 'I_2', 'I_3', 'I_4', 'I_5', 'I_6')
% %title(['Strain number :' num2str(drawStrain1)])
% grid on
% axis([0 200 0 1e7/2])

if(saveFigs == 1)
    figname = ['Multistrain_SIR'];
    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 xDim yDim])%,[0 0 8 6.45])% ,[0 0 8 6])
    %set(gcf, 'renderer', 'painters');
    print(figname, '-dpng','-r400')
    %set(gcf, 'PaperSize', [xDim yDim])
    %print(figname, '-dpdf', '-fillpage')
end


% plot selection strengths
xDim = 10+2;
yDim = 3+6;
fig2 = figure('Units','centimeters', ...
                'Position', [2 2 xDim yDim]);
% row 1 :  2 + 1.81 + 0.2 + 1.81 + 0.2 + 1.81 + 1.5  + 4 + 1.5 + 4 + + 0.5
% row 2:   1 + 4.25 + 0.5 + 4.25 + 0.5 + 4.25 + 0.5 + 4.25 + 0.5
leftMargin = 1.75;
rightMargin = 0.2+2;
bottomMargin = 1;
topMargin = 0.2;

vgap = 1.5;
%width = 10 - leftMargin - rightMargin;
width = (xDim - leftMargin - rightMargin);
height1 = (yDim - bottomMargin - topMargin - vgap)/3;

ha(102) = axes('Units','centimeters', ...
                'Position',[leftMargin bottomMargin+vgap+height1*2 width height1], ...
                'XTickLabel','', ...
                'YTickLabel','');
ha(103) = axes('Units','centimeters', ...
                'Position',[leftMargin bottomMargin width height1*2], ...
                'XTickLabel','', ...
                'YTickLabel','');

axes(ha(102))
bar(perSiteSelection)
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
      'YTick'       , [-maxSelVal -0.5*maxSelVal 0 0.5*maxSelVal maxSelVal], ...'YTick'       , [-0.1  -0.05 0 0.05 0.1], ...
      'Fontsize', 8, ...
      'LineWidth', 0.5)
xlabel('Locus')
ylabel({'selection', 'coefficient'} )
grid on
axes(ha(103))
heatmapMtx = genotypes;
for i = 1:L
    xValCell{i} = i;
end
for j = 1:numStrains
    yvalCell{j} = ['Variant ' num2str(j)];
end
heatmap(xValCell, yvalCell, heatmapMtx,'ColorbarVisible','off')
set(gca, ...
      'Fontsize', 8)
xlabel('Locus')
 
for i =1:numStrains
    dimDummy = [0.1 0.1 0.1 0.1]; % dummy position (in normalized units)
    textNA = annotation('textbox',dimDummy,'String',['I_{0,str ' num2str(num2str(i)) '}: ' num2str(I_init(i))],'FitBoxToText','on');
    textNA.Units = 'centimeter';
    textNA.Position = [xDim-rightMargin+0.1 height1*2+bottomMargin-(0.5*i + 0.2*(i-1)) 2 0.5];
    textNA.LineStyle = 'none';
    textNA.FontSize = 9;
end


if(saveFigs == 1)
    figname = ['perSiteSelction'];
    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 xDim yDim])%,[0 0 8 6.45])% ,[0 0 8 6])
    %set(gcf, 'renderer', 'painters');
    print(figname, '-dpng','-r400')
    %set(gcf, 'PaperSize', [xDim yDim])
    %print(figname, '-dpdf', '-fillpage')
end


% plot SIR for all t = 1:1:x
xDim = 10;
yDim = 8;
fig3 = figure('Units','centimeters', ...
                'Position', [12 12 xDim yDim]);
% row 1 :  2 + 1.81 + 0.2 + 1.81 + 0.2 + 1.81 + 1.5  + 4 + 1.5 + 4 + + 0.5
% row 2:   1 + 4.25 + 0.5 + 4.25 + 0.5 + 4.25 + 0.5 + 4.25 + 0.5
leftMargin = 1.5;
rightMargin = 0.5;
bottomMargin = 1;
topMargin = 0.5;

vgap = 0.6;
hgap = 1.5;
%width = 10 - leftMargin - rightMargin;
width = (xDim - leftMargin - rightMargin);
height1 = (yDim - bottomMargin - topMargin);

ha(104) = axes('Units','centimeters', ...
                'Position',[leftMargin bottomMargin width height1], ...
                'XTickLabel','', ...
                'YTickLabel','');
axes(ha(104))
plot(t_Interp, sum(I_Interp,2), 'color', color_scheme11(99,:), 'LineWidth',2)
hold on
lineStyleCell{1} = '-';
lineStyleCell{2} = '--';
lineStyleCell{3} = ':';
lineStyleCell{4} = '-.';
lineStyleCell{5} = '--';
lineStyleCell{6} = '-';
for i = 1:dim
    plot(t_Interp, I_Interp(:,i), 'LineStyle', lineStyleCell{i}, 'color', colorCell{i}, 'LineWidth',1)
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
      'Fontsize', 8, ...
      'LineWidth', 0.5)
xlabel('Time (t)')
ylabel('Population (N)')
if(numStrains == 6)
    legend('I', ['I_1, R_{0,1} = ' num2str(R0_all(1))], ['I_2, R_{0,2} = ' num2str(R0_all(2))], ['I_3, R_{0,3} = ' num2str(R0_all(3))], ['I_4, R_{0,4} = ' num2str(R0_all(4))], ['I_5, R_{0,5} = ' num2str(R0_all(5))], ['I_6, R_{0,6} = ' num2str(R0_all(6))])
elseif(numStrains == 2)
    legend('I', ['I_1, R_{0,1} = ' num2str(R0_all(1))], ['I_2, R_{0,2} = ' num2str(R0_all(2))])
end
%title(['Strain number :' num2str(drawStrain1)])
grid on
axis([0 200 0 2*max(sum(I_Interp,2))])

if(saveFigs == 1)
    figname = ['Infected_allStrains'];
    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 xDim yDim])%,[0 0 8 6.45])% ,[0 0 8 6])
    %set(gcf, 'renderer', 'painters');
    print(figname, '-dpng','-r400')
    %set(gcf, 'PaperSize', [xDim yDim])
    %print(figname, '-dpdf', '-fillpage')
end

% barchart of sssampled sequences per day
% figure
% b1 = bar(sum(numSampledSeqsPerStrain,2))
% b1.BarWidth = 1;
% b1.FaceColor = color_scheme31(70,:);


xDim = 10;
yDim = 8;
fig4 = figure('Units','centimeters', ...
                'Position', [2 12 xDim yDim]);
% row 1 :  2 + 1.81 + 0.2 + 1.81 + 0.2 + 1.81 + 1.5  + 4 + 1.5 + 4 + + 0.5
% row 2:   1 + 4.25 + 0.5 + 4.25 + 0.5 + 4.25 + 0.5 + 4.25 + 0.5
leftMargin = 1.5;
rightMargin = 0.5;
bottomMargin = 1;
topMargin = 0.5;

vgap = 0.6;
hgap = 1.5;
%width = 10 - leftMargin - rightMargin;
width = (xDim - leftMargin - rightMargin);
height1 = (yDim - bottomMargin - topMargin);%2.65;

ha(105) = axes('Units','centimeters', ...
                'Position',[leftMargin bottomMargin width height1], ...
                'XTickLabel','', ...
                'YTickLabel','');
axes(ha(105))
b1 = bar((numSampledSeqsPerStrain), 'stacked')
b1(1).BarWidth = 1;
b1(2).BarWidth = 1;
b1(3).BarWidth = 1;
b1(4).BarWidth = 1;
b1(5).BarWidth = 1;
b1(6).BarWidth = 1;
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
      'Fontsize', 8, ...
      'LineWidth', 0.5)
xlabel('Time (t)')
ylabel('Count of sequences sampled per day')
title(['Total sequences sampled: ' num2str(sum(sum(numSampledSeqsPerStrain)))])
if(saveFigs == 1)
    figname = ['sequenceSampling_SIR'];
    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 xDim yDim])%,[0 0 8 6.45])% ,[0 0 8 6])
    %set(gcf, 'renderer', 'painters');
    print(figname, '-dpng','-r400')
    %set(gcf, 'PaperSize', [xDim yDim])
    %print(figname, '-dpdf', '-fillpage')
end


% % make seqDatesVec for all strains combined
% seqsCollectedPerDay = sum(numSampledSeqsPerStrain,2);
% totalSequences = sum(seqsCollectedPerDay);
% numSeqDays = length(seqsCollectedPerDay);
% seqDatesVec = zeros(1, totalSequences); % contains the dats on whihc each sequence was obtained
% count = 0;
% 
% for k = 1:numSeqDays
%     seqDatesVec(count+1:(count+seqsCollectedPerDay(k))) = k;
%     count = count + seqsCollectedPerDay(k);    
% end
%
% figure
% histogram(seqDatesVec)
% 
% % make seqDatesVec for all strains separate
% numSeqDaysThisStrain = zeros(1, numStrains);
% for j = 1:numStrains
%     seqsCollectedPerDayThisStrainTemp = numSampledSeqsPerStrain(:,j);
%     totalSequencesThisStrainTemp = sum(seqsCollectedPerDayThisStrainTemp);
%     numSeqDaysThisStrainTemp = length(seqsCollectedPerDayThisStrainTemp);
%     seqDatesVecThisStrainTemp = zeros(1, totalSequencesThisStrainTemp); % contains the dats on whihc each sequence was obtained
%     countThisStrainTemp = 0;
% 
%     for k = 1:numSeqDaysThisStrainTemp
%         seqDatesVecThisStrainTemp(countThisStrainTemp+1:(countThisStrainTemp+seqsCollectedPerDayThisStrainTemp(k))) = k;
%         countThisStrainTemp = countThisStrainTemp + seqsCollectedPerDayThisStrainTemp(k);    
%     end
%     numSeqDaysThisStrain(j) = numSeqDaysThisStrainTemp;
%     seqDatesVecThisStrainCell{j} = seqDatesVecThisStrainTemp;
% end
%
% figure
% histogram(seqDatesVec, 0.5:1:80.5)

%%
% plot allele frequencies
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
    plot(alleleFreq(:,i), 'color', color_scheme31(70,:), 'linewidth', 1)
    hold on
    plot(alleleFreqHigh(:,i), 'LineStyle', '-', 'color', color_scheme21(70,:), 'linewidth', 0.5)
    
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
      'Fontsize', 8, ...
      'LineWidth', 0.5)
     axis([samplingTime(1) samplingTime(end) 0 1])
     ylabel('Frequency')
     title(['Locus ' num2str(i) ', s_' num2str(i) ' = ' num2str(perSiteSelection(i))])
     if(i <= 8)
        set(gca, ...
          'XTickLabel', ' ')
     else
         xlabel('Time (t)')
     end
     
end

if(saveFigs == 1)
    figname = ['AlleleTrajs'];
    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 xDim yDim])%,[0 0 8 6.45])% ,[0 0 8 6])
    %set(gcf, 'renderer', 'painters');
    print(figname, '-dpng','-r400')
    %set(gcf, 'PaperSize', [xDim yDim])
    %print(figname, '-dpdf', '-fillpage')
end

%%
% plot genotypefrequencies
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
% ha(7) = axes('Units','centimeters', ...
%                 'Position',[leftMargin bottomMargin+1*vgap+1*height1 width height1], ...
%                 'XTickLabel','', ...
%                 'YTickLabel','');
% ha(8) = axes('Units','centimeters', ...
%                 'Position',[leftMargin+hgap+width bottomMargin+1*vgap+1*height1 width height1], ...
%                 'XTickLabel','', ...
%                 'YTickLabel',''); 
% ha(9) = axes('Units','centimeters', ...
%                 'Position',[leftMargin bottomMargin+0*vgap+0*height1 width height1], ...
%                 'XTickLabel','', ...
%                 'YTickLabel','');
% ha(10) = axes('Units','centimeters', ...
%                 'Position',[leftMargin+hgap+width bottomMargin+0*vgap+0*height1 width height1], ...
%                 'XTickLabel','', ...
%                 'YTickLabel','');   
            
for i = 1:size(genotypes, 1)
    axes(ha(i))
    %plot(genFreq(:,i), 'color', color_scheme31(70,:), 'linewidth', 1)
    %hold on
    plot(genFreqHigh(:,i), 'LineStyle', '--', 'color', color_scheme21(70,:), 'linewidth', 0.5)
    
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
      'Fontsize', 8, ...
      'LineWidth', 0.5)
     axis([samplingTime(1) samplingTime(end) 0 1])
     ylabel('Frequency')
     title(['Genotype ' num2str(i) ', g_' num2str(i) ' = ' num2str(sum(perSiteSelection.*genotypes(i,:)))])
     if(i <= 4)
        set(gca, ...
          'XTickLabel', ' ')
     else
         xlabel('Time (t)')
     end
     
end

if(saveFigs == 1)
    figname = ['GenoTrajs'];
    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 xDim yDim])%,[0 0 8 6.45])% ,[0 0 8 6])
    %set(gcf, 'renderer', 'painters');
    print(figname, '-dpng','-r400')
    %set(gcf, 'PaperSize', [xDim yDim])
    %print(figname, '-dpdf', '-fillpage')
end

%% plot R0,t 
xDim = 20;
yDim = 6;
fig6 = figure('Units','centimeters', ...
                'Position', [2 2 xDim yDim]);
leftMargin = 1.5;
rightMargin = 0.5;
bottomMargin = 1;
topMargin = 0.5;

vgap = 0.6;
hgap = 1.5;
width = (xDim - leftMargin - rightMargin);
height1 = (yDim - bottomMargin - topMargin);%2.65;

ha(1) = axes('Units','centimeters', ...
                'Position',[leftMargin bottomMargin width height1], ...
                'XTickLabel','', ...
                'YTickLabel','');
axes(ha(1))        
for i = 1:size(genotypes, 1)
    
    plot(t_Interp, betaTimeVarying_Interp(:,i)*N/gamma(i), 'color', colorCell{i}, 'linewidth', 1)
    hold on
    
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
      'Fontsize', 8, ...
      'LineWidth', 0.5)
     %axis([samplingTime(1) samplingTime(end) 0 1])
     ylabel('R_t')
     %title(['Locus ' num2str(i) ', s_' num2str(i) ' = ' num2str(perSiteSelection(i))])
     title('R_t estimates for NY from https://rt.live/')
     
     
end
plot(0:1:(200-1), ones(1,200), '--', 'color', color_scheme21(50,:), 'linewidth', 0.5) 
    xlabel('Time (t)')
if(saveFigs == 1)
    figname = ['R_t'];
    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 xDim yDim])%,[0 0 8 6.45])% ,[0 0 8 6])
    %set(gcf, 'renderer', 'painters');
    print(figname, '-dpng','-r400')
    %set(gcf, 'PaperSize', [xDim yDim])
    %print(figname, '-dpdf', '-fillpage')
end