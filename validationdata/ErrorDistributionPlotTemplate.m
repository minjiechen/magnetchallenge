close all; clear all; clc;

set(0,'DefaultAxesFontName', 'LM Roman 12')
set(0,'DefaultAxesFontSize', 20)
set(0,'DefaultAxesFontWeight', 'Bold')
set(0,'DefaultTextFontname', 'LM Roman 12')
set(0,'DefaultTextFontSize', 20)
set(0,'DefaultTextFontWeight', 'Bold')

set(groot, 'defaultFigureColor', [1 1 1]); % White background for pictures
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
% It is not possible to set property defaults that apply to the colorbar label. 
% set(groot, 'defaultAxesFontSize', 12);
% set(groot, 'defaultColorbarFontSize', 12);
% set(groot, 'defaultLegendFontSize', 12);
% set(groot, 'defaultTextFontSize', 12);
set(groot, 'defaultAxesXGrid', 'on');
set(groot, 'defaultAxesYGrid', 'on');
set(groot, 'defaultAxesZGrid', 'on');
set(groot, 'defaultAxesXMinorTick', 'on');
set(groot, 'defaultAxesYMinorTick', 'on');
set(groot, 'defaultAxesZMinorTick', 'on');
set(groot, 'defaultAxesXMinorGrid', 'on', 'defaultAxesXMinorGridMode', 'manual');
set(groot, 'defaultAxesYMinorGrid', 'on', 'defaultAxesYMinorGridMode', 'manual');
set(groot, 'defaultAxesZMinorGrid', 'on', 'defaultAxesZMinorGridMode', 'manual');

red = [200 36 35]/255;
blue = [40 120 181]/255;
a = 170;
gray = [a,a,a]/255;

%Update the material variable!
Material = '3C90';

% Begin Loading Data:

%Load Predictions Spreadsheet and Measured Spreadsheet
% Then calculate error using absolute relative error

pred = load(['.\validationdata\Result\pred_',Material,'.csv']);
meas = load(['.\validationdata\',Material,'\Volumetric_Loss.csv']);

% Relative error is the metric of interest, using absolute values:
error  = 100 * abs(meas - pred) ./ abs(meas);

% Alternatively, load spreadsheet with error values already calculated

%error = load('ERRORPath');

figure(20);
histogram(error,"NumBins",50,'FaceColor',blue,'FaceAlpha',1,'Normalization','probability'); hold on;

y1 = 0.09;
y = linspace(0,y1,20);
plot(mean(abs(error))*ones(size(y)),y,'--','Color',red,'LineWidth',2);
text(mean(abs(error))+0.25,y1,['Avg=',num2str(mean(abs(error)),4),'\%'],'Color',red);

y2 = 0.07;
y = linspace(0,y2,20);
plot(prctile(error,95)*ones(size(y)),y,'--','Color',red,'LineWidth',2);
text(prctile(error,95)+0.25,y2,['95-Prct=',num2str(prctile(error,95),4),'\%'],'Color',red);

% y3 = 0.05;
% y = linspace(0,y3,20);
% plot(prctile(error,99)*ones(size(y)),y,'--','Color',red,'LineWidth',2);
% text(prctile(error,99)+0.25,y3,['99-Prct=',num2str(prctile(error,99),4),'\%'],'Color',red);

y4 = 0.02;
y = linspace(0,y4,20);
plot(max(abs(error))*ones(size(y)),y,'--','Color',red,'LineWidth',2);
text(max(abs(error))-0.6,y4,['Max=',num2str(max(abs(error)),4),'\%'],'Color',red);

xlabel('Relative Error of Core Loss [\%]');
ylabel('Ratio of Data Points');
%Set x and y limits, if needed
%  xlim([0 19]);
%  ylim([0 0.13]);
% mean(error)

set(gca, 'box', 'on')
set(gcf,'Position',[850,550,780,430])
title(['Error Distribution for ', Material])
subtitle(['Avg=',num2str(mean(abs(error)),4),'\%, ', ...
    '95-Prct=',num2str(prctile(error,95),3),'\%, ', ...
    '99-Prct=',num2str(prctile(error,99),3),'\%, ', ...
    'Max=',num2str(max(abs(error)),4),'\%'], ...
    'FontSize',18)

[mean(abs(error)) rms(error) prctile(error,95) max(abs(error))];
