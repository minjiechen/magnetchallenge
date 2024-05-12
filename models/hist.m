clear all,clc,close all
% Change default axes fonts.
set(0,'DefaultAxesFontName', 'Arial')
set(0,'DefaultAxesFontSize', 12)

% Change default text fonts.
set(0,'DefaultTextFontname', 'Arial')
set(0,'DefaultTextFontSize', 12)

data = csvread('Data.csv');
std_A=data(:,1)*100;
error_A=data(:,2)*100;
std_B=data(:,3)*100;
error_B=data(:,4)*100;
std_C=data(:,5)*100;
error_C=data(:,6)*100;
std_D=data(:,7)*100;
error_D=data(:,8)*100;
std_E=data(:,9)*100;
error_E=data(:,10)*100;


figure(1)
subplot 151, loglog(std_A,error_A,'r*'),axis([1 100 1 100]);grid on;
subplot 152, loglog(std_B,error_B,'b*'),axis([1 100 1 100]);grid on;
subplot 153, loglog(std_C,error_C,'g*'),axis([1 100 1 100]);grid on;
subplot 154, loglog(std_D,error_D,'k*'),axis([1 100 1 100]);grid on;
subplot 155, loglog(std_E,error_E,'m*'),axis([1 100 1 100]);grid on;