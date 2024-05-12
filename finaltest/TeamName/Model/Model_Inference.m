clear all; close all; clc;

rng(1); % Reproducibility

global Material
Material = 'Material A';

main();

%% Run Main
function main()

    [data_B, data_F, data_T] = load_dataset();
    core_loss(data_B, data_F, data_T);

end

%% Load Dataset
function [data_B, data_F, data_T] = load_dataset()

    global Material
    
    data_B = readmatrix(['./Testing/', Material, '/B_Field.csv']); % N by 1024, in T
    data_F = readmatrix(['./Testing/', Material, '/Frequency.csv']); % N by 1, in Hz
    data_T = readmatrix(['./Testing/', Material, '/Temperature.csv']); % N by 1, in C

end

%% Calculate Core Loss
function core_loss(data_B, data_F, data_T)

    global Material

    %================ Wrap your model or algorithm here===================%
    
    % Here's just an example:
    
    len = length(data_F);
    data_P = rand(len,1);
    
    %=====================================================================%
    
    writematrix(data_P, ['./Testing/Volumetric_Loss_', Material, '.csv']);
    
    fprintf('Model inference is finished! \n');

end

%% End