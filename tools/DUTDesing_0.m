% The input required is the limits of the Data Acquisition System and the
% core to be designed
% Used to get a plot of the range of Hdc, f, and Bac roughly capable of measuring
% Contact: Diego Serrano, ds9056@princeton.edu, Princeton University

%% Clear previous varaibles and add the paths
clear % Clear variable in the workspace
clc % Clear command window
close all % Close all open figures
addpath('Scripts') % Add the folder where the scripts are located
cd ..; % To go to the previous folder 

%% Set the style for the plots
PlotStyle; close;

%% Core parameters

% Core
Material = 'N87';
ki = 0.1518;
alpha = 1.4722;
beta = 2.6147;
mu_ra_max = 3800;  % Max amplitude permeability (relative)
Bsat = 390e-3; % mT
% Shape
Shape = 'R34.0X20.5X12.5';
Le = 0.08206; % Effective length (m)
Ae = 0.0000826; % Effective area (m2)
Ve = 0.000006778; % Effective volume (m3)
Al = 2790e-9; % Inductance per turn squared (H) 
% Turns 
N = 5; % Primary number of turns

L = N^2*Al;
k = ki*2^(beta+1)*pi^(alpha-1)*(0.2761+1.7061/(alpha+1.354));
%% DAS limits
Vdc_max = 80.0; % Maximum input voltage of the voltage supply (limited by the 100V series capacitor)
Vdc_min = 5.0 ; % Minimum imposed by this specific voltage supply (cannot be set to 1 V as the power supply resets the limits in RMT mode)
Vac_max = 50.0;  % The output of the power amplifier looks distorted if the amplitude is above 50 V
Vac_min = 1.0;  % To ensure a good resolution of the waveform
I_max = 2;  % Maximum total current
PV_min = 1000;  % Minimum losses based on the iGSE equations, only an estimate
PV_max = 5000000;  % Maximum losses, points are skipped if outside this range
Loss_min = 1e-3; % W 
%% Iterate Hdc, f, and Bac
n_sine=0;
n_tri=0;
n_tra=0;

for Hdc=0:2.5:2000
    Idc=Hdc*Le/N;
    Bdc=Hdc*1.256637e-6*mu_ra_max;
    for logf=3:0.05:8
        f=10^logf;
        w=2*pi*f;
        for logB=-3:0.05:0.3
            Bac=10^logB;
            Bpk=Bdc+Bac;

            % Sinusoidal
            Vac=N*Ae*Bac*w;
            Iac=Vac/L/w;
            Ipk=Idc+Iac;
            PV=k*Bac^beta*f^alpha;
            Loss=PV*Ve;
            if Bpk<Bsat && ...
               Vac>Vac_min && ...
               Vac<Vac_max && ...
               Ipk<I_max && ...
               PV>PV_min && ...
               PV<PV_max && ...
               Loss>Loss_min

               n_sine=n_sine+1;
               H_sine(n_sine)=Hdc;
               F_sine(n_sine)=f;
               B_sine(n_sine)=Bac;
               P_sine(n_sine)=PV;
            end

            % Triangular 50%
            Vac=N*Ae*4*Bac*f;
            Iac=Vac/L/4/f;
            Ipk=Idc+Iac;
            %PV=ki*f*(2*Bac)^(beta-alpha)*(2*Bac/0.5*f)^alpha*2*0.5/f;
            PV=ki*2^(beta+1)*Bac^beta*f^alpha*0.5^(1-alpha);
            Loss=PV*Ve;
            if Bpk<Bsat && ...
               Vac>Vdc_min && ...
               Vac<Vdc_max && ...
               Ipk<I_max && ...
               PV>PV_min && ...
               PV<PV_max && ...
               Loss>Loss_min

               n_tri=n_tri+1;
               H_tri(n_tri)=Hdc;
               F_tri(n_tri)=f;
               B_tri(n_tri)=Bac;
               P_tri(n_tri)=PV;
            end

            % Trapezoidal 10% rise and 10% fall
            Vac=N*Ae*20*Bac*f;
            Iac=Vac/L/20/f;
            Ipk=Idc+Iac;
            PV=ki*2^(beta+1)*Bac^beta*f^alpha*0.5^(1-alpha);
            Loss=PV*Ve;
            if Bpk<Bsat && ...
               Vac>Vdc_min && ...
               Vac<Vdc_max && ...
               Ipk<I_max && ...
               PV>PV_min && ...
               PV<PV_max && ...
               Loss>Loss_min
               n_tra=n_tra+1;

               H_tra(n_tra)=Hdc;
               F_tra(n_tra)=f;
               B_tra(n_tra)=Bac;
               P_tra(n_tra)=PV;
            end
        end
    end
end
%% Plot the range where data can more or less be measured
figure;

subplot(1,3,1)
hold on;
scatter3(B_sine*1e3, F_sine*1e-3, H_sine, 5, P_sine*1e-3, 'filled');
plot([10 10],[50 500],'-k', 'linewidth', 5)
plot([300 300],[50 500],'-k', 'linewidth', 5)
plot([10 300],[50 50],'-k', 'linewidth', 5)
plot([10 300],[500 500],'-k', 'linewidth', 5)
c = colorbar; c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';
c.Label.String = "$P_V$ [kW/m$^3$]"; 
xlabel('$B_{ac}$ [mT]');
ylabel('$f$ [kHz]');
zlabel('$H_{dc}$ [A/m]')

set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); set(gca, 'ColorScale', 'log'); view(2);
title([Material, ', ', Shape, ', ' num2str(N), ' turns, Sinusoidal'])

subplot(1,3,2)
hold on;
scatter3(B_tri*1e3, F_tri*1e-3, H_tri, 5, P_tri*1e-3, 'filled');
plot([10 10],[50 500],'-k', 'linewidth', 5)
plot([300 300],[50 500],'-k', 'linewidth', 5)
plot([10 300],[50 50],'-k', 'linewidth', 5)
plot([10 300],[500 500],'-k', 'linewidth', 5)
c = colorbar; c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';
c.Label.String = "$P_V$ [kW/m$^3$]"; 
xlabel('$B_{ac}$ [mT]');
ylabel('$f$ [kHz]');
zlabel('$H_{dc}$ [A/m]')
set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); set(gca, 'ColorScale', 'log'); view(2);
title([Material, ', ', Shape, ', ' num2str(N), ' turns, Triangular 50\%'])

subplot(1,3,3)
hold on;
scatter3(B_tra*1e3, F_tra*1e-3, H_tra, 5, P_tra*1e-3, 'filled');
plot([10 10],[50 500],'-k', 'linewidth', 5)
plot([300 300],[50 500],'-k', 'linewidth', 5)
plot([10 300],[50 50],'-k', 'linewidth', 5)
plot([10 300],[500 500],'-k', 'linewidth', 5)
c = colorbar; c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';
c.Label.String = "$P_V$ [kW/m$^3$]"; 
xlabel('$B_{ac}$ [mT]');
ylabel('$f$ [kHz]');
zlabel('$H_{dc}$ [A/m]')
set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); set(gca, 'ColorScale', 'log'); view(2);
title([Material, ', ', Shape, ', ' num2str(N), ' turns, Trapezoidal 10\% rise/fall'])

set(gcf,'units','points','position',[100,100,1300,400])
set(findall(gcf,'-property','FontSize'),'FontSize',12)