clear all; %close all; clc;

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
function [data_B, data_F, data_T, data_P] = load_dataset()

    global Material
    
    data_B = readmatrix(['./Testing/', Material, '/B_Field.csv']); % N by 1024, in T
    data_F = readmatrix(['./Testing/', Material, '/Frequency.csv']); % N by 1, in Hz
    data_T = readmatrix(['./Testing/', Material, '/Temperature.csv']); % N by 1, in C

end

%% Calculate Core Loss
function core_loss(data_B, data_F, data_T)

    global Material
    Model2=load("MagNetModel2.mat");
    Model =load("MagNetModel.mat");
    %================ Chose material parameters===================%
    if     strcmp(Material,'Material A')
        
        Tri_parameters=[Model2.Model2.Mat_MaterialA.T25.Parameters;
                        Model2.Model2.Mat_MaterialA.T50.Parameters;
                        Model2.Model2.Mat_MaterialA.T70.Parameters;
                        Model2.Model2.Mat_MaterialA.T90.Parameters];
        Sin_parameters1=Model.Model.Mat_MaterialA.T25.SinParameters;
        Sin_parameters2=Model.Model.Mat_MaterialA.T50.SinParameters;
        Sin_parameters3=Model.Model.Mat_MaterialA.T70.SinParameters;
        Sin_parameters4=Model.Model.Mat_MaterialA.T90.SinParameters;
    elseif strcmp(Material,'Material B')
        Tri_parameters=[Model2.Model2.Mat_MaterialB.T25.Parameters;
                        Model2.Model2.Mat_MaterialB.T50.Parameters;
                        Model2.Model2.Mat_MaterialB.T70.Parameters;
                        Model2.Model2.Mat_MaterialB.T90.Parameters];
        Sin_parameters1=Model.Model.Mat_MaterialB.T25.SinParameters;
        Sin_parameters2=Model.Model.Mat_MaterialB.T50.SinParameters;
        Sin_parameters3=Model.Model.Mat_MaterialB.T70.SinParameters;
        Sin_parameters4=Model.Model.Mat_MaterialB.T90.SinParameters;
    elseif strcmp(Material,'Material C')
        Tri_parameters=[Model2.Model2.Mat_MaterialC.T25.Parameters;
                        Model2.Model2.Mat_MaterialC.T50.Parameters;
                        Model2.Model2.Mat_MaterialC.T70.Parameters;
                        Model2.Model2.Mat_MaterialC.T90.Parameters];
        Sin_parameters1=Model.Model.Mat_MaterialC.T25.SinParameters;
        Sin_parameters2=Model.Model.Mat_MaterialC.T50.SinParameters;
        Sin_parameters3=Model.Model.Mat_MaterialC.T70.SinParameters;
        Sin_parameters4=Model.Model.Mat_MaterialC.T90.SinParameters;
    elseif strcmp(Material,'Material D')
        Tri_parameters=[Model2.Model2.Mat_MaterialD.T25.Parameters;
                        Model2.Model2.Mat_MaterialD.T50.Parameters;
                        Model2.Model2.Mat_MaterialD.T70.Parameters;
                        Model2.Model2.Mat_MaterialD.T90.Parameters];
        Sin_parameters1=Model.Model.Mat_MaterialD.T25.SinParameters;
        Sin_parameters2=Model.Model.Mat_MaterialD.T50.SinParameters;
        Sin_parameters3=Model.Model.Mat_MaterialD.T70.SinParameters;
        Sin_parameters4=Model.Model.Mat_MaterialD.T90.SinParameters;
    elseif strcmp(Material,'Material E')
        Tri_parameters=[Model2.Model2.Mat_MaterialE.T25.Parameters;
                        Model2.Model2.Mat_MaterialE.T50.Parameters;
                        Model2.Model2.Mat_MaterialE.T70.Parameters;
                        Model2.Model2.Mat_MaterialE.T90.Parameters];
        Sin_parameters1=Model.Model.Mat_MaterialE.T25.SinParameters;
        Sin_parameters2=Model.Model.Mat_MaterialE.T50.SinParameters;
        Sin_parameters3=Model.Model.Mat_MaterialE.T70.SinParameters;
        Sin_parameters4=Model.Model.Mat_MaterialE.T90.SinParameters;
    end
    %==================== Classify and simplify waveforms ======================%
    FOURIER=abs(fft(data_B')');
    d_Flux=diff(data_B')';
    ad_Flux=[mean(d_Flux(:,001:060)')',mean(d_Flux(:,103:163)')',mean(d_Flux(:,205:265)')',mean(d_Flux(:,308:368)')',mean(d_Flux(:,410:470)')',...
             mean(d_Flux(:,512:572)')',mean(d_Flux(:,615:675)')',mean(d_Flux(:,717:777)')',mean(d_Flux(:,820:880)')',mean(d_Flux(:,922:988)')'];
    Flux_=([zeros(size(data_B,1),1) cumsum(ad_Flux,2)]-mean(cumsum(ad_Flux,2),2))*1024/10;
    top_Flux= ad_Flux > max(ad_Flux,[],2)-1/3*(max(ad_Flux,[],2)-min(ad_Flux,[],2));
    bot_Flux= ad_Flux < min(ad_Flux,[],2)+1/3*(max(ad_Flux,[],2)-min(ad_Flux,[],2));
    Shape=ones(size(data_B,1),1);
    Shape(sum(top_Flux+bot_Flux,2)==10)=2;
    Shape(abs(FOURIER(:,2))./sum(abs(FOURIER(:,2:20)),2)>0.95)=3;
    Duty=sum(top_Flux,2);
    Flux_pp =max(data_B ,[],2)-min(data_B ,[],2);
    Flux_pp_=max(Flux_,[],2)-min(Flux_,[],2);
    for n_fixTrap=1:size(Flux_,1)
        if Shape(n_fixTrap)==1
            correction=find(top_Flux(n_fixTrap,:)==0,1,'last');
            if correction~=10
                Flux_(n_fixTrap,1:10)=circshift(Flux_(n_fixTrap,1:10),-correction);
                Flux_(n_fixTrap,1:10)=circshift(Flux_(n_fixTrap,1:10),-correction);
                top_Flux(n_fixTrap,1:10)=circshift(top_Flux(n_fixTrap,1:10),-correction);
                bot_Flux(n_fixTrap,1:10)=circshift(bot_Flux(n_fixTrap,1:10),-correction);
            end
        end
    end
    %% GENERATE DEFINITIONS
    % TRIANGULAR
    SSet = Shape==2;
    Data.Tri.F    = data_F(SSet)';
    Data.Tri.T    = data_T(SSet)';
    Data.Tri.D    = [Duty(SSet)';10-Duty(SSet)']/10;
    Data.Tri.DB   = [1;1].*Flux_pp_(SSet)';
    Data.Tri.dBdt = abs(Data.Tri.DB.*Data.Tri.F./Data.Tri.D);
    Data.Tri.index = SSet;
    % TRAPEZOIDAL
    SSet = Shape==1;
    k_fixTrap=0;
    i_fixTrap=find(Shape==1);
    Data.Tra.F   =data_F    (SSet)';
    Data.Tra.T   =data_T    (SSet)';
    a=top_Flux-bot_Flux;
    for n=1:length(SSet)
        i0A(n)=find(a(n,:)==+1,1,'first');
        i1A(n)=find(a(n,:)==+1,1,'last');
        i0B(n)=find(a(n,:)==-1,1,'first');
        i1B(n)=find(a(n,:)==-1,1,'last');
    
        B0A(n)=Flux_(n,i0A(n));
        B1A(n)=Flux_(n,i1A(n)+1);
        B0B(n)=Flux_(n,i0B(n));
        B1B(n)=Flux_(n,i1B(n)+1);
    end
    Data.Tra.D   = [i1A(SSet)-i0A(SSet)+1;...
                                         (10-(i1A(SSet)-i0A(SSet)+1+i1B(SSet)-i0B(SSet)+1))/2;...
                                         i1B(SSet)-i0B(SSet)+1;...
                                         (10-(i1A(SSet)-i0A(SSet)+1+i1B(SSet)-i0B(SSet)+1))/2]/10;
    Data.Tra.DB  =abs([B1A(SSet)-B0A(SSet);...
                                            B0B(SSet)-B1A(SSet);...
                                            B1B(SSet)-B0B(SSet);...
                                            B0A(SSet)-B1B(SSet)]);
    Data.Tra.dBdt=abs(Data.Tra.DB.*Data.Tra.F./Data.Tra.D);
    Data.Tra.index = SSet;
    % SINUSOIDAL
    SSet=Shape==3;
    Data.Sin.F  = data_F   (SSet)';
    Data.Sin.T  = data_T   (SSet)';
    Data.Sin.DB = Flux_pp(SSet)';
    Data.Sin.index = SSet;
    %% GET SINUSOIDAL LOSSES
    for Temp=[25,50,70,90]
% LOAD DATA FROM TEMP --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP 
        SSet=       Data.Sin.T==Temp;
        DB  =       Data.Sin.DB  (SSet);
        F   =       Data.Sin.F   (SSet);
        if     Temp==25
            Parameters=Sin_parameters1;
        elseif Temp==50
            Parameters=Sin_parameters2;
        elseif Temp==70
            Parameters=Sin_parameters3;
        elseif Temp==90
            Parameters=Sin_parameters4;
        end
% ESTIMATE LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES
        Pe = exp(Parameters(log([F',DB']))');
        Data.Sin.Pe(SSet)=Pe;
    end
    data_Pe(Data.Sin.index)=Data.Sin.Pe;
    %% GET TRIANGULAR LOSSES
    for Temp=[25,50,70,90]
% LOAD DATA FROM TEMP --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP 
        SSet=       Data.Tri.T==Temp;
        dBdt =     Data.Tri.dBdt(:,SSet);
        DB   = max(Data.Tri.DB  (:,SSet));
        D    =     Data.Tri.D   (:,SSet);
        if     Temp==25
            Parameters=Tri_parameters(1,:);
        elseif Temp==50
            Parameters=Tri_parameters(2,:);
        elseif Temp==70
            Parameters=Tri_parameters(3,:);
        elseif Temp==90
            Parameters=Tri_parameters(4,:);
        end
% EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE
        P_tSE   = exp(Parameters(1)+log(dBdt)*Parameters(2)+log(DB)*Parameters(3))+...
                  exp(Parameters(4)+log(dBdt)*Parameters(5)+log(DB)*Parameters(6));
% ESTIMATE LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES
        Pe = sum(P_tSE.*D);
        Data.Tri.Pe(SSet)=Pe;
    end
    data_Pe(Data.Tri.index)=Data.Tri.Pe;
    %% GET TRAPEZOIDAL LOSSES
    for Temp=[25,50,70,90]
% LOAD DATA FROM TEMP --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP  --- LOAD DATA FROM TEMP 
        SSet=       Data.Tra.T==Temp;
        dBdt_Tra=       Data.Tra.dBdt(:,SSet);
        DB_Tra  =   max(Data.Tra.DB  (:,SSet));
        D_Tra   =       Data.Tra.D   (:,SSet);
        F_Tra   =       Data.Tra.F   (:,SSet);
        T_Tra   =D_Tra./F_Tra;
        dBdt = dBdt_Tra;
        DB   =   DB_Tra;
        D    =    D_Tra;
        T    =    T_Tra(2,:);
        if     Temp==25
            Parameters=Tri_parameters(1,:);
        elseif Temp==50
            Parameters=Tri_parameters(2,:);
        elseif Temp==70
            Parameters=Tri_parameters(3,:);
        elseif Temp==90
            Parameters=Tri_parameters(4,:);
        end
% EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE --- EVALUATE ciGSE
        P_tSE   = exp(Parameters(1)+log(dBdt)*Parameters(2)+log(DB)*Parameters(3))+...
                  exp(Parameters(4)+log(dBdt)*Parameters(5)+log(DB)*Parameters(6));
% EVALUATE RELAXATION --- EVALUATE RELAXATION --- EVALUATE RELAXATION --- EVALUATE RELAXATION --- EVALUATE RELAXATION --- EVALUATE RELAXATION --- EVALUATE RELAXATION --- EVALUATE RELAXATION --- EVALUATE RELAXATION
        E_Rel   = exp(Parameters(7)+log(   T)*Parameters(8)+log(DB)*Parameters(9)).*...
                  (T_Tra(1,:)==T_Tra(3,:));
% ESTIMATE LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES --- ESTIMATEA LOSSES
        Pe = sum(P_tSE.*D)+E_Rel.*F_Tra;
        Data.Tra.Pe(SSet)=Pe;
    end
    data_Pe(Data.Tra.index)=Data.Tra.Pe;
    data_Pe=data_Pe';
    % Write output
    writematrix(data_Pe, ['./Testing/Volumetric_Loss_', Material, '.csv']);
    fprintf('Model inference is finished! \n');

end

%% End