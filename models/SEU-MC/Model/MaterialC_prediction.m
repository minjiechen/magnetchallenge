clearvars -except  B_Field Frequency  Temperature;
clc;

%material NAME
Material = 'Material C';

% Begin Loading Data:
%Load B_Field Frequency Temperature

B_Field = load(['.\',Material,'\B_Field.csv']);
Frequency= load(['.\',Material,'\Frequency.csv']);
Temperature= load(['.\',Material,'\Temperature.csv']);

%main program
[P]=GetVolumticloss(B_Field,Temperature,Frequency);

%Function for calculating Volumtric_Loss
function [P]=GetVolumticloss(B,T,f)

datalength=length(f);      %get data length

%%%initialization
losssum=zeros(datalength,1);
Bmax=zeros(datalength,1);
P=zeros(datalength,1);
%%%


for i=1:datalength              %%%Process each set of data
    B_fft=fft(B(i,:),1024);     %%%FFT
    freq=f(i);                  %%%Obtain fundamental frequency
    Ayy=(abs(B_fft))/(1024/2);  %%%Obtain the magnetic density amplitudes of each harmonic after FFT
    Bmax(i,1)=(max(B(i,:))-min(B(i,:)))/2;   %%%Calculate the peak magnetic density, where Bmax=Bpeak in the article
    if T(i)==25                 %%%Temperature distribution processing
        for ii=1:511            %%%1024 data points correspond to 511th harmonic
            f_fft=freq*ii;      %%%The frequency of each harmonic
            if(f_fft<200000)
                %%%In the original training model, we used Pv/[(\phi)^2/2]=ke*f^2+kh*f,
                %%%Therefore, the format of the formula here and other 4 materials is slightly different from that in the 5-page-report.
                ke=2.8051E-02*Bmax(i,1)^3-1.8173e-02*Bmax(i,1)^2+6.4727e-03*Bmax(i,1)+2.6667e-04;
                kh=3626.7*Bmax(i,1)^3-2646*Bmax(i,1)^2+780.94*Bmax(i,1)-10.247;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=-1.2495e-02*Bmax(i,1)^2+5.8113e-03*Bmax(i,1)+8.1692e-04;
                kh=-356.47*Bmax(i,1)^2+614.95*Bmax(i,1)-119.39;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            end
            losssum(i,1)=losssum(i,1)+2*pi*f_fft*(XL+Xc)*Ayy(ii+1)^2/2;
        end            
        P(i,1)=losssum(i,1);
    end
    if T(i)==50                 %%%Temperature distribution processing
        for ii=1:511
            f_fft=freq*ii;
            if(f_fft<200000)
                ke=3.7005e-02*Bmax(i,1)^3-2.1741E-02*Bmax(i,1)^2+5.4852e-03*Bmax(i,1)+2.7931e-04;
                kh=-670.98*Bmax(i,1)^2+659.31*Bmax(i,1)-26.211;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=-1.0471e-01*Bmax(i,1)^3+2.3638e-04*Bmax(i,1)^2+1.5964e-03*Bmax(i,1)+8.2413e-04;
                kh=67142*Bmax(i,1)^3-14792*Bmax(i,1)^2+2143.2*Bmax(i,1)-149.56;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            end
            losssum(i,1)=losssum(i,1)+2*pi*f_fft*(XL+Xc)*Ayy(ii+1)^2/2;
        end
        P(i,1)=losssum(i,1);
    end    
    if T(i)==70                 %%%Temperature distribution processing
        for ii=1:511
            f_fft=freq*ii;
            if(f_fft<170000)
                ke=-4.1342e-03*Bmax(i,1)^2+4.5648E-03*Bmax(i,1)+2e-04;
                kh=-3109.8*Bmax(i,1)^3+1496.2*Bmax(i,1)^2+145.59*Bmax(i,1)-3.034;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=-6.8645e-02*Bmax(i,1)^3+3.6833e-03*Bmax(i,1)^2+6.0284e-03*Bmax(i,1)+8.5639e-04;
                kh=-86300*Bmax(i,1)^4+66730*Bmax(i,1)^3-12584*Bmax(i,1)^2+972.27*Bmax(i,1)-135.86;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
           end
            losssum(i,1)=losssum(i,1)+2*pi*f_fft*(XL+Xc)*Ayy(ii+1)^2/2;
        end
        P(i,1)=losssum(i,1);
    end
    if T(i)==90                 %%%Temperature distribution processing
        for ii=1:511
            f_fft=freq*ii;
            if(f_fft<170000)
                ke=-8.0681e-03*Bmax(i,1)^2+6.0642E-03*Bmax(i,1)+1.9005e-04;
                kh=499.39*Bmax(i,1)^2+194.73*Bmax(i,1)-9.3965;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=-5.9616e-03*Bmax(i,1)^2+5.945e-03*Bmax(i,1)+1.0021e-03;
                kh=589.26*Bmax(i,1)^2+164.34*Bmax(i,1)-143.69;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            end
            losssum(i,1)=losssum(i,1)+2*pi*f_fft*(XL+Xc)*Ayy(ii+1)^2/2;
        end
        P(i,1)=losssum(i,1);
    end
end

%%%Export CSV file
writematrix(P,'Volumtric_Loss_Material C.csv')

end



