clearvars -except  B_Field Frequency  Temperature;
clc;

%material NAME
Material = 'Material A';

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
            if(f_fft<120000)
                %%%In the original training model, we used Pv/[(\phi)^2/2]=ke*f^2+kh*f,
                %%%Therefore, the format of the formula here and other 4 materials is slightly different from that in the 5-page-report.
                ke=2.6037*Bmax(i,1)^4-1.9102*Bmax(i,1)^3+4.3333e-01*Bmax(i,1)^2-2.1326e-02*Bmax(i,1)+1.7003e-03;
                kh=3.6582e+6*Bmax(i,1)^5-3.3762e+6*Bmax(i,1)^4+1.1885e+6*Bmax(i,1)^3-1.9626e+5*Bmax(i,1)^2+1.462e+4*Bmax(i,1)-1.8073e+2;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=6.8579e-01*Bmax(i,1)^3-3.7617e-01*Bmax(i,1)^2+7.1497e-02*Bmax(i,1)+1.5723e-03;
                kh=125114*Bmax(i,1)^3-19637*Bmax(i,1)^2-285.08*Bmax(i,1)-229.63;
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
            if(f_fft<160000)
                ke=-3.0765e-02*Bmax(i,1)^2+1.9978e-02*Bmax(i,1)+5.3520e-04;
                kh=-748.48*Bmax(i,1)^2+561.4*Bmax(i,1)+23.711;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=-2.5756e-01*Bmax(i,1)^2+8.3892e-02*Bmax(i,1)+1.2657e-03;
                kh=38001*Bmax(i,1)^2-9210.2*Bmax(i,1)-119.01;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            end
            if(f_fft>600000)
                f_fft=freq*(ii+1);
                ke=-2.5756e-01*Bmax(i,1)^2+8.3892e-02*Bmax(i,1)+1.2657e-03;
                kh=38001*Bmax(i,1)^2-9210.2*Bmax(i,1)-119.01;
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
            if(f_fft<200000)
                ke=-1.7562e-02*Bmax(i,1)^2+1.3161e-02*Bmax(i,1)+1.0816e-03;
                kh=29180*Bmax(i,1)^3-13630*Bmax(i,1)^2+2125.6*Bmax(i,1)-50.963;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=-5.0118e+1*Bmax(i,1)^5+4.4283e+1*Bmax(i,1)^4-1.2993e+1*Bmax(i,1)^3+1.2823*Bmax(i,1)^2+1.4283e-2*Bmax(i,1)+2.3239e-3;
                kh=1.4865e+7*Bmax(i,1)^5-1.2073e+7*Bmax(i,1)^4+3.305e+6*Bmax(i,1)^3-3.2266e+5*Bmax(i,1)^2+4.2771e+3*Bmax(i,1)-2.9764e+2;
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
            if(f_fft<110000)
                ke=-2.4406e-02*Bmax(i,1)^2+1.8087e-02*Bmax(i,1)+1.2937e-03;
                kh=738.23*Bmax(i,1)^2-57.568*Bmax(i,1)-49.825;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=321.29*Bmax(i,1)^6-407.87*Bmax(i,1)^5+185.67*Bmax(i,1)^4-36.662*Bmax(i,1)^3+2.7996*Bmax(i,1)^2-6.6667e-03*Bmax(i,1)+2.5782e-03;
                kh=-1.7015e+8*Bmax(i,1)^6+1.7123e+8*Bmax(i,1)^5-6.4608e+7*Bmax(i,1)^4+1.105e+7*Bmax(i,1)^3-7.8728e+5*Bmax(i,1)^2+1.2618e+4*Bmax(i,1)-3.5447e+2;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            end
            losssum(i,1)=losssum(i,1)+2*pi*f_fft*(XL+Xc)*Ayy(ii+1)^2/2;
        end
        P(i,1)=losssum(i,1);
    end
end

%%%Export CSV file
writematrix(P,'Volumtric_Loss_Material A.csv')
end



