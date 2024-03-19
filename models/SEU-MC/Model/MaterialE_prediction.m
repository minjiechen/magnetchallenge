clearvars -except  B_Field Frequency  Temperature;
clc;

%material NAME
Material = 'Material E';

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
            if(Bmax(i,1)>0.039)
                %%%In the original training model, we used Pv/[(\phi)^2/2]=ke*f^2+kh*f,
                %%%Therefore, the format of the formula here and other 4 materials is slightly different from that in the 5-page-report.
                ke=2.3517e-03*Bmax(i,1)^2+1.5644e-03*Bmax(i,1)+3.0919e-05;
                kh=-2011.3*Bmax(i,1)^2+2422.5*Bmax(i,1)-61.069;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=3.4183e-01*Bmax(i,1)^2-2.9237e-02*Bmax(i,1)+7.0561e-04;
                kh=-27850*Bmax(i,1)^2+6850*Bmax(i,1)-190.63;
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
            if(Bmax(i,1)>0.06)
                ke=3.2557e-03*Bmax(i,1)^2+1.5263e-03*Bmax(i,1)+7.6166e-05;
                kh=-2659.9*Bmax(i,1)^2+2735.2*Bmax(i,1)-148.53;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=9.5591e-02*Bmax(i,1)^2-1.0951e-02*Bmax(i,1)+4.9478e-04;
                kh=-2539.3*Bmax(i,1)^2+2683.2*Bmax(i,1)-143.61;
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
            if(Bmax(i,1)>0.06)
                ke=8.1961e-04*Bmax(i,1)^2+2.6633e-03*Bmax(i,1)-2.5126e-05;
                kh=-2485.1*Bmax(i,1)^2+2754.3*Bmax(i,1)-149.74;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=5.3827e-02*Bmax(i,1)^2-7.9523e-03*Bmax(i,1)+4.4851e-04;
                kh=-2485.1*Bmax(i,1)^2+2754.3*Bmax(i,1)-149.74;
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
            if(Bmax(i,1)>0.06)
                ke=-1.1256e-01*Bmax(i,1)^3+6.2093e-02*Bmax(i,1)^2-6.541e-03*Bmax(i,1)+3.185e-04;
                kh=21483*Bmax(i,1)^3-16116*Bmax(i,1)^2+5353.3*Bmax(i,1)-243.43;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=-1.0804e-02*Bmax(i,1)+6.818e-04;
                kh=21483*Bmax(i,1)^3-16116*Bmax(i,1)^2+5353.3*Bmax(i,1)-243.43;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            end
            losssum(i,1)=losssum(i,1)+2*pi*f_fft*(XL+Xc)*Ayy(ii+1)^2/2;
        end
        P(i,1)=losssum(i,1);
    end
end

%%%Export CSV file
writematrix(P,'Volumtric_Loss_Material E.csv')

end



