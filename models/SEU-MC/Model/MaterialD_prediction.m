clearvars -except  B_Field Frequency  Temperature;
clc;

%material NAME
Material = 'Material D';

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
                %%%In the original training model, we used Pv/[(\phi)^2/2]=ke*f^2+kh*f,
                %%%Therefore, the format of the formula here and other 4 materials is slightly different from that in the 5-page-report.
                ke=-6.212e-03*Bmax(i,1)^2+3.8341e-03*Bmax(i,1)+8.4071e-05;
                kh=-2649.6*Bmax(i,1)^2+2317.3*Bmax(i,1)-6.2411;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            losssum(i,1)=losssum(i,1)+2*pi*f_fft*(XL+Xc)*Ayy(ii+1)^2/2;
        end            
        P(i,1)=losssum(i,1);
    end
    if T(i)==50                 %%%Temperature distribution processing
        for ii=1:511
            f_fft=freq*ii;
                ke=3.452e-03*Bmax(i,1)+6.6056e-05;
                kh=1372.9*Bmax(i,1)-35.082;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            losssum(i,1)=losssum(i,1)+2*pi*f_fft*(XL+Xc)*Ayy(ii+1)^2/2;
        end
        P(i,1)=losssum(i,1);
    end    
    if T(i)==70                 %%%Temperature distribution processing
        for ii=1:511
            f_fft=freq*ii;
                ke=-2.4808e-02*Bmax(i,1)^3+5.8596e-03*Bmax(i,1)^2+3.0314e-03*Bmax(i,1)+3.6412e-05;
                kh=2616.3*Bmax(i,1)^2+748.57*Bmax(i,1)-32.727;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            losssum(i,1)=losssum(i,1)+2*pi*f_fft*(XL+Xc)*Ayy(ii+1)^2/2;
        end
        P(i,1)=losssum(i,1);
    end
    if T(i)==90                 %%%Temperature distribution processing
        for ii=1:511
            f_fft=freq*ii;
                ke=-3.1179e-02*Bmax(i,1)^3+1.6654e-02*Bmax(i,1)^2-2.1718e-05*Bmax(i,1)+2.8032e-04;
                kh=1892.8*Bmax(i,1)-142.46;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            losssum(i,1)=losssum(i,1)+2*pi*f_fft*(XL+Xc)*Ayy(ii+1)^2/2;
        end
        P(i,1)=losssum(i,1);
    end
end

%%%Export CSV file
writematrix(P,'Volumtric_Loss_Material D.csv')

end



