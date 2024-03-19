clearvars -except  B_Field Frequency  Temperature;
clc;

%material NAME
Material = 'Material B';

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
            if(f_fft<101000&&Bmax(i,1)<0.07)
                %%%In the original training model, we used Pv/[(\phi)^2/2]=ke*f^2+kh*f,
                %%%Therefore, the format of the formula here and other 4 materials is slightly different from that in the 5-page-report.
                ke=0.0118034*Bmax(i,1)+0.000358092;
                kh=-2.1233e+5*Bmax(i,1)^3+2.8996e+4*Bmax(i,1)^2-1.3965e+3*Bmax(i,1)+3.5814e+1;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=2.4894e-3*Bmax(i,1)^2+7.2967e-05*Bmax(i,1)+1.2919e-03;
                kh=-2511.7*Bmax(i,1)^2+1396.2*Bmax(i,1)-90.89;
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
            if(f_fft<90000)
                ke=6.2892e-3*Bmax(i,1)^2+2.4151e-3*Bmax(i,1)+1.0843e-3;
                kh=-2313.5*Bmax(i,1)^2+1177.5*Bmax(i,1)-45.32;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=0.0091045*Bmax(i,1)^2-0.0018646*Bmax(i,1)+0.0016265;
                kh=4903.2*Bmax(i,1)^3-5523*Bmax(i,1)^2+1994.9*Bmax(i,1)-98.412;
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
            if(f_fft<90000)
                ke=2.1665e-02*Bmax(i,1)^2-4.937e-04*Bmax(i,1)+1.5638e-3;
                kh=-4774.2*Bmax(i,1)^2+1758.8*Bmax(i,1)-56;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=1.7566e-02*Bmax(i,1)^2-3.7084e-03*Bmax(i,1)+1.931e-03;
                kh=-4482.4*Bmax(i,1)^3-4823.8*Bmax(i,1)^2+2271.8*Bmax(i,1)-79.522;
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
                ke=2.4099e-02*Bmax(i,1)^2-3.5152e-03*Bmax(i,1)+2.4191e-3;
                kh=-4801.2*Bmax(i,1)^2+1984.9*Bmax(i,1)-71.314;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            else
                ke=3.9132e-02*Bmax(i,1)^2-6.2605e-03*Bmax(i,1)+2.1861e-03;
                kh=50203*Bmax(i,1)^3-24798*Bmax(i,1)^2+4011.3*Bmax(i,1)-52.029;
                XL=ke*f_fft/(2*pi);
                Xc=kh/(2*pi);
            end
            losssum(i,1)=losssum(i,1)+2*pi*f_fft*(XL+Xc)*Ayy(ii+1)^2/2;
        end
        P(i,1)=losssum(i,1);
    end
end

%%%Export CSV file
writematrix(P,'Volumtric_Loss_Material B.csv')
end



