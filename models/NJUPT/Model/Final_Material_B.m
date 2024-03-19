function [Pred_Material_B] = Final_MaterialB(B_Field,Frequency,Temperature)
%Read the B_waveform for Material B
  trwavedata=readmatrix('B:\NJUPT（Final）\Model\Training\Material B\B_Field.csv');
  N=size(trwavedata,1);
  trB_predict_sum=[];
  trnumslope_sum=[];
  trwaveform_sum=[];
  for y=1:N % The waveform corresponding to the classification flux density B data series
  treverywave=trwavedata(y,:);
  trB_predict(1,:)=max(treverywave);% The maximum value of the flux density data series is the flux density of the test point
  trB_predict_sum(y,:)= trB_predict(1,:);
    if trwavedata(y,18)>0% First isolate the sine wave
    trnumslope=3;
    trwaveform=1; 
    tr_D=0;
    tr_D1=0;
    tr_D3=0;
   else % Separate rectangular wave and pulse wave
    tr_fdeverywave=100*trwavedata(y,:);
    [TF,S1] = ischange(tr_fdeverywave,'linear','Threshold',10);
    trslopetable=tabulate(S1);
     if min(trslopetable(:,2))<80
      [lr,lc]=min(trslopetable(:,2));
      trdyslope=trslopetable(lc,1);
      S1(find(S1==trdyslope))=[];
      trslopetable(lc,:)=[];
     end 
     [C,ia,ic] = unique(S1);
     trnumslope=size(trslopetable,1);
     if trnumslope==4
      trwaveform=3;    
      trslopeindex=sort(ia);
      tr_D1=round(trslopeindex(2,1)/1000,1);
      tr_D3=round((trslopeindex(4,1)-trslopeindex(3,1))/1000,1);
      tr_D=0;
     else
      [TF,S1] = ischange(tr_fdeverywave,'linear','Threshold',50);
    trslopetable=tabulate(S1);
     if min(trslopetable(:,2))<80
      [lr,lc]=min(trslopetable(:,2));
      trdyslope=trslopetable(lc,1);
      S1(find(S1==trdyslope))=[];
      trslopetable(lc,:)=[];
     end 
     [C,ia,ic] = unique(S1);
     trnumslope=size(trslopetable,1);
      if trnumslope==4
      trwaveform=3;    
      trslopeindex=sort(ia);
      tr_D1=round(trslopeindex(2,1)/1000,1);
      tr_D3=round((trslopeindex(4,1)-trslopeindex(3,1))/1000,1);
      tr_D=0;
      elseif trnumslope==2&ia(1,1)>round(ia(1,1)/100)*100-22&ia(1,1)<round(ia(1,1)/100)*100+22
      trwaveform=2;
      tr_D=round(max(ia)/1000,1);
      tr_D1=0;
      tr_D3=0;
      else
      [TF,S1] = ischange(tr_fdeverywave,'linear','Threshold',0.8);
    trslopetable=tabulate(S1);
     if min(trslopetable(:,2))<80
      [lr,lc]=min(trslopetable(:,2));
      trdyslope=trslopetable(lc,1);
      S1(find(S1==trdyslope))=[];
      trslopetable(lc,:)=[];
     end 
     [C,ia,ic] = unique(S1);
     trnumslope=size(trslopetable,1);
       if trnumslope==4
      trwaveform=3;    
      trslopeindex=sort(ia);
      tr_D1=round(trslopeindex(2,1)/1000,1);
      tr_D3=round((trslopeindex(4,1)-trslopeindex(3,1))/1000,1);
      tr_D=0;
      else trnumslope==3
           trwaveform=2;    
           trslopeindex=sort(ia);
           tr_D=1-round((trslopeindex(3,1)-trslopeindex(2,1))/1000,1);
           tr_D1=0;  
           tr_D3=0;
     end
   end
 end
end
 trnumslope_every(1,:)=trnumslope;
 trnumslope_sum(y,:)=trnumslope_every(1,:);
 trwaveform_every(1,1)=trwaveform;
 trwaveform_every(1,2)=tr_D;
 trwaveform_every(1,3)=tr_D1;
 trwaveform_every(1,4)=tr_D3;
 trwaveform_sum(y,1)= trwaveform_every(1,1);
 trwaveform_sum(y,2)= trwaveform_every(1,2);
 trwaveform_sum(y,3)= trwaveform_every(1,3);
 trwaveform_sum(y,4)= trwaveform_every(1,4);
end 

tr_Tdata=readmatrix('B:\NJUPT（Final）\Model\Training\Material B\Temperature.csv'); 
tr_Tsumdata=tr_Tdata;
tr_Fdata=readmatrix('B:\NJUPT（Final）\Model\Training\Material B\Frequency.csv'); 
tr_Fsumdata=tr_Fdata;
tr_realloss_data=readmatrix('B:\NJUPT（Final）\Model\Training\Material B\Volumetric_Loss.csv'); 
tr_realloss_sumdata=tr_realloss_data;
index2_sin25=find(trwaveform_sum(:,1)==1&tr_Tsumdata(:,1)==25);
MBsin25=[tr_Fsumdata(index2_sin25,:),trB_predict_sum(index2_sin25,:),tr_realloss_sumdata(index2_sin25,:)];
index2_sin50=find(trwaveform_sum(:,1)==1&tr_Tsumdata(:,1)==50);
MBsin50=[tr_Fsumdata(index2_sin50,:),trB_predict_sum(index2_sin50,:),tr_realloss_sumdata(index2_sin50,:)];
index2_sin70=find(trwaveform_sum(:,1)==1&tr_Tsumdata(:,1)==70);
MBsin70=[tr_Fsumdata(index2_sin70,:),trB_predict_sum(index2_sin70,:),tr_realloss_sumdata(index2_sin70,:)];
index2_sin90=find(trwaveform_sum(:,1)==1&tr_Tsumdata(:,1)==90);
MBsin90=[tr_Fsumdata(index2_sin90,:),trB_predict_sum(index2_sin90,:),tr_realloss_sumdata(index2_sin90,:)];

index2_rect25_01=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.1&tr_Tsumdata(:,1)==25);
MBrect25_01=[tr_Fsumdata(index2_rect25_01,:),trB_predict_sum(index2_rect25_01,:),tr_realloss_sumdata(index2_rect25_01,:)];
index2_rect25_02=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.2&tr_Tsumdata(:,1)==25);
MBrect25_02=[tr_Fsumdata(index2_rect25_02,:),trB_predict_sum(index2_rect25_02,:),tr_realloss_sumdata(index2_rect25_02,:)];
index2_rect25_03=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.3&tr_Tsumdata(:,1)==25);
MBrect25_03=[tr_Fsumdata(index2_rect25_03,:),trB_predict_sum(index2_rect25_03,:),tr_realloss_sumdata(index2_rect25_03,:)];
index2_rect25_04=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.4&tr_Tsumdata(:,1)==25);
MBrect25_04=[tr_Fsumdata(index2_rect25_04,:),trB_predict_sum(index2_rect25_04,:),tr_realloss_sumdata(index2_rect25_04,:)];
index2_rect25_05=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.5&tr_Tsumdata(:,1)==25);
MBrect25_05=[tr_Fsumdata(index2_rect25_05,:),trB_predict_sum(index2_rect25_05,:),tr_realloss_sumdata(index2_rect25_05,:)];
index2_rect25_06=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.6&tr_Tsumdata(:,1)==25);
MBrect25_06=[tr_Fsumdata(index2_rect25_06,:),trB_predict_sum(index2_rect25_06,:),tr_realloss_sumdata(index2_rect25_06,:)];
index2_rect25_07=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.7&tr_Tsumdata(:,1)==25);
MBrect25_07=[tr_Fsumdata(index2_rect25_07,:),trB_predict_sum(index2_rect25_07,:),tr_realloss_sumdata(index2_rect25_07,:)];
index2_rect25_08=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.8&tr_Tsumdata(:,1)==25);
MBrect25_08=[tr_Fsumdata(index2_rect25_08,:),trB_predict_sum(index2_rect25_08,:),tr_realloss_sumdata(index2_rect25_08,:)];
index2_rect25_09=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.9&tr_Tsumdata(:,1)==25);
MBrect25_09=[tr_Fsumdata(index2_rect25_09,:),trB_predict_sum(index2_rect25_09,:),tr_realloss_sumdata(index2_rect25_09,:)];

index2_rect50_01=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.1&tr_Tsumdata(:,1)==50);
MBrect50_01=[tr_Fsumdata(index2_rect50_01,:),trB_predict_sum(index2_rect50_01,:),tr_realloss_sumdata(index2_rect50_01,:)];
index2_rect50_02=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.2&tr_Tsumdata(:,1)==50);
MBrect50_02=[tr_Fsumdata(index2_rect50_02,:),trB_predict_sum(index2_rect50_02,:),tr_realloss_sumdata(index2_rect50_02,:)];
index2_rect50_03=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.3&tr_Tsumdata(:,1)==50);
MBrect50_03=[tr_Fsumdata(index2_rect50_03,:),trB_predict_sum(index2_rect50_03,:),tr_realloss_sumdata(index2_rect50_03,:)];
index2_rect50_04=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.4&tr_Tsumdata(:,1)==50);
MBrect50_04=[tr_Fsumdata(index2_rect50_04,:),trB_predict_sum(index2_rect50_04,:),tr_realloss_sumdata(index2_rect50_04,:)];
index2_rect50_05=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.5&tr_Tsumdata(:,1)==50);
MBrect50_05=[tr_Fsumdata(index2_rect50_05,:),trB_predict_sum(index2_rect50_05,:),tr_realloss_sumdata(index2_rect50_05,:)];
index2_rect50_06=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.6&tr_Tsumdata(:,1)==50);
MBrect50_06=[tr_Fsumdata(index2_rect50_06,:),trB_predict_sum(index2_rect50_06,:),tr_realloss_sumdata(index2_rect50_06,:)];
index2_rect50_07=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.7&tr_Tsumdata(:,1)==50);
MBrect50_07=[tr_Fsumdata(index2_rect50_07,:),trB_predict_sum(index2_rect50_07,:),tr_realloss_sumdata(index2_rect50_07,:)];
index2_rect50_08=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.8&tr_Tsumdata(:,1)==50);
MBrect50_08=[tr_Fsumdata(index2_rect50_08,:),trB_predict_sum(index2_rect50_08,:),tr_realloss_sumdata(index2_rect50_08,:)];
index2_rect50_09=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.9&tr_Tsumdata(:,1)==50);
MBrect50_09=[tr_Fsumdata(index2_rect50_09,:),trB_predict_sum(index2_rect50_09,:),tr_realloss_sumdata(index2_rect50_09,:)];

index2_rect70_01=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.1&tr_Tsumdata(:,1)==70);
MBrect70_01=[tr_Fsumdata(index2_rect70_01,:),trB_predict_sum(index2_rect70_01,:),tr_realloss_sumdata(index2_rect70_01,:)];
index2_rect70_02=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.2&tr_Tsumdata(:,1)==70);
MBrect70_02=[tr_Fsumdata(index2_rect70_02,:),trB_predict_sum(index2_rect70_02,:),tr_realloss_sumdata(index2_rect70_02,:)];
index2_rect70_03=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.3&tr_Tsumdata(:,1)==70);
MBrect70_03=[tr_Fsumdata(index2_rect70_03,:),trB_predict_sum(index2_rect70_03,:),tr_realloss_sumdata(index2_rect70_03,:)];
index2_rect70_04=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.4&tr_Tsumdata(:,1)==70);
MBrect70_04=[tr_Fsumdata(index2_rect70_04,:),trB_predict_sum(index2_rect70_04,:),tr_realloss_sumdata(index2_rect70_04,:)];
index2_rect70_05=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.5&tr_Tsumdata(:,1)==70);
MBrect70_05=[tr_Fsumdata(index2_rect70_05,:),trB_predict_sum(index2_rect70_05,:),tr_realloss_sumdata(index2_rect70_05,:)];
index2_rect70_06=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.6&tr_Tsumdata(:,1)==70);
MBrect70_06=[tr_Fsumdata(index2_rect70_06,:),trB_predict_sum(index2_rect70_06,:),tr_realloss_sumdata(index2_rect70_06,:)];
index2_rect70_07=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.7&tr_Tsumdata(:,1)==70);
MBrect70_07=[tr_Fsumdata(index2_rect70_07,:),trB_predict_sum(index2_rect70_07,:),tr_realloss_sumdata(index2_rect70_07,:)];
index2_rect70_08=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.8&tr_Tsumdata(:,1)==70);
MBrect70_08=[tr_Fsumdata(index2_rect70_08,:),trB_predict_sum(index2_rect70_08,:),tr_realloss_sumdata(index2_rect70_08,:)];
index2_rect70_09=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.9&tr_Tsumdata(:,1)==70);
MBrect70_09=[tr_Fsumdata(index2_rect70_09,:),trB_predict_sum(index2_rect70_09,:),tr_realloss_sumdata(index2_rect70_09,:)];

index2_rect90_01=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.1&tr_Tsumdata(:,1)==90);
MBrect90_01=[tr_Fsumdata(index2_rect90_01,:),trB_predict_sum(index2_rect90_01,:),tr_realloss_sumdata(index2_rect90_01,:)];
index2_rect90_02=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.2&tr_Tsumdata(:,1)==90);
MBrect90_02=[tr_Fsumdata(index2_rect90_02,:),trB_predict_sum(index2_rect90_02,:),tr_realloss_sumdata(index2_rect90_02,:)];
index2_rect90_03=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.3&tr_Tsumdata(:,1)==90);
MBrect90_03=[tr_Fsumdata(index2_rect90_03,:),trB_predict_sum(index2_rect90_03,:),tr_realloss_sumdata(index2_rect90_03,:)];
index2_rect90_04=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.4&tr_Tsumdata(:,1)==90);
MBrect90_04=[tr_Fsumdata(index2_rect90_04,:),trB_predict_sum(index2_rect90_04,:),tr_realloss_sumdata(index2_rect90_04,:)];
index2_rect90_05=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.5&tr_Tsumdata(:,1)==90);
MBrect90_05=[tr_Fsumdata(index2_rect90_05,:),trB_predict_sum(index2_rect90_05,:),tr_realloss_sumdata(index2_rect90_05,:)];
index2_rect90_06=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.6&tr_Tsumdata(:,1)==90);
MBrect90_06=[tr_Fsumdata(index2_rect90_06,:),trB_predict_sum(index2_rect90_06,:),tr_realloss_sumdata(index2_rect90_06,:)];
index2_rect90_07=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.7&tr_Tsumdata(:,1)==90);
MBrect90_07=[tr_Fsumdata(index2_rect90_07,:),trB_predict_sum(index2_rect90_07,:),tr_realloss_sumdata(index2_rect90_07,:)];
index2_rect90_08=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.8&tr_Tsumdata(:,1)==90);
MBrect90_08=[tr_Fsumdata(index2_rect90_08,:),trB_predict_sum(index2_rect90_08,:),tr_realloss_sumdata(index2_rect90_08,:)];
index2_rect90_09=find(trwaveform_sum(:,1)==2&trwaveform_sum(:,2)==0.9&tr_Tsumdata(:,1)==90);
MBrect90_09=[tr_Fsumdata(index2_rect90_09,:),trB_predict_sum(index2_rect90_09,:),tr_realloss_sumdata(index2_rect90_09,:)];

index2_pul25_01_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==25);
MBpul25_01_01=[tr_Fsumdata(index2_pul25_01_01,:),trB_predict_sum(index2_pul25_01_01,:),tr_realloss_sumdata(index2_pul25_01_01,:)];
index2_pul25_01_03=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.3&tr_Tsumdata(:,1)==25);
MBpul25_01_03=[tr_Fsumdata(index2_pul25_01_03,:),trB_predict_sum(index2_pul25_01_03,:),tr_realloss_sumdata(index2_pul25_01_03,:)];
index2_pul25_01_05=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.5&tr_Tsumdata(:,1)==25);
MBpul25_01_05=[tr_Fsumdata(index2_pul25_01_05,:),trB_predict_sum(index2_pul25_01_05,:),tr_realloss_sumdata(index2_pul25_01_05,:)];
index2_pul25_01_07=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.7&tr_Tsumdata(:,1)==25);
MBpul25_01_07=[tr_Fsumdata(index2_pul25_01_07,:),trB_predict_sum(index2_pul25_01_07,:),tr_realloss_sumdata(index2_pul25_01_07,:)];
index2_pul25_02_02=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.2&trwaveform_sum(:,4)==0.2&tr_Tsumdata(:,1)==25);
MBpul25_02_02=[tr_Fsumdata(index2_pul25_02_02,:),trB_predict_sum(index2_pul25_02_02,:),tr_realloss_sumdata(index2_pul25_02_02,:)];
index2_pul25_02_04=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.2&trwaveform_sum(:,4)==0.4&tr_Tsumdata(:,1)==25);
MBpul25_02_04=[tr_Fsumdata(index2_pul25_02_04,:),trB_predict_sum(index2_pul25_02_04,:),tr_realloss_sumdata(index2_pul25_02_04,:)];
index2_pul25_02_06=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.2&trwaveform_sum(:,4)==0.6&tr_Tsumdata(:,1)==25);
MBpul25_02_06=[tr_Fsumdata(index2_pul25_02_06,:),trB_predict_sum(index2_pul25_02_06,:),tr_realloss_sumdata(index2_pul25_02_06,:)];
index2_pul25_03_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.3&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==25);
MBpul25_03_01=[tr_Fsumdata(index2_pul25_03_01,:),trB_predict_sum(index2_pul25_03_01,:),tr_realloss_sumdata(index2_pul25_03_01,:)];
index2_pul25_03_03=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.3&trwaveform_sum(:,4)==0.3&tr_Tsumdata(:,1)==25);
MBpul25_03_03=[tr_Fsumdata(index2_pul25_03_03,:),trB_predict_sum(index2_pul25_03_03,:),tr_realloss_sumdata(index2_pul25_03_03,:)];
index2_pul25_03_05=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.3&trwaveform_sum(:,4)==0.5&tr_Tsumdata(:,1)==25);
MBpul25_03_05=[tr_Fsumdata(index2_pul25_03_05,:),trB_predict_sum(index2_pul25_03_05,:),tr_realloss_sumdata(index2_pul25_03_05,:)];
index2_pul25_04_02=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.4&trwaveform_sum(:,4)==0.2&tr_Tsumdata(:,1)==25);
MBpul25_04_02=[tr_Fsumdata(index2_pul25_04_02,:),trB_predict_sum(index2_pul25_04_02,:),tr_realloss_sumdata(index2_pul25_04_02,:)];
index2_pul25_04_04=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.4&trwaveform_sum(:,4)==0.4&tr_Tsumdata(:,1)==25);
MBpul25_04_04=[tr_Fsumdata(index2_pul25_04_04,:),trB_predict_sum(index2_pul25_04_04,:),tr_realloss_sumdata(index2_pul25_04_04,:)];
index2_pul25_05_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.5&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==25);
MBpul25_05_01=[tr_Fsumdata(index2_pul25_05_01,:),trB_predict_sum(index2_pul25_05_01,:),tr_realloss_sumdata(index2_pul25_05_01,:)];
index2_pul25_05_03=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.5&trwaveform_sum(:,4)==0.3&tr_Tsumdata(:,1)==25);
MBpul25_05_03=[tr_Fsumdata(index2_pul25_05_03,:),trB_predict_sum(index2_pul25_05_03,:),tr_realloss_sumdata(index2_pul25_05_03,:)];
index2_pul25_06_02=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.6&trwaveform_sum(:,4)==0.2&tr_Tsumdata(:,1)==25);
MBpul25_06_02=[tr_Fsumdata(index2_pul25_06_02,:),trB_predict_sum(index2_pul25_06_02,:),tr_realloss_sumdata(index2_pul25_06_02,:)];
index2_pul25_07_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.7&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==25);
MBpul25_07_01=[tr_Fsumdata(index2_pul25_07_01,:),trB_predict_sum(index2_pul25_07_01,:),tr_realloss_sumdata(index2_pul25_07_01,:)];

index2_pul50_01_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==50);
MBpul50_01_01=[tr_Fsumdata(index2_pul50_01_01,:),trB_predict_sum(index2_pul50_01_01,:),tr_realloss_sumdata(index2_pul50_01_01,:)];
index2_pul50_01_03=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.3&tr_Tsumdata(:,1)==50);
MBpul50_01_03=[tr_Fsumdata(index2_pul50_01_03,:),trB_predict_sum(index2_pul50_01_03,:),tr_realloss_sumdata(index2_pul50_01_03,:)];
index2_pul50_01_05=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.5&tr_Tsumdata(:,1)==50);
MBpul50_01_05=[tr_Fsumdata(index2_pul50_01_05,:),trB_predict_sum(index2_pul50_01_05,:),tr_realloss_sumdata(index2_pul50_01_05,:)];
index2_pul50_01_07=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.7&tr_Tsumdata(:,1)==50);
MBpul50_01_07=[tr_Fsumdata(index2_pul50_01_07,:),trB_predict_sum(index2_pul50_01_07,:),tr_realloss_sumdata(index2_pul50_01_07,:)];
index2_pul50_02_02=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.2&trwaveform_sum(:,4)==0.2&tr_Tsumdata(:,1)==50);
MBpul50_02_02=[tr_Fsumdata(index2_pul50_02_02,:),trB_predict_sum(index2_pul50_02_02,:),tr_realloss_sumdata(index2_pul50_02_02,:)];
index2_pul50_02_04=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.2&trwaveform_sum(:,4)==0.4&tr_Tsumdata(:,1)==50);
MBpul50_02_04=[tr_Fsumdata(index2_pul50_02_04,:),trB_predict_sum(index2_pul50_02_04,:),tr_realloss_sumdata(index2_pul50_02_04,:)];
index2_pul50_02_06=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.2&trwaveform_sum(:,4)==0.6&tr_Tsumdata(:,1)==50);
MBpul50_02_06=[tr_Fsumdata(index2_pul50_02_06,:),trB_predict_sum(index2_pul50_02_06,:),tr_realloss_sumdata(index2_pul50_02_06,:)];
index2_pul50_03_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.3&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==50);
MBpul50_03_01=[tr_Fsumdata(index2_pul50_03_01,:),trB_predict_sum(index2_pul50_03_01,:),tr_realloss_sumdata(index2_pul50_03_01,:)];
index2_pul50_03_03=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.3&trwaveform_sum(:,4)==0.3&tr_Tsumdata(:,1)==50);
MBpul50_03_03=[tr_Fsumdata(index2_pul50_03_03,:),trB_predict_sum(index2_pul50_03_03,:),tr_realloss_sumdata(index2_pul50_03_03,:)];
index2_pul50_03_05=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.3&trwaveform_sum(:,4)==0.5&tr_Tsumdata(:,1)==50);
MBpul50_03_05=[tr_Fsumdata(index2_pul50_03_05,:),trB_predict_sum(index2_pul50_03_05,:),tr_realloss_sumdata(index2_pul50_03_05,:)];
index2_pul50_04_02=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.4&trwaveform_sum(:,4)==0.2&tr_Tsumdata(:,1)==50);
MBpul50_04_02=[tr_Fsumdata(index2_pul50_04_02,:),trB_predict_sum(index2_pul50_04_02,:),tr_realloss_sumdata(index2_pul50_04_02,:)];
index2_pul50_04_04=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.4&trwaveform_sum(:,4)==0.4&tr_Tsumdata(:,1)==50);
MBpul50_04_04=[tr_Fsumdata(index2_pul50_04_04,:),trB_predict_sum(index2_pul50_04_04,:),tr_realloss_sumdata(index2_pul50_04_04,:)];
index2_pul50_05_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.5&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==50);
MBpul50_05_01=[tr_Fsumdata(index2_pul50_05_01,:),trB_predict_sum(index2_pul50_05_01,:),tr_realloss_sumdata(index2_pul50_05_01,:)];
index2_pul50_05_03=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.5&trwaveform_sum(:,4)==0.3&tr_Tsumdata(:,1)==50);
MBpul50_05_03=[tr_Fsumdata(index2_pul50_05_03,:),trB_predict_sum(index2_pul50_05_03,:),tr_realloss_sumdata(index2_pul50_05_03,:)];
index2_pul50_06_02=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.6&trwaveform_sum(:,4)==0.2&tr_Tsumdata(:,1)==50);
MBpul50_06_02=[tr_Fsumdata(index2_pul50_06_02,:),trB_predict_sum(index2_pul50_06_02,:),tr_realloss_sumdata(index2_pul50_06_02,:)];
index2_pul50_07_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.7&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==50);
MBpul50_07_01=[tr_Fsumdata(index2_pul50_07_01,:),trB_predict_sum(index2_pul50_07_01,:),tr_realloss_sumdata(index2_pul50_07_01,:)];

index2_pul70_01_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==70);
MBpul70_01_01=[tr_Fsumdata(index2_pul70_01_01,:),trB_predict_sum(index2_pul70_01_01,:),tr_realloss_sumdata(index2_pul70_01_01,:)];
index2_pul70_01_03=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.3&tr_Tsumdata(:,1)==70);
MBpul70_01_03=[tr_Fsumdata(index2_pul70_01_03,:),trB_predict_sum(index2_pul70_01_03,:),tr_realloss_sumdata(index2_pul70_01_03,:)];
index2_pul70_01_05=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.5&tr_Tsumdata(:,1)==70);
MBpul70_01_05=[tr_Fsumdata(index2_pul70_01_05,:),trB_predict_sum(index2_pul70_01_05,:),tr_realloss_sumdata(index2_pul70_01_05,:)];
index2_pul70_01_07=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.7&tr_Tsumdata(:,1)==70);
MBpul70_01_07=[tr_Fsumdata(index2_pul70_01_07,:),trB_predict_sum(index2_pul70_01_07,:),tr_realloss_sumdata(index2_pul70_01_07,:)];
index2_pul70_02_02=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.2&trwaveform_sum(:,4)==0.2&tr_Tsumdata(:,1)==70);
MBpul70_02_02=[tr_Fsumdata(index2_pul70_02_02,:),trB_predict_sum(index2_pul70_02_02,:),tr_realloss_sumdata(index2_pul70_02_02,:)];
index2_pul70_02_04=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.2&trwaveform_sum(:,4)==0.4&tr_Tsumdata(:,1)==70);
MBpul70_02_04=[tr_Fsumdata(index2_pul70_02_04,:),trB_predict_sum(index2_pul70_02_04,:),tr_realloss_sumdata(index2_pul70_02_04,:)];
index2_pul70_02_06=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.2&trwaveform_sum(:,4)==0.6&tr_Tsumdata(:,1)==70);
MBpul70_02_06=[tr_Fsumdata(index2_pul70_02_06,:),trB_predict_sum(index2_pul70_02_06,:),tr_realloss_sumdata(index2_pul70_02_06,:)];
index2_pul70_03_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.3&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==70);
MBpul70_03_01=[tr_Fsumdata(index2_pul70_03_01,:),trB_predict_sum(index2_pul70_03_01,:),tr_realloss_sumdata(index2_pul70_03_01,:)];
index2_pul70_03_03=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.3&trwaveform_sum(:,4)==0.3&tr_Tsumdata(:,1)==70);
MBpul70_03_03=[tr_Fsumdata(index2_pul70_03_03,:),trB_predict_sum(index2_pul70_03_03,:),tr_realloss_sumdata(index2_pul70_03_03,:)];
index2_pul70_03_05=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.3&trwaveform_sum(:,4)==0.5&tr_Tsumdata(:,1)==70);
MBpul70_03_05=[tr_Fsumdata(index2_pul70_03_05,:),trB_predict_sum(index2_pul70_03_05,:),tr_realloss_sumdata(index2_pul70_03_05,:)];
index2_pul70_04_02=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.4&trwaveform_sum(:,4)==0.2&tr_Tsumdata(:,1)==70);
MBpul70_04_02=[tr_Fsumdata(index2_pul70_04_02,:),trB_predict_sum(index2_pul70_04_02,:),tr_realloss_sumdata(index2_pul70_04_02,:)];
index2_pul70_04_04=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.4&trwaveform_sum(:,4)==0.4&tr_Tsumdata(:,1)==70);
MBpul70_04_04=[tr_Fsumdata(index2_pul70_04_04,:),trB_predict_sum(index2_pul70_04_04,:),tr_realloss_sumdata(index2_pul70_04_04,:)];
index2_pul70_05_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.5&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==70);
MBpul70_05_01=[tr_Fsumdata(index2_pul70_05_01,:),trB_predict_sum(index2_pul70_05_01,:),tr_realloss_sumdata(index2_pul70_05_01,:)];
index2_pul70_05_03=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.5&trwaveform_sum(:,4)==0.3&tr_Tsumdata(:,1)==70);
MBpul70_05_03=[tr_Fsumdata(index2_pul70_05_03,:),trB_predict_sum(index2_pul70_05_03,:),tr_realloss_sumdata(index2_pul70_05_03,:)];
index2_pul70_06_02=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.6&trwaveform_sum(:,4)==0.2&tr_Tsumdata(:,1)==70);
MBpul70_06_02=[tr_Fsumdata(index2_pul70_06_02,:),trB_predict_sum(index2_pul70_06_02,:),tr_realloss_sumdata(index2_pul70_06_02,:)];
index2_pul70_07_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.7&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==70);
MBpul70_07_01=[tr_Fsumdata(index2_pul70_07_01,:),trB_predict_sum(index2_pul70_07_01,:),tr_realloss_sumdata(index2_pul70_07_01,:)];

index2_pul90_01_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==90);
MBpul90_01_01=[tr_Fsumdata(index2_pul90_01_01,:),trB_predict_sum(index2_pul90_01_01,:),tr_realloss_sumdata(index2_pul90_01_01,:)];
index2_pul90_01_03=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.3&tr_Tsumdata(:,1)==90);
MBpul90_01_03=[tr_Fsumdata(index2_pul90_01_03,:),trB_predict_sum(index2_pul90_01_03,:),tr_realloss_sumdata(index2_pul90_01_03,:)];
index2_pul90_01_05=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.5&tr_Tsumdata(:,1)==90);
MBpul90_01_05=[tr_Fsumdata(index2_pul90_01_05,:),trB_predict_sum(index2_pul90_01_05,:),tr_realloss_sumdata(index2_pul90_01_05,:)];
index2_pul90_01_07=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.1&trwaveform_sum(:,4)==0.7&tr_Tsumdata(:,1)==90);
MBpul90_01_07=[tr_Fsumdata(index2_pul90_01_07,:),trB_predict_sum(index2_pul90_01_07,:),tr_realloss_sumdata(index2_pul90_01_07,:)];
index2_pul90_02_02=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.2&trwaveform_sum(:,4)==0.2&tr_Tsumdata(:,1)==90);
MBpul90_02_02=[tr_Fsumdata(index2_pul90_02_02,:),trB_predict_sum(index2_pul90_02_02,:),tr_realloss_sumdata(index2_pul90_02_02,:)];
index2_pul90_02_04=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.2&trwaveform_sum(:,4)==0.4&tr_Tsumdata(:,1)==90);
MBpul90_02_04=[tr_Fsumdata(index2_pul90_02_04,:),trB_predict_sum(index2_pul90_02_04,:),tr_realloss_sumdata(index2_pul90_02_04,:)];
index2_pul90_02_06=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.2&trwaveform_sum(:,4)==0.6&tr_Tsumdata(:,1)==90);
MBpul90_02_06=[tr_Fsumdata(index2_pul90_02_06,:),trB_predict_sum(index2_pul90_02_06,:),tr_realloss_sumdata(index2_pul90_02_06,:)];
index2_pul90_03_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.3&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==90);
MBpul90_03_01=[tr_Fsumdata(index2_pul90_03_01,:),trB_predict_sum(index2_pul90_03_01,:),tr_realloss_sumdata(index2_pul90_03_01,:)];
index2_pul90_03_03=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.3&trwaveform_sum(:,4)==0.3&tr_Tsumdata(:,1)==90);
MBpul90_03_03=[tr_Fsumdata(index2_pul90_03_03,:),trB_predict_sum(index2_pul90_03_03,:),tr_realloss_sumdata(index2_pul90_03_03,:)];
index2_pul90_03_05=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.3&trwaveform_sum(:,4)==0.5&tr_Tsumdata(:,1)==90);
MBpul90_03_05=[tr_Fsumdata(index2_pul90_03_05,:),trB_predict_sum(index2_pul90_03_05,:),tr_realloss_sumdata(index2_pul90_03_05,:)];
index2_pul90_04_02=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.4&trwaveform_sum(:,4)==0.2&tr_Tsumdata(:,1)==90);
MBpul90_04_02=[tr_Fsumdata(index2_pul90_04_02,:),trB_predict_sum(index2_pul90_04_02,:),tr_realloss_sumdata(index2_pul90_04_02,:)];
index2_pul90_04_04=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.4&trwaveform_sum(:,4)==0.4&tr_Tsumdata(:,1)==90);
MBpul90_04_04=[tr_Fsumdata(index2_pul90_04_04,:),trB_predict_sum(index2_pul90_04_04,:),tr_realloss_sumdata(index2_pul90_04_04,:)];
index2_pul90_05_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.5&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==90);
MBpul90_05_01=[tr_Fsumdata(index2_pul90_05_01,:),trB_predict_sum(index2_pul90_05_01,:),tr_realloss_sumdata(index2_pul90_05_01,:)];
index2_pul90_05_03=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.5&trwaveform_sum(:,4)==0.3&tr_Tsumdata(:,1)==90);
MBpul90_05_03=[tr_Fsumdata(index2_pul90_05_03,:),trB_predict_sum(index2_pul90_05_03,:),tr_realloss_sumdata(index2_pul90_05_03,:)];
index2_pul90_06_02=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.6&trwaveform_sum(:,4)==0.2&tr_Tsumdata(:,1)==90);
MBpul90_06_02=[tr_Fsumdata(index2_pul90_06_02,:),trB_predict_sum(index2_pul90_06_02,:),tr_realloss_sumdata(index2_pul90_06_02,:)];
index2_pul90_07_01=find(trwaveform_sum(:,1)==3&trwaveform_sum(:,3)==0.7&trwaveform_sum(:,4)==0.1&tr_Tsumdata(:,1)==90);
MBpul90_07_01=[tr_Fsumdata(index2_pul90_07_01,:),trB_predict_sum(index2_pul90_07_01,:),tr_realloss_sumdata(index2_pul90_07_01,:)];

B_Field=readmatrix('B:\NJUPT（Final）\Model\Testing\Material B\B_Field.csv');
  B_predict_sum=[];
  numslope_sum=[];
  waveform_sum=[];
for w=1:size(B_Field,1)% The waveform corresponding to the classification flux density B data series
 everywave=B_Field(w,:);
 B_predict(1,:)=max(everywave);% The maximum value of the flux density data series is the flux density of the test point
 B_predict_sum(w,:)= B_predict(1,:);
 if B_Field(w,18)>0% First isolate the sine wave
   numslope=3;
   waveform=1; 
   D=0;
   D1=0;
   D3=0;
 else % Separate rectangular wave and pulse wave
    fdeverywave=100*B_Field(w,:);
    [TF,S1] = ischange(fdeverywave,'linear','Threshold',10);
    slopetable=tabulate(S1);
     if min(slopetable(:,2))<80
      [lr,lc]=min(slopetable(:,2));
      dyslope=slopetable(lc,1);
      S1(find(S1==dyslope))=[];
      slopetable(lc,:)=[];
     end 
     [C,ia,ic] = unique(S1);
     numslope=size(slopetable,1);
     if numslope==4
      waveform=3;    
      slopeindex=sort(ia);
      D1=round(slopeindex(2,1)/1000,1);
      D3=round((slopeindex(4,1)-slopeindex(3,1))/1000,1);
      D=0;
     else
      [TF,S1] = ischange(fdeverywave,'linear','Threshold',50);
      slopetable=tabulate(S1);
      if min(slopetable(:,2))<80
       [lr,lc]=min(slopetable(:,2));
       dyslope=slopetable(lc,1);
       S1(find(S1==dyslope))=[];
       slopetable(lc,:)=[];
      end 
      [C,ia,ic] = unique(S1);
      numslope=size(slopetable,1); 
      if numslope==4
       waveform=3;    
       slopeindex=sort(ia);
       D1=round(slopeindex(2,1)/1000,1);
       D3=round((slopeindex(4,1)-slopeindex(3,1))/1000,1);
       D=0;
     elseif numslope==2&ia(1,1)>round(ia(1,1)/100)*100-22&ia(1,1)<round(ia(1,1)/100)*100+22
         waveform=2;
         D=round(max(ia)/1000,1);
         D1=0;
         D3=0;
      else
       [TF,S1] = ischange(fdeverywave,'linear','Threshold',0.8);
       slopetable=tabulate(S1);
       if min(slopetable(:,2))<80
        [lr,lc]=min(slopetable(:,2));
        dyslope=slopetable(lc,1);
        S1(find(S1==dyslope))=[];
        slopetable(lc,:)=[];
       end 
       [C,ia,ic] = unique(S1);
       numslope=size(slopetable,1); 
       if numslope==4
        waveform=3;    
        slopeindex=sort(ia);
        D1=round(slopeindex(2,1)/1000,1);
        D3=round((slopeindex(4,1)-slopeindex(3,1))/1000,1);
        D=0;
       else numslope==3
         waveform=2;
         D=round(max(ia)/1000,1);
         D1=0;
         D3=0;
       end
   end
     end
 end
 numslope_every(1,:)=numslope;
 numslope_sum(w,:)=numslope_every(1,:);
 waveform_every(1,1)=waveform;
 waveform_every(1,2)=D;
 waveform_every(1,3)=D1;
 waveform_every(1,4)=D3;
 waveform_sum(w,1)= waveform_every(1,1);
 waveform_sum(w,2)= waveform_every(1,2);
 waveform_sum(w,3)= waveform_every(1,3);
 waveform_sum(w,4)= waveform_every(1,4);
end 

Temperature=readmatrix('B:\NJUPT（Final）\Model\Testing\Material B\Temperature.csv');
Tsumdata=Temperature;
Frequency=readmatrix('B:\NJUPT（Final）\Model\Testing\Material B\Frequency.csv');
Fsumdata=Frequency;

TPlossstore_array=[]; 
    csT=Tsumdata;%Reading temperature 
    csF=Fsumdata;%Reading frequency
    csB=B_predict_sum;%Read the flux density
    cswave=waveform_sum;%Read waveform information for the material test data
    Plossstore=[];
    zljdstore=[];
    cssjstore=[csT,csF,csB];
    for e=1:length(csT)
        T=num2str(csT(e,1));%The temperature at each test point of the material is read as a string
        f=csF(e,1);%Read the frequency of each test point for the material
        B=csB(e,1);% Read the magnetic flux density for the material at each test point
        Excitation=cswave(e,1);% Read the excitation waveform information for each test point of the material 
        if Excitation==2%Select the training waveform database of response according to the excitation waveform
            Excitation='rect';
            D=num2str(cswave(e,2)*1*10);
            bl='0';
            sjkname=['MB' Excitation T '_' bl D];%Gets the database name corresponding to the test point
            sjk=eval(sjkname);
            Excitation=2;
        elseif Excitation==3
            Excitation='pul';
            D1=num2str(cswave(e,3)*10);
            D3=num2str(cswave(e,4)*10);
            bl='0';
            sjkname=['MB' Excitation T '_' bl D1 '_' bl D3];%Gets the database name corresponding to the test point
            sjk=eval(sjkname);
            Excitation=3;
         else
            Excitation='sin';
            sjkname=['MB' Excitation T];%Gets the database name corresponding to the test point
            sjk=eval(sjkname);
            Excitation=1;
        end
      fB1=find(sjk(:,1)==f&sjk(:,2)==B);       
sjk(fB1,:)=[];
%Search for the first row adjacent to the B value
knljd1_B=abs(sjk(:,2)-B);
knljd1_Bmin=min(knljd1_B);
[row1,col1]=find(knljd1_B==knljd1_Bmin);
ljd1_B=sjk(row1,[1,2,3]);
sjk(row1,:)=[];%After the first neighboring row is found, the row is removed from the original matrix to facilitate the second neighboring value search
%Search for the second row adjacent to the B value
knljd2_B=abs(sjk(:,2)-B);
knljd2_Bmin=min(knljd2_B);
[row2,col2]=find(knljd2_B==knljd2_Bmin);
ljd2_B=sjk(row2,[1,2,3]);
%Search for the third row adjacent to the B value
sjk(row2,:)=[];
knljd3_B=abs(sjk(:,2)-B);
knljd3_Bmin=min(knljd3_B);
[row3,col3]=find(knljd3_B==knljd3_Bmin);
ljd3_B=sjk(row3,[1,2,3]);
%Search for the fourth row adjacent to the B value
sjk(row3,:)=[];
knljd4_B=abs(sjk(:,2)-B);
knljd4_Bmin=min(knljd4_B);
[row4,col4]=find(knljd4_B==knljd4_Bmin);
ljd4_B=sjk(row4,[1,2,3]);
%Combine the rows of the four adjacent values of B found in the zljd_B matrix
zljd_B=[ljd1_B;ljd2_B;ljd3_B;ljd4_B];
%If there are points with the same frequency and different flux in zljd_B_25, 
% when these two points are selected at the same time, the equation will be wrong, so if it is found out, it will be removed
fxtstore=[];
for i=1:size(zljd_B,1)-1
    for j=i+1:size(zljd_B,1)
        if zljd_B(i,1)==zljd_B(j,1)
            fxt(1,:)=zljd_B(j,:);
            fxtstore(j,:)=fxt(1,:);
        end
    end
end
if ~isempty(fxtstore)
fxtstore(all(fxtstore==0,2),:)=[];
for k=1:size(fxtstore)
     scfxt=find(fxtstore(k,1)==zljd_B(:,1)&fxtstore(k,2)==zljd_B(:,2));
     zljd_B(scfxt,:)=[];
end
end
[m,l]=size(zljd_B);%Determine the number of adjacent points in zljd_B
if m==1%If there is only one neighboring point in zljd_B, find another neighboring point in the original matrix
    fB4=find(sjk(:,1)==zljd_B(1,1));
    sjk(fB4,:)=[];
    knljd5_B=abs(sjk(:,2)-B);
    knljd5_Bmin=min(knljd5_B);
    [row5,col5]=find(knljd5_B==knljd5_Bmin);
    ljd5_B=sjk(row5,[1,2,3]);
    zljd_B=[zljd_B;ljd5_B];
end
%The adjacent points in zljd_B are divided into three parts: less than f, greater than f and equal to f
xyfstore=[];
dyfstore=[];
deyfstore=[];
for i=1:size(zljd_B,1)
   if zljd_B(i,1)<f
      xyf(1,:)=zljd_B(i,:);
      xyfstore(i,:)=xyf(1,:);
      if ~isempty(xyfstore)
      xyfstore(all(xyfstore==0,2),:)=[];
      end
      elseif zljd_B(i,1)>f
      dyf(1,:)=zljd_B(i,:);
      dyfstore(i,:)=dyf(1,:); 
      if ~isempty(dyfstore)
      dyfstore(all(dyfstore==0,2),:)=[];
      end
      elseif zljd_B(i,1)==f
       deyf(1,:)=zljd_B(i,:);
      deyfstore(i,:)=deyf(1,:) ;
      if ~isempty(deyfstore)
      deyfstore(all(deyfstore==0,2),:)=[];
      end
   end
end 
%Select the final proximity points according to the above classification
if isempty(xyfstore)&isempty(deyfstore)&~isempty(dyfstore)
      dyfwcljd1_f=abs(dyfstore(:,1)-f);
      dyfwcljd1_fmin=min(dyfwcljd1_f);
      [row8,col8]=find(dyfwcljd1_f==dyfwcljd1_fmin);
      dyfljd1_f=dyfstore(row8,[1,2,3]);
      fB2=find(dyfstore(:,1)==dyfljd1_f(1,1)&dyfstore(:,2)==dyfljd1_f(1,2));
      scdyfstore=dyfstore;
      scdyfstore(fB2,:)=[];
      dyfwcljd2_f=abs(scdyfstore(:,1)-f);
      dyfwcljd2_fmin=min(dyfwcljd2_f);
      [row9,col9]=find(dyfwcljd2_f==dyfwcljd2_fmin);
      dyfljd2_f=scdyfstore(row9,[1,2,3]);
      zljd=[dyfljd1_f,dyfljd2_f];
      zljdstore=[zljdstore;zljd];
elseif isempty(xyfstore)&~isempty(deyfstore)&~isempty(dyfstore)
      dyfwcljd1_f=abs(dyfstore(:,1)-f);
      dyfwcljd1_fmin=min(dyfwcljd1_f);
      [row8,col8]=find(dyfwcljd1_f==dyfwcljd1_fmin);
      dyfljd1_f=dyfstore(row8,[1,2,3]);
      zljd=[dyfljd1_f,deyfstore(1,[1,2,3])];
      zljdstore=[zljdstore;zljd];
  elseif isempty(dyfstore)&isempty(deyfstore)&~isempty(xyfstore)
      xyfwcljd1_f=abs(xyfstore(:,1)-f);
      xyfwcljd1_fmin=min(xyfwcljd1_f);
      [row10,col10]=find(xyfwcljd1_f==xyfwcljd1_fmin);
      xyfljd1_f=xyfstore(row10,[1,2,3]);
      fB3=find(xyfstore(:,1)==xyfljd1_f(1,1)&xyfstore(:,2)==xyfljd1_f(1,2));
      scxyfstore=xyfstore;
      scxyfstore(fB3,:)=[];
      xyfwcljd2_f=abs(scxyfstore(:,1)-f);
      xyfwcljd2_fmin=min(xyfwcljd2_f);
      [row11,col11]=find(xyfwcljd2_f==xyfwcljd2_fmin);
      xyfljd2_f=scxyfstore(row11,[1,2,3]);
      zljd=[xyfljd1_f,xyfljd2_f];
      zljdstore=[zljdstore;zljd];
elseif isempty(dyfstore)&~isempty(deyfstore)&~isempty(xyfstore)
      xyfwcljd1_f=abs(xyfstore(:,1)-f);
      xyfwcljd1_fmin=min(xyfwcljd1_f);
      [row10,col10]=find(xyfwcljd1_f==xyfwcljd1_fmin);
      xyfljd1_f=xyfstore(row10,[1,2,3]);
      zljd=[xyfljd1_f,deyfstore(1,[1,2,3])];
      zljdstore=[zljdstore;zljd];
elseif isempty(xyfstore)&isempty(dyfstore)&~isempty(deyfstore) 
      zljd=deyfstore;
      zljdstore=[zljdstore;zljd];
else 
      xyfwcljd1_f=abs(xyfstore(:,1)-f);
      xyfwcljd1_fmin=min(xyfwcljd1_f);
      [row12,col12]=find(xyfwcljd1_f==xyfwcljd1_fmin);
      xyfljd1_f=xyfstore(row12,[1,2,3]);
      dyfwcljd1_f=abs(dyfstore(:,1)-f);
      dyfwcljd1_fmin=min(dyfwcljd1_f);
      [row13,col13]=find(dyfwcljd1_f==dyfwcljd1_fmin);
      dyfljd1_f=dyfstore(row13,[1,2,3]);
      zljd=[xyfljd1_f,dyfljd1_f];
      zljdstore=[zljdstore;zljd];
end 

  %Do core loss separation calculations
  t1=zljd(1,4)/zljd(1,1);
  t2=zljd(1,5)/zljd(1,2);
  a1=[1,1;t1*t2.^2,t1.^2*t2.^2];%Coefficient on the left hand side of a quadratic equation
  b1=[zljd(1,3) zljd(1,6)]';%Coefficient on the right-hand side of a quadratic equation
  c1=a1\b1;%Equation solution vector
%The results of flux separation are discussed in different cases
  if c1(2,1)<0
     if zljd(1,1)>f&zljd(1,4)>f%If both adjacent points are greater than f, the smaller of the two points is chosen for calculation
      Ploss=f/zljd(1,1)*(B/zljd(1,2))^2*zljd(1,3);
      Ph=Ploss;
      Pe=0;
      Plossstore=[Plossstore;Ploss];
     elseif zljd(1,1)<f&zljd(1,4)<f%If both adjacent points are less than f, choose the larger of the two points for calculation
      Ploss=f/zljd(1,1)*(B/zljd(1,2))^2*zljd(1,3);
      Ph=Ploss;
      Pe=0;
      Plossstore=[Plossstore;Ploss];
     else %If one of the two adjacent points is greater than f and one is less than f, take the mean of both
      Plossxyf=f/zljd(1,1)*(B/zljd(1,2))^2*zljd(1,3);
      Plossdyf=f/zljd(1,4)*(B/zljd(1,5))^2*zljd(1,6);
      Ploss=(Plossxyf+ Plossdyf)/2;
      Ph=Ploss;
      Pe=0;
      Plossstore=[Plossstore;Ploss];
     end       
  elseif c1(1,1)<0
      if zljd(1,1)>f&zljd(1,4)>f
      Ploss=(f/zljd(1,1))^2*(B/zljd(1,2))^2*zljd(1,3);
      Ph=0;
      Pe=Ploss;
      Plossstore=[Plossstore;Ploss];
      elseif zljd(1,1)<f&zljd(1,4)<f
      Ploss=(f/zljd(1,1))^2*(B/zljd(1,2))^2*zljd(1,3);
      Ph=0;
      Pe=Ploss;
      Plossstore=[Plossstore;Ploss];
      else 
      Plossxyf=(f/zljd(1,1))^2*(B/zljd(1,2))^2*zljd(1,3);
      Plossdyf=(f/zljd(1,4))^2*(B/zljd(1,5))^2*zljd(1,6);
      Ploss=(Plossxyf+ Plossdyf)/2;
      Ph=0;
      Pe=Ploss;
      Plossstore=[Plossstore;Ploss];
      end 
  else 
      Ph=(f/zljd(1,1))*(B/zljd(1,2))^2*c1(1,1);
      Pe=(f/zljd(1,1))^2*(B/zljd(1,2))^2*c1(2,1);
      Ploss=Ph+Pe;
      Plossstore=[Plossstore;Ploss];
  end  
end
Pred_Material_B=Plossstore;
csvwrite('Pred_Material B.csv', Pred_Material_B);%Output the predicted value of the material test points as a.csv file
end