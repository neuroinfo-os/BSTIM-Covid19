clear all;
close all;
days(1) = 31;
days(2) = 29;
days(3) = 31;
days(4) = 30;
days(5) = 31;

dbstop if error
cd('D:\Dropbox\Covid_19_FIT_Modelle')
load('PrepocessedData.mat')

max_Period      = days(1)+days(2)+days(3)+6;

Data_cases = [];
Data_death = [];

for IDXL = Data.LK.IDs
   IDX1         = find(Data.LKID==IDXL);
   act_times    = Data.Time_ID(IDX1);
   act_new_cases= Data.NewCases(IDX1);
   act_death    = Data.Death(IDX1);
   act_age_ID   = Data.AgegroupID(IDX1);
   
   [act_times IDXS] = sort(act_times);
   act_new_cases    = act_new_cases(IDXS);
   act_death        = act_death(IDXS);
   act_age_ID       = act_age_ID(IDXS);  
   time_vec_new_cases  = zeros(length(Data.Agegroups.ID),max_Period);
   time_vec_new_death  = zeros(length(Data.Agegroups.ID),max_Period);
   
   if isempty(Data_cases)
       Data_cases = zeros(length(Data.LK.IDs),length(Data.Agegroups.ID),max_Period);
       Data_death = zeros(length(Data.LK.IDs),length(Data.Agegroups.ID),max_Period);
   end
   for IDX =1:length( act_times)      
       time_vec_new_cases(act_age_ID(IDX), act_times(IDX))=time_vec_new_cases(act_age_ID(IDX), act_times(IDX))+act_new_cases(IDX);
       time_vec_new_death(act_age_ID(IDX), act_times(IDX))=time_vec_new_death(act_age_ID(IDX), act_times(IDX))+act_death(IDX);
   end
   Data.LK.Cases{IDXL}.time_vec_new_cases =time_vec_new_cases;
   Data.LK.Cases{IDXL}.time_vec_new_death =time_vec_new_death;
   Data_cases(IDXL,:,:)=time_vec_new_cases; 
   Data_death(IDXL,:,:)=time_vec_new_death; 
   if mod(IDXL,6)==1
       figure;
   end
        
   
   subplot(6,2,2*(mod(IDXL-1,6)+1)-1)
   plot(1:max_Period,Data.LK.Cases{IDXL}.time_vec_new_cases);
   title(Data.LK.names{IDXL})
   subplot(6,2,2*(mod(IDXL-1,6)+1))
   plot(1:max_Period,Data.LK.Cases{IDXL}.time_vec_new_death);
   legend(Data.Agegroups.Name,'Location','eastoutside')
  
end
Data.Data_cases=Data_cases;
Data.Data_death=Data_death;
save('PrepocessedData2.mat','Data')