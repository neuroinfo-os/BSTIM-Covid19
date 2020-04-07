clear all;
close all;
dbstop if error
cd('D:\Dropbox\Covid_19_FIT_Modelle')


import_CVS;

%load('COVID_19_cuurent.mat')
days(1) = 31;
days(2) = 29;
days(3) = 31;
days(4) = 30;
days(5) = 31;
RKICOVID19 =RKICOVID1;
StatesID    = table2array (RKICOVID19(2:end,1));
StateName   = table2array (RKICOVID19(2:end,2));
DateItem    = table2array (RKICOVID19(2:end,9));
LKName      = table2array (RKICOVID19(2:end,3));
LKID        = zeros(size(StatesID));

Data.StatesID  =StatesID;
Data.StateName =StateName;
Data.LKName =LKName;
Data.LKID =LKID;
Data.NewCases = table2array(RKICOVID19(2:end,6));
Data.Death    = table2array(RKICOVID19(2:end,7));
Data.Agegroup = table2array(RKICOVID19(2:end,4));

Agegroups     = unique(Data.Agegroup);
Data.Agegroups.Name = Agegroups  ;
Data.Agegroups.ID   = 1:length(Agegroups)  ;
Data.AgegroupID     = zeros(size(StatesID));

State.IDs   = unique(StatesID);
for IDX=1:length(State.IDs)
    act_state_ID = State.IDs(IDX);
    State.Name{IDX}=StateName(min(find(StatesID ==act_state_ID)));
end
Data.Time_ID = zeros(size(DateItem)); 
for IDX=1:length(DateItem);
    monthID = str2num(DateItem{IDX}(6:7));
    dayId   = str2num(DateItem{IDX}(9:10));
    if monthID==1
        Time_ID = dayId ;
    end
    if monthID==2
        Time_ID = dayId +days(1) ;
    end
    if monthID==3
        Time_ID = dayId +days(2) +days(1) ;
    end
    if monthID==4
        Time_ID = dayId +days(3)+days(2) +days(1) ;
    end
    if monthID==5
        Time_ID = dayId +days(4)+days(3)+days(2) +days(1) ;
    end
    Data.Time_ID(IDX) = Time_ID;
    for IDXA =1:length(Data.Agegroups.ID)
        if strcmp(Data.Agegroup{IDX},Data.Agegroups.Name{IDXA})
            Data.AgegroupID(IDX) = IDXA; 
        end
    end
end


LK.names    = unique(LKName );
LK.IDs      = 1:length(LK.names);
for IDX=1:length(LK.names);
    act_name = LK.names(IDX)
    for IDX1=1:length(LKName)
       if strcmp(LKName{IDX1},act_name)==1
          LKID(IDX1)  = IDX;
       end
    end
end
Data.State = State;
Data.LK    =LK;
Data.LKID =LKID;
save('PrepocessedData.mat','Data')

%Name_State  = 
