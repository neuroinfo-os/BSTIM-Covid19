clear all;
close all;

dbstop if error
cd('D:\Dropbox\Covid_19_FIT_Modelle')
load('PrepocessedData2.mat')
load('Germany_pop_data.mat')

% Covid Age groups {'A00-A04';'A05-A14';'A15-A34';'A35-A59';'A60-A79';'A80+';'unbekannt'}
% Match 
% 'A00..04'
% 'A05..09' + 'A10..14'
% 'A15..19' + 'A20..24' +  'A25..29'+  'A30..34'
% 'A35..39' + 'A40..44' +  'A45..49' +  'A50..54' + 'A55..59'
% 'A60..64' + 'A65..69' +  'A70..74' +  'A75..79'
% 'A80+'
% unbekannt

Take_Year = 2018;
Year          = table2array(germanypopulationdata(:,5));
IDX_thisyear  = find(Year==Take_Year);
Year          = table2array(germanypopulationdata(IDX_thisyear,5));

StatesName    = table2array(germanypopulationdata(IDX_thisyear,1));
LKName        = table2array(germanypopulationdata(IDX_thisyear,2));
Gender        = table2array(germanypopulationdata(IDX_thisyear,3));
Gender_is_m   = strcmp(Gender,'m');
Age_Group     = table2array(germanypopulationdata(IDX_thisyear,4));
Year          = table2array(germanypopulationdata(IDX_thisyear,5));
Population    = table2array(germanypopulationdata(IDX_thisyear,6));
Age_Group_ID  = zeros(size(Age_Group ));
UniqueAgeGr   =unique(Age_Group );
for IDX=1:length(UniqueAgeGr )
   actstr       = UniqueAgeGr(IDX); 
   IDX_this_age = find(strcmp(Age_Group,actstr));
   Age_Group_ID(IDX_this_age) = IDX;
end


for IDXAge=1:7 
    
    if IDXAge==1
       JoinIDX = 1; 
    end
    if IDXAge==2
       JoinIDX = [2 3]; 
    end
    if IDXAge==3
       JoinIDX = [4 5 6 7]; 
    end
    if IDXAge==4
       JoinIDX = [8 9 10 11 12];        
    end
    if IDXAge==5
       JoinIDX = [13 14 15];  
       
    end
    if IDXAge==6
       JoinIDX = [16];   
       
    end
    if IDXAge==7
       JoinIDX = [1:16]; 
       
    end
%     for IDXLK=1:length(LKName)
%         if ~isempty(findstr(LKName{IDXLK},'Aachen'))
%            yuuh=1; 
%         end
%     end
    for IDXLK=1:length(Data.LK.names);
        act_name = Data.LK.names{IDXLK};
        act_ID   = Data.LK.IDs(IDXLK);
        
        
        if strfind(act_name, 'ö')>0
            IDS     = strfind(act_name, 'ö');
            act_name =[act_name(1:(IDS-1)) 'Ã¶' act_name((IDS+1):end)]; 
        end
        if  strfind(act_name, 'ä')>0
            IDS     = strfind(act_name, 'ä');
            act_name =[act_name(1:(IDS-1)) 'Ã¤' act_name((IDS+1):end)]; 
        end
        
        if  strfind(act_name, 'ü')>0
            IDS     = strfind(act_name, 'ü');
            act_name =[act_name(1:(IDS-1)) 'Ã¼' act_name((IDS+1):end)]; 
        end
        if  strfind(act_name, 'ß')>0
            IDS     = strfind(act_name, 'ß');
            act_name =[act_name(1:(IDS-1)) 'ÃŸ' act_name((IDS+1):end)]; 
        end
        if  IDXLK==25
            act_name ='LK BergstraÃŸe'
        end
        if  IDXLK==78
            act_name ='LK GieÃŸen'
        end
        if  IDXLK==84
            act_name ='LK GroÃŸ-Gerau'
        end
        if  IDXLK==94
            act_name ='LK HaÃŸberge'
        end
        if  IDXLK==134
            act_name ='LK LudwigslustÂ–Parchim'
        end
        if  IDXLK==163
            act_name ='LK Neustadt/Aisch-Bad Windsheim'
        end
        if  IDXLK==271
            act_name = 'LK VorpommernÂ–Greifswald'
        end
        if  IDXLK==412
            act_name = 'StÃ¤dteRegion Aachen'
        end
        if  IDXLK==272
            act_name = 'LK VorpommernÂ–RÃ¼gen'
        end
        IDX_LKs = find(strcmp(act_name,LKName));  
        
        if isempty(IDX_LKs )
            warning('shit')
            find(strncmpi(act_name,LKName,10) )
        end
        Temp1 = Age_Group_ID(IDX_LKs);
        Temp2 = Gender_is_m(IDX_LKs); 
        Temp3 = Population(IDX_LKs);        
        IDXm  = find(Temp2==1);
        IDXf  = find(Temp2==0);
      
        Data.LK.Population(IDXLK,IDXAge,1)= sum(Temp3(IDXm( JoinIDX ) ));
        Data.LK.Population(IDXLK,IDXAge,2)= sum(Temp3(IDXf( JoinIDX ) ));
       
    end
end
save('PrepocessedData3.mat','Data')