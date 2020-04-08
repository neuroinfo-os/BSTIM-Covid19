clear all;
close all;

formats.Fig =1;
formats.JPG =1;
formats.Res =150;
formats.PDF =1;
formats.eps =1;

days(1) = 31;
days(2) = 29;
days(3) = 31;
days(4) = 30;
days(5) = 31;
Max_order_poly      = 3;
%Max_order_polyEnd   = 4;
Max_order_Peri      = 3;
nr_knots            = 4;
start_day           = 45;
Prediciton_horizon  =21;
end_days            = 6;
use_cum_sum_flag    = 0;
use_splines         = 0;
Historie{1}.day     = 28;
Historie{1}.event   ='Pat ONE'
Historie{2}.day     = days(1)+15;
Historie{2}.event   ='Kappensitzung Gangelt'
Historie{3}.day     = days(1)+days(2)+17;
Historie{3}.event   ='Schulen'
Historie{4}.day     = days(1)+days(2)+25;
Historie{4}.event   ='Kontakt Verbot'
Historie{5}.day     = 96;
Historie{5}.event   ='Last Data'

dbstop if error
cd('D:\Dropbox\Covid_19_FIT_Modelle')
load('PrepocessedData3.mat')

Data.Data_cases = Data.Data_cases(:,:,1:(size(Data.Data_cases,3)-1));
Data.Data_death = Data.Data_death(:,:,1:(size(Data.Data_death,3)-1));
max_Period      = size(Data.Data_cases,3);


%SumCases_Germany= squeeze(sum(Data.Data_cases,1));
%SumDeath_Germany= squeeze(sum(Data.Data_death,1));




%%%%% Case one all Germany - New Cases 
if use_cum_sum_flag    == 1;
    CumSumCases_Germany= sum(cumsum(squeeze(sum(Data.Data_cases,1))')',1);
    CumSumDeath_Germany= sum(cumsum(squeeze(sum(Data.Data_death,1))')',1);
else
    CumSumCases_Germany= sum((squeeze(sum(Data.Data_cases,1))')',1);
    CumSumDeath_Germany= sum((squeeze(sum(Data.Data_death,1))')',1);
end

if use_splines==0
    Time_vec           = 1:length(CumSumCases_Germany);
    BasisMAT_Trend     = zeros(Max_order_poly+1,length(CumSumCases_Germany));
    
    for Order=0:Max_order_poly
        BasisMAT_Trend(Order+1,:)=Time_vec.^(Order);        
        normK(Order+1)           = max(BasisMAT_Trend(Order+1,:));
        BasisMAT_Trend(Order+1,:)=BasisMAT_Trend(Order+1,:)./( normK(Order+1) );
    end
    Time_vecp           = 1:(length(CumSumCases_Germany)+Prediciton_horizon);
    BasisMAT_Trendp     = zeros(Max_order_poly+1,length(Time_vecp));
    for Order=0:Max_order_poly
        BasisMAT_Trendp(Order+1,:)=Time_vecp.^(Order);
        BasisMAT_Trendp(Order+1,:)=BasisMAT_Trendp(Order+1,:)./( normK(Order+1) );
    end

end
BasisMAT_Trendp(end+1,:)=BasisMAT_Trendp(1,:).*0;
temp           = (1:size(Data.Data_cases,3))-size(Data.Data_cases,3)+end_days;
temp(find(temp<0))=0;
%tempB1            = temp.^2;
tempB2            = temp.^4;
tempB2            = tempB2./max(tempB2);
%CumSumCases_Germany(find(CumSumCases_Germany)==0)=NaN;

BasisMAT_Trend     = [BasisMAT_Trend ; tempB2];
IDX_Component_End  = size(BasisMAT_Trend,1);
Design_Matrix = BasisMAT_Trend;
[beta,dev,stats]            = glmfit(Design_Matrix' , CumSumCases_Germany,'poisson','constant','off');
yfit                        = glmval(beta,Design_Matrix','log','constant','off');
beta_pred                   = beta;

beta_pred(IDX_Component_End)    = 0;
yfit_prediction             = glmval(beta_pred,Design_Matrix','log','constant','off');

figure

plot(yfit,'g')
hold on
plot(CumSumCases_Germany)
plot(yfit_prediction,'r')
growth_factor               =  beta'*Design_Matrix;
%growth_factor_prediction    =  beta_pred'*Design_Matrix;

if use_cum_sum_flag    == 1;
    figure
    subplot(4,1,1)
    plot(1:length(CumSumCases_Germany),CumSumCases_Germany)
    title('Trend only')
    hold on
    plot(1:length(CumSumCases_Germany),yfit,'r')
    subplot(4,1,2)
    semilogy(1:length(CumSumCases_Germany),CumSumCases_Germany)
    hold on
    semilogy(1:length(CumSumCases_Germany),yfit,'r')
    if use_splines==1
            semilogy(knots,yfit(knots),'rx')
    end   
    set(gca,'Ylim',[1 max(yfit)*1.5]);
    subplot(4,1,3)
    plot(1:length(CumSumCases_Germany),growth_factor)
    hold on
    if use_splines==1
        plot(knots,growth_factor(knots),'rx')
    end
    title('exponent')
    subplot(4,1,4)
    plot(diff(growth_factor)*100)
    set(gca,'Ylim',[0 max(diff(growth_factor))*110]);
    if use_splines==1
        plot(knots,growth_factor(knots),'rx')
    end
    title('Growth Factor %')
    
else
    
    figure
    subplot(5,1,1)
    plot(1:length(CumSumCases_Germany),(CumSumCases_Germany))
    title('Trend only - New cases per day')
    hold on
    plot(1:length(CumSumCases_Germany),(yfit),'r')
    
    subplot(5,1,2)
    plot(1:length(CumSumCases_Germany),cumsum(CumSumCases_Germany))
    title('Trend only - Cumulative')
    hold on
    plot(1:length(CumSumCases_Germany),cumsum(yfit),'r')
    
    subplot(5,1,3)
    semilogy(1:length(CumSumCases_Germany),cumsum(CumSumCases_Germany),'b')
    title('Trend only - Cumulative')
    hold on
    temp = cumsum(yfit);
    semilogy(1:length(CumSumCases_Germany),temp,'r');
    hold on
    set(gca,'Ylim',[1 max([max(cumsum(CumSumCases_Germany)) max(temp)])*2])
    
    subplot(5,1,4)
    temp = diff(log(cumsum(yfit)));
    temp(find(cumsum(yfit)<1))=0;
    plot(temp*100)
    hold on
    set(gca,'Ylim',[0 max(temp)*110]);
    if use_splines==1
        plot((knots(5:end)-1),temp((knots(5:end)-1))*100,'rx')
    end
    title('Growth Factor %')
    
    subplot(5,1,5)
    temp = growth_factor;
    plot(temp)
    title('Exponent lambda')
end


%Periodic Component
%------------------------------------------------------------------------
%------------------------------------------------------------------------
%------------------------------------------------------------------------

if use_cum_sum_flag    == 1;
    CumSumCases_Germany= sum(cumsum(squeeze(sum(Data.Data_cases,1))')',1);
    CumSumDeath_Germany= sum(cumsum(squeeze(sum(Data.Data_death,1))')',1);
else
    CumSumCases_Germany= sum((squeeze(sum(Data.Data_cases,1))')',1);
    CumSumDeath_Germany= sum((squeeze(sum(Data.Data_death,1))')',1);
end

Time_vec           = 1:size(CumSumCases_Germany,2);

for Order=1:Max_order_Peri
    BasisMAT_Periodic(Order,:)  =(mod(Time_vec,7)+1).^(Order);
    BasisMAT_Periodic(Order,:)  =BasisMAT_Periodic(Order,:)./(max(BasisMAT_Periodic(Order,:)));
    BasisMAT_PeriodicAv(Order,:)=BasisMAT_Periodic(Order,:).*0+mean(BasisMAT_Periodic(Order,:));
    
    BasisMAT_Periodicp(Order,:)  =(mod(Time_vecp,7)+1).^(Order);
    BasisMAT_Periodicp(Order,:)  =BasisMAT_Periodicp(Order,:)./(max(BasisMAT_Periodicp(Order,:)));
    BasisMAT_PeriodicAvp(Order,:)=BasisMAT_Periodicp(Order,:).*0+mean(BasisMAT_Periodicp(Order,:));
end
%plot(BasisMAT_Periodic');

Design_Matrix    = [BasisMAT_Trend ; BasisMAT_Periodic];
Design_MatrixNOP = [BasisMAT_Trendp ; BasisMAT_PeriodicAvp];
%BasisMAT_Trendp

[beta,dev,stats] = glmfit(Design_Matrix' , CumSumCases_Germany,'poisson','constant','off');
%beta(IDX_Component_End) = 0;
yfit    = glmval(beta,Design_Matrix','log','constant','off');
yfitNOP = glmval(beta,Design_MatrixNOP','log','constant','off');
Periodic_component  =  beta((size(BasisMAT_Trend,1)+1):end)'*Design_Matrix((size(BasisMAT_Trend,1)+1):end,:);
growth_factor       =  beta'*Design_Matrix;
betat = beta;
betat(IDX_Component_End) =0;
yfitNOP2 = glmval(betat,Design_MatrixNOP','log','constant','off');


ID=figure;
fig_size_paper=[1, 1, 20, 28];
fig_size_screen=[1, 1, 600, 800];
set(ID,'PaperType','A4');
set(ID,'PaperOrientation','portrait');
set(ID,'PaperUnits','centimeters');
set(ID,'PaperPosition',fig_size_paper);
set(ID,'Units','pixel');
set(ID,'Position',fig_size_screen);

subplot(6,1,1)
plot(1:length(CumSumCases_Germany),(CumSumCases_Germany))
title('Trend +Period - Green cleaned from periodic - New cases per day')
hold on
plot(1:length(CumSumCases_Germany),(yfit),'r')
plot(1:length(yfitNOP2),(yfitNOP),'g')
plot(1:length(yfitNOP2),(yfitNOP2),'m')
xlabel('Days start Jan 1st')
set(gca,'Xlim',[1 length(yfitNOP2)]);
for IDXE =1:length(Historie)
    plot([Historie{IDXE}.day  Historie{IDXE}.day ],[0 max(CumSumCases_Germany)*1.1],'m')
    text(Historie{IDXE}.day,max(CumSumCases_Germany)*0.1,Historie{IDXE}.event,'Rotation',90)
end


legend('Reported','Model Trend+Period.','Model Trend only','Predicition Model - cleaned trend','Location','west')


subplot(6,1,2)
plot(1:length(CumSumCases_Germany),cumsum(CumSumCases_Germany))
title('Trend +Period - Green cleaned from periodic - Cumulative')
hold on
plot(1:length(CumSumCases_Germany),cumsum(yfit),'r')
plot(1:length(yfitNOP2),cumsum(yfitNOP),'g')
plot(1:length(yfitNOP2),cumsum(yfitNOP2),'m')
set(gca,'Xlim',[1 length(yfitNOP2)]);
for IDXE =1:length(Historie)
    plot([Historie{IDXE}.day  Historie{IDXE}.day ],[0 max(cumsum(CumSumCases_Germany))*1.1],'m')
    text(Historie{IDXE}.day,max(cumsum(CumSumCases_Germany))*0.1,Historie{IDXE}.event,'Rotation',90)
end
legend('Reported','Model Trend+Period.','Model Trend only','Predicition Model - cleaned trend','Location','west')
xlabel('Days starting Jan 1st')

subplot(6,1,3)
semilogy(1:length(CumSumCases_Germany),cumsum(CumSumCases_Germany),'b')
title('Trend +Period - Green cleaned from periodic - Cumulative')
hold on
temp = cumsum(yfit);
semilogy(1:length(CumSumCases_Germany),temp,'r');
semilogy(1:length(yfitNOP2),cumsum(yfitNOP),'g');
semilogy(1:length(yfitNOP2),cumsum(yfitNOP2),'m');
hold on
if use_splines==1
    plot(knots,temp(knots),'rx')
end
set(gca,'Ylim',[1 max([max(cumsum(CumSumCases_Germany)) max(temp)])*2])
set(gca,'Xlim',[1 length(yfitNOP2)]);
legend('Reported','Model Trend+Period.','Model Trend only','Predicition Model - cleaned trend','Location','west')
xlabel('Days starting Jan 1st')

subplot(6,1,4)
%temp = diff(log(cumsum(yfit)));

tempb = cumsum(yfitNOP2);
temp = yfitNOP2./tempb
temp(find(cumsum(yfitNOP2)<1))=0;
plot(temp*100,'g')
hold on

tempb = cumsum(yfit);
temp = yfit./tempb
temp(find(cumsum(yfit)<1))=0;
plot(temp*100,'r')
hold on

set(gca,'Ylim',[0 max(temp)*110]);
set(gca,'Xlim',[1 length(yfitNOP2)]);
if use_splines==1
    plot((knots(5:end)-1),temp((knots(5:end)-1))*100,'rx')
end
for IDXE =1:length(Historie)
    plot([Historie{IDXE}.day  Historie{IDXE}.day ],[0 max(temp)*110],'m')
    text(Historie{IDXE}.day,max(temp)*110*0.1,Historie{IDXE}.event,'Rotation',90)
end
xlabel('Days starting Jan 1st')
ylabel('% growth per day')
title('Trend +Period - Green cleaned from periodic - Growth Factor %')
legend('Growth rate - Prediciton Model','Growth rate Trend +Period ','Location','west')

subplot(6,1,5)
%temp = diff(log(cumsum(yfit)));

tempb = cumsum(yfitNOP2);
temp = yfitNOP2./tempb
temp = log(2)./log(1+temp );

plot(temp,'g')
hold on
set(gca,'Ylim',[0 25]);
set(gca,'Xlim',[1 length(yfitNOP2)]);
if use_splines==1
    plot((knots(5:end)-1),temp((knots(5:end)-1)),'rx')
end
for IDXE =1:length(Historie)
    plot([Historie{IDXE}.day  Historie{IDXE}.day ],[0 max(temp)*1.1],'m')
    text(Historie{IDXE}.day,max(temp)*110*0.1,Historie{IDXE}.event,'Rotation',90)
end
xlabel('Days starting Jan 1st')
ylabel('# Days to double cases')
title('Trend +Period - Green cleaned from periodic - Growth Factor %')
legend('Period for doubling - Prediciton Model','Location','west')

subplot(6,1,6)
temp = growth_factor;
plot(temp)
title('Exponent lambda')
set(gca,'Ylim',[-10 max(temp)*1.2]);
set(gca,'Xlim',[1 length(yfitNOP2)]);
xlabel('Days starting Jan 1st')


print_figure(formats,'Model3_fit')


ID=figure;
fig_size_paper=[1, 1, 20, 28];
fig_size_screen=[1, 1, 600, 800];
set(ID,'PaperType','A4');
set(ID,'PaperOrientation','portrait');
set(ID,'PaperUnits','centimeters');
set(ID,'PaperPosition',fig_size_paper);
set(ID,'Units','pixel');
set(ID,'Position',fig_size_screen);


semilogy(1:length(CumSumCases_Germany),cumsum(CumSumCases_Germany),'b')
title('Trend +Period - Green cleaned from periodic - Cumulative')
hold on
temp = cumsum(yfit);
semilogy(1:length(CumSumCases_Germany),temp,'r');
semilogy(1:length(yfitNOP2),cumsum(yfitNOP),'g');
semilogy(1:length(yfitNOP2),cumsum(yfitNOP),'m');
hold on
if use_splines==1
    plot(knots,temp(knots),'rx')
end
set(gca,'Ylim',[10 max([max(cumsum(CumSumCases_Germany)) max(temp)])*2])
set(gca,'Xlim',[45 length(yfitNOP2)]);
legend('Reported','Model Trend+Period.','Model Trend only','Prediciton Model','Location','west')
xlabel('Days starting Jan 1st')

print_figure(formats,'Model3Log_fitL')

ID=figure;
fig_size_paper=[1, 1, 20, 28];
fig_size_screen=[1, 1, 600, 800];
set(ID,'PaperType','A4');
set(ID,'PaperOrientation','portrait');
set(ID,'PaperUnits','centimeters');
set(ID,'PaperPosition',fig_size_paper);
set(ID,'Units','pixel');
set(ID,'Position',fig_size_screen);



%Model across LKs different growth factor same principle shape
%Periodic Component
%------------------------------------------------------------------------
%------------------------------------------------------------------------
%------------------------------------------------------------------------
Population    = Data.LK.Population;
Population    = squeeze(sum(Population,3));
LogPopulation = log(Population);
%LogPopulation = LogPopulation./max(max(LogPopulation));
if use_cum_sum_flag    == 1;
    CumSumCases_LK= cumsum(Data.Data_cases,3);
    CumSumDeath_LK= cumsum(Data.Data_death,3);
else
    CumSumCases_LK= (Data.Data_cases);
    CumSumDeath_LK= (Data.Data_death);
end

% Time_vec           = 1:size(CumSumCases_LK,3);
% BasisMAT_Trend     = zeros(Max_order_poly+1,size(CumSumCases_LK,3));
% for Order=0:Max_order_poly
%     BasisMAT_Trend(Order+1,:)=Time_vec.^(Order);
%     BasisMAT_Trend(Order+1,:)=BasisMAT_Trend(Order+1,:)./(max(BasisMAT_Trend(Order+1,:)));
% end

for IDXAge=1:size(CumSumDeath_LK,2)
    
    
    target_vec      =[];
    Design_Matrix   =[];
    for IDXLK=1:size(CumSumDeath_LK,1)
        Indiv_Growth    = ones(1,size(CumSumDeath_LK,3))*LogPopulation(IDXLK,IDXAge);
        act_cases       = squeeze(CumSumCases_LK(IDXLK,IDXAge,:));
        if sum(act_cases)~=0
            DesignMat_part  = [BasisMAT_Trend ; BasisMAT_Periodic];%BasisMAT_Trend(1:end,:);
            %DesignMat_part(1,:) = DesignMat_part(1,:);
            DesignMat_part2     =[DesignMat_part ; Indiv_Growth];
            target_vec      =[ target_vec  act_cases'];
            Design_Matrix   =[ Design_Matrix  DesignMat_part2];
        else
            warning('No cases')
        end
    end
    target_vec(find(target_vec)==0)=NaN;
    target_vec(find(target_vec<0))=0;
    
    IDX_Trend        = (1:size(BasisMAT_Trend,1));
    IDX_period       = (1:size(BasisMAT_Periodic,1))+size(BasisMAT_Trend,1);
    IDX_scale        = (size(BasisMAT_Periodic,1))+size(BasisMAT_Trend,1)+1;
    
    [beta,dev,stats] = glmfit(Design_Matrix' , target_vec,'poisson','constant','off');
    yfit = glmval(beta,Design_Matrix','log','constant','off');
    growth_factor    =  beta(IDX_Trend )'*BasisMAT_Trend(IDX_Trend ,:);
    periodic_factor  =  beta(IDX_period)'*BasisMAT_Periodic(:,:);
    periodic_cleaned =  beta(IDX_period)'*BasisMAT_PeriodicAv(:,:);
    Offset           =  beta(end);
    
    figure;
    for IDXP=1:3
        if IDXP==1
            logpp=8;
        end
        if IDXP==2
            logpp=10.5;
        end
        if IDXP==3
            logpp=13;
        end
        subplot(3,1,IDXP)        
        
        ndiv_Growth        = ones(1,size(BasisMAT_Periodicp,2))*logpp;
        DesignMat_part      = [BasisMAT_Trendp ; BasisMAT_Periodicp];%BasisMAT_Trend(1:end,:);
        DesignMat_partNOP      = [BasisMAT_Trendp ; BasisMAT_PeriodicAvp];%BasisMAT_Trend(1:end,:);
        DesignMat_part2     =[DesignMat_part ; ndiv_Growth];
        DesignMat_part3     =[DesignMat_partNOP ; ndiv_Growth];
        betat               = beta;
        betat(IDX_Component_End)= 0;
        yfitp               = glmval(beta,DesignMat_part2','log','constant','off');
        plot(1:length(yfitp ),yfitp ,'g')
        hold on
        yfitp               = glmval(beta,DesignMat_part3','log','constant','off');
        plot(1:length(yfitp ),yfitp ,'m')
    end
    

   
    
    target_vec2      =[]; 
    yfitLK           =[];
    yfitLKA          =[];
    Modelfit         =[];
    ModelfitLK       =[];
    ModelfitLKA      =[];
    target_vec2_cumsum       =[]; 
    yfit_cumsum              =[];
    yfitLK_cumsum            =[];
    yfitLKA_cumsum           =[];
    Modelfit_cumsum          =[];
    ModelfitLK_cumsum        =[];
    ModelfitLKA_cumsum       =[];
    for IDXLK=1:size(CumSumDeath_LK,1)
        Design_Matrix2   =[];
        DesignMat_part   =[];
        act_cases       = squeeze(CumSumCases_LK(IDXLK,IDXAge,:));
        if sum(act_cases)~=0
            Model3                  = growth_factor+periodic_factor +Offset*LogPopulation(IDXLK,IDXAge);
            Model3_cleaned          = growth_factor+periodic_cleaned +Offset*LogPopulation(IDXLK,IDXAge);;
            Model3_notcleaned       = growth_factor+periodic_factor +Offset*LogPopulation(IDXLK,IDXAge);
            DesignMat_part(1,:)     = growth_factor.*0+1;
          
            target_vec2         =[ act_cases'];
            target_vec2(find(target_vec2)==0)=NaN;
            target_vec2(find(target_vec2<0))=0;
            [beta2,dev,stats]   = glmfit(DesignMat_part' , target_vec2,'poisson','constant','off','offset',Model3);
            beta_scale(IDXLK)   = beta2;
            yfit2               = glmval(beta2,DesignMat_part','log','constant','off','offset',Model3');
            yfitLK              = [yfitLK  ; yfit2];
            target_vec2_cumsum  =[target_vec2_cumsum  cumsum(target_vec2)]; 
            yfitLK_cumsum       =[yfitLK_cumsum    cumsum(yfit2)'];
            
            DesignMat_part2     = DesignMat_part;            
            DesignMat_part2(2,:)= Model3;
            DesignMat_part2(3,:)= Model3.^2;
            [beta2A,dev,stats]   = glmfit(DesignMat_part2' , target_vec2,'poisson','constant','off');
            beta_scale2(IDXLK,:) = beta2A;
            yfit2A              = glmval(beta2A,DesignMat_part2','log','constant','off');
            yfitLKA             = [yfitLKA  ; yfit2A];
            yfitLKA_cumsum       =[yfitLKA_cumsum    cumsum(yfit2A)'];
            
            Indiv_Growth        = ones(1,size(CumSumDeath_LK,3))*LogPopulation(IDXLK,IDXAge);
            DesignMat_part      = [BasisMAT_Trend ; BasisMAT_Periodic];%BasisMAT_Trend(1:end,:);            
            DesignMat_part2     =[DesignMat_part ; Indiv_Growth]; 
            yfitt               = glmval(beta,DesignMat_part2','log','constant','off');
            yfit_cumsum         =[ yfit_cumsum  cumsum(yfitt)' ];
            Observed(IDXLK,:)       =target_vec2;
            Modelfit(IDXLK,:)       =yfitt;
            ModelfitLK(IDXLK,:)     =yfit2;
            ModelfitLKA(IDXLK,:)    =yfit2A;
            
            Observed_cumsum(IDXLK,:)       =cumsum(target_vec2);
            Modelfit_cumsum(IDXLK,:)       =cumsum(yfitt);
            ModelfitLK_cumsum(IDXLK,:)     =cumsum(yfit2);
            ModelfitLKA_cumsum(IDXLK,:)    =cumsum(yfit2A);
        else
            
        end
    end
    diff_fit =mean((Modelfit-Observed).^2,2);
    diff_fit_LK =mean((ModelfitLK-Observed).^2,2);
    diff_fit_LKA =mean((ModelfitLKA-Observed).^2,2);
    
    %diff_fit_cumsum =(mean((Modelfit_cumsum-Observed_cumsum).^2,2)).^.05;
    %diff_fit_LK_cumsum =(mean((ModelfitLK_cumsum-Observed_cumsum).^2,2)).^.05;
    %diff_fit_LKA_cumsum =(mean((ModelfitLKA_cumsum-Observed_cumsum).^2,2)).^.05;
    
    ID=figure;
    fig_size_paper=[1, 1, 20, 28];
    fig_size_screen=[1, 1, 600, 800];
    set(ID,'PaperType','A4');
    set(ID,'PaperOrientation','portrait');
    set(ID,'PaperUnits','centimeters');
    set(ID,'PaperPosition',fig_size_paper);
    set(ID,'Units','pixel');
    set(ID,'Position',fig_size_screen);

    plot(Model3_cleaned  )
    hold on
    plot(Model3_notcleaned,'g')
    legend('Model-All LKs same cleaned','Model-All LKs with Periodicity','Location','west')
    title('Log Model: lambda')
    set(gca,'Xlim',[1 length(CumSumCases_Germany)]);
    xlabel('Days starting Jan 1st')
    for IDXE =1:length(Historie)
        plot([Historie{IDXE}.day  Historie{IDXE}.day ],[min((Model3_notcleaned))*1.1 max(Model3_notcleaned)*1.1],'m')
        text(Historie{IDXE}.day,min((Model3_notcleaned)),Historie{IDXE}.event,'Rotation',90)
    end

    print_figure(formats,['Model3_LK_temporal_kernel_AgeGroupId' num2str(IDXAge) ])


    ID=figure;
    fig_size_paper=[1, 1, 20, 28];
    fig_size_screen=[1, 1, 600, 800];
    set(ID,'PaperType','A4');
    set(ID,'PaperOrientation','portrait');
    set(ID,'PaperUnits','centimeters');
    set(ID,'PaperPosition',fig_size_paper);
    set(ID,'Units','pixel');
    set(ID,'Position',fig_size_screen);  
    
    subplot(4,1,1)
    plot(target_vec2_cumsum)
    hold on
    plot(yfit_cumsum,'r-');
    plot(yfitLK_cumsum,'m-');
    plot(yfitLKA_cumsum,'g-');
    set(gca,'xlim',[1 length(yfitLKA_cumsum)]);
    legend('Reported','Model- Time kernel same - Scaled by log P','Model- Time kernel same - Scaled by LK','Model- Time kernel different','Location','west')
    xlabel('Landkreise')
    ylabel('mean E2')
    subplot(4,1,2)
    
    plot(diff_fit,'b')
    hold on 
    plot(diff_fit_LK,'g')
    plot(diff_fit_LKA,'r')
    legend('Av. E2 Mod. kernel same - Scaled by log Pop','Scaled kernel','different kernel ','Location','west') 
    set(gca,'xlim',[1 412]);    
    templ =diff_fit;
    [trash ID] = sort(templ,'descend');
    for IDXE =1:(10)        
        text(ID(IDXE)-4,0,Data.LK.names{ID(IDXE)},'Rotation',90)
    end
    xlabel('Landkreise')
    ylabel('mean E2')
    
    subplot(4,1,3)
    plot(diff_fit_LK,'g')
    legend('Av. E2 Mod. Scaled kernel and Reported','Location','west')     
    set(gca,'xlim',[1 412]);    
    templ =diff_fit_LK;
    [trash ID] = sort(templ,'descend');
     for IDXE =1:(10)        
        text(ID(IDXE)-4,0,Data.LK.names{ID(IDXE)},'Rotation',90)
    end
    xlabel('Landkreise')
    ylabel('mean E2')
    
    subplot(4,1,4)
    plot(diff_fit_LKA,'r')
    legend('Av. E2 Mod. different kernel and Reported','Location','west') 
    set(gca,'xlim',[1 412]);
    
    templ =diff_fit_LKA;
    [trash ID] = sort(templ,'descend');
    for IDXE =1:(10)        
        text(ID(IDXE)-4,0,Data.LK.names{ID(IDXE)},'Rotation',90)
    end
    xlabel('Landkreise')
    ylabel('mean E2')
    print_figure(formats,['Model3_LK_Error_AgeGroupId' num2str(IDXAge) ])
    
    
    
%     figure   
%     subplot(2,1,1)
%     plot(target_vec)
%     hold on
%     plot(yfit,'r-');
%     plot(yfitLK,'m-');
%     plot(yfitLKA,'g-');
%     subplot(2,1,2)
%     plot(diff_fit)
%     hold on
%     plot(diff_fit_LK,'r')
%     plot(diff_fit_LKA,'g')
%     
%     figure
%     subplot(2,3,1)
%     plot(yfit_cumsum, yfitLK_cumsum,'r.')
%     axis square
%     set(gca,'xlim',[0 max([max(yfit) max(yfitLK)]) ]);
%     set(gca,'ylim',[0 max([max(yfit) max(yfitLK)])]);
%     xlabel('scaled by log Population')
%     ylabel('Individual growth factor')
%     
%     subplot(2,3,3)
%     plot(yfitLKA_cumsum, yfitLK_cumsum,'r.')
%     axis square
%     set(gca,'xlim',[0 max([max(yfit) max(yfitLK)]) ]);
%     set(gca,'ylim',[0 max([max(yfit) max(yfitLK)])]);
%     xlabel('Individual growth factor adapted shape')
%     ylabel('Individual growth factor')
%     
%     subplot(2,3,5)
%     plot(yfit_cumsum, yfitLKA_cumsum,'r.')
%     axis square
%     set(gca,'xlim',[0 max([max(yfit) max(yfitLK)]) ]);
%     set(gca,'ylim',[0 max([max(yfit) max(yfitLK)])]);
%     xlabel('scaled by log Population')
%     ylabel('Individual growth factor adapted shape')
%     
%     subplot(2,3,2)
%     plot(target_vec2_cumsum, yfit_cumsum,'r.')
%     axis square
%     set(gca,'xlim',[0 max([max(yfit) max(target_vec)]) ]);
%     set(gca,'ylim',[0 max([max(yfit) max(target_vec)])]);
%     xlabel('real')
%     ylabel('Individual growth factor adapted shape')
%     
%     subplot(2,3,4)
%     plot(target_vec1_cumsum, yfitLK_cumsum,'r.')
%     axis square
%     set(gca,'xlim',[0 max([max(yfitLK) max(target_vec)]) ]);
%     set(gca,'ylim',[0 max([max(yfitLK) max(target_vec)])]);
%     xlabel('real')
%     ylabel('Individual growth factor adapted shape')
%     
%     subplot(2,3,6)
%     plot(target_vec2_cumsum, yfitLKA_cumsum,'r.')
%     axis square
%     set(gca,'xlim',[0 max([max(yfitLKA) max(target_vec)]) ]);
%     set(gca,'ylim',[0 max([max(yfitLKA) max(target_vec)])]);
%     xlabel('real')
%     ylabel('Individual growth factor adapted shape')
    
    
    
%     
%     
%     figure
%     subplot(2,1,1)
%     plot( target_vec)
%     title('Trend only')
%     hold on
%     plot(yfit,'r')
%     plot(yfitLK,'g')
%     subplot(2,1,2)
%     semilogy( target_vec)
%     hold on
%     semilogy(yfit,'r')
%     set(gca,'Ylim',[1 max(yfit)*1.5])
%     growth_factor    =  beta(1:size(BasisMAT_Trend,1))'*BasisMAT_Trend;
    
    
end

