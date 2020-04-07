clear all;
close all;
days(1) = 31;
days(2) = 29;
days(3) = 31;
days(4) = 30;
days(5) = 31;
Max_order_poly      = 3;
Max_order_polyEnd   = 3;
Max_order_Peri      = 5;
nr_knots            = 3;
end_days            = 3;
use_cum_sum_flag    = 0;

dbstop if error
cd('D:\Dropbox\Covid_19_FIT_Modelle')
load('PrepocessedData3.mat')

max_Period      = days(1)+days(2)+days(3)+1;


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


knots        = round(linspace(1, (size(Data.Data_cases,3)-end_days) , nr_knots));  
knots        = [knots round((size(Data.Data_cases,3)-end_days/2))];
knots        = unique(knots);
knots        = augknt(knots, 4);


Design_MatrixSplines = spcol(knots, 4, 1:size(Data.Data_cases,3))';

plot(Design_MatrixSplines)


CumSumCases_Germany(find(CumSumCases_Germany)==0)=NaN;
Time_vec           = [1:(size(Data.Data_cases,3)-end_days) zeros(1,end_days)];
Time_vec2           =[zeros(1,size(Data.Data_cases,3)-end_days) (1:end_days)];

BasisMAT_Trend1     = zeros(Max_order_poly+1,size(Data.Data_cases,3));
for Order=0:Max_order_poly
    BasisMAT_Trend1(Order+1,:)=Time_vec.^(Order);
    BasisMAT_Trend1(Order+1,:)=BasisMAT_Trend1(Order+1,:)./(max(BasisMAT_Trend1(Order+1,:)));
end
for Order=1:Max_order_polyEnd
    BasisMAT_Trend2(Order,:)=Time_vec2.^(Order);
    BasisMAT_Trend2(Order,:)=BasisMAT_Trend2(Order,:)./(max(BasisMAT_Trend2(Order,:)));
end

%BasisMAT_Trend =[BasisMAT_Trend1 ; BasisMAT_Trend2];
BasisMAT_Trend = Design_MatrixSplines;
%plot(BasisMAT_Trend');
Design_Matrix = BasisMAT_Trend;
[beta,dev,stats] = glmfit(Design_Matrix' , CumSumCases_Germany,'poisson','constant','off');
yfit = glmval(beta,Design_Matrix','log','constant','off');
growth_factor    =  beta'*Design_Matrix;
%growth_factorend =  beta(5:7)'*Design_Matrix(5:7,:);

figure
subplot(2,1,1)
plot(1:length(CumSumCases_Germany),CumSumCases_Germany)
title('Trend only')
hold on
plot(1:length(CumSumCases_Germany),yfit,'r')
subplot(2,1,2)
semilogy(1:length(CumSumCases_Germany),CumSumCases_Germany)
hold on
semilogy(1:length(CumSumCases_Germany),yfit,'r')
set(gca,'Ylim',[1 max(yfit)*1.5])

if use_cum_sum_flag    == 0;
    figure
    subplot(2,1,1)
    plot(1:length(CumSumCases_Germany),cumsum(CumSumCases_Germany))
    title('Trend only - Cumulative is computer after fitting')
    hold on
    plot(1:length(CumSumCases_Germany),cumsum(yfit),'r')
    subplot(2,1,2)
    semilogy(1:length(CumSumCases_Germany),cumsum(CumSumCases_Germany))
    hold on
    semilogy(1:length(CumSumCases_Germany),cumsum(yfit),'r');
    set(gca,'Ylim',[1 max([max(yfit) max(cumsum(CumSumCases_Germany))])*2])
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


for Order=1:Max_order_Peri
    BasisMAT_Periodic(Order,:)=(mod(Time_vec,7)+1).^(Order);
    BasisMAT_Periodic(Order,:)=BasisMAT_Periodic(Order,:)./(max(BasisMAT_Periodic(Order,:)));
end
%plot(BasisMAT_Periodic');

Design_Matrix = [BasisMAT_Trend ; BasisMAT_Periodic];
[beta,dev,stats] = glmfit(Design_Matrix' , CumSumCases_Germany,'poisson','constant','off');
yfit = glmval(beta,Design_Matrix','log','constant','off');
Periodic_component  =  beta((Max_order_poly+2):end)'*Design_Matrix((Max_order_poly+2):end,:);
growth_factor       =  beta'*Design_Matrix;

figure
subplot(2,1,1)
plot(1:length(CumSumCases_Germany),CumSumCases_Germany)
title('Trend + Periodic')
hold on
plot(1:length(CumSumCases_Germany),yfit,'r')
subplot(2,1,2)
semilogy(1:length(CumSumCases_Germany),CumSumCases_Germany)
hold on
semilogy(1:length(CumSumCases_Germany),yfit,'r')
set(gca,'Ylim',[1 max(yfit)*1.5])

if use_cum_sum_flag    == 0;
    figure
    subplot(2,1,1)
    plot(1:length(CumSumCases_Germany),cumsum(CumSumCases_Germany))
    title('Trend only - Cumulative is computer after fitting')
    hold on
    plot(1:length(CumSumCases_Germany),cumsum(yfit),'r')
    subplot(2,1,2)
    semilogy(1:length(CumSumCases_Germany),cumsum(CumSumCases_Germany))
    hold on
    semilogy(1:length(CumSumCases_Germany),cumsum(yfit),'r');
    set(gca,'Ylim',[1 max(yfit)*1.5])
end





%Model across LKs same growth factor
%Periodic Component
%------------------------------------------------------------------------
%------------------------------------------------------------------------
%------------------------------------------------------------------------
Population    = Data.LK.Population;
Population    = squeeze(sum(Population,3));
LogPopulation = log(Population);
LogPopulation = LogPopulation./max(max(LogPopulation));
if use_cum_sum_flag    == 1;
    CumSumCases_LK= cumsum(Data.Data_cases,3);
    CumSumDeath_LK= cumsum(Data.Data_death,3);
else
    CumSumCases_LK= (Data.Data_cases);
    CumSumDeath_LK= (Data.Data_death);
end

Time_vec           = 1:size(CumSumCases_LK,3);
BasisMAT_Trend     = zeros(Max_order_poly+1,size(CumSumCases_LK,3));
for Order=0:Max_order_poly
    BasisMAT_Trend(Order+1,:)=Time_vec.^(Order);
    BasisMAT_Trend(Order+1,:)=BasisMAT_Trend(Order+1,:)./(max(BasisMAT_Trend(Order+1,:)));
end

for IDXAge=3:size(CumSumDeath_LK,2)
    
    target_vec      =[];
    Design_Matrix   =[];
    
    for IDXLK=1:size(CumSumDeath_LK,1)
        Indiv_Growth    = zeros(size(CumSumDeath_LK,1),size(CumSumDeath_LK,3));
        Indiv_Growth(IDXLK,:)    =1;
        act_cases       = squeeze(CumSumCases_LK(IDXLK,IDXAge,:));
        if sum(act_cases)~=0
            DesignMat_part  = BasisMAT_Trend(2:end,:);
            DesignMat_part(1,:) = DesignMat_part(1,:);
            DesignMat_part2     =[DesignMat_part ; Indiv_Growth];
            target_vec      =[ target_vec  act_cases'];
            Design_Matrix   =[ Design_Matrix  DesignMat_part2];
        else 
            warning('No cases')
        end
    end
    target_vec(find(target_vec)==0)=NaN;
    target_vec(find(target_vec<0))=0;
    
    [beta,dev,stats] = glmfit(Design_Matrix' , target_vec,'poisson','constant','off');
    yfit = glmval(beta,Design_Matrix','log','constant','off');
    growth_factor    =  beta(2:size(BasisMAT_Trend,1))'*BasisMAT_Trend(2:end,:);
    
    figure
    subplot(2,1,1)
    plot( target_vec)
    title('Trend only')
    hold on
    plot(yfit,'r')
    subplot(2,1,2)
    semilogy( target_vec)
    hold on
    semilogy(yfit,'r')
    
    figure
    subplot(2,1,1)
    plot(yfit,target_vec,'.')
    hold on
    xlabel('model')
    ylabel('real')
    subplot(2,1,2)
    
    temp = exp(beta((length(beta)-411):length(beta)));
    temp(find(temp>0.1))=median(temp);
    plot(temp./max(temp),LogPopulation(:,IDXAge),'r.')
    axis square
    set(gca,'xlim',[0 1]);
    set(gca,'ylim',[0 1]);
    xlabel('model coef')
    ylabel('Real pop')
    
    target_vec2      =[];
    Design_Matrix2   =[];
    DesignMat_part   =[];
    yfitLK           =[];
    for IDXLK=1:size(CumSumDeath_LK,1)
        act_cases       = squeeze(CumSumCases_LK(IDXLK,IDXAge,:));
        if sum(act_cases)~=0
            DesignMat_part(1,:)  = growth_factor.*0+1;
            DesignMat_part(2,:)  = growth_factor;
            target_vec2      =[ act_cases'];
            target_vec2(find(target_vec2)==0)=NaN;
            target_vec2(find(target_vec2<0))=0;
            [beta2,dev,stats] = glmfit(DesignMat_part' , target_vec2,'poisson','constant','off');
            yfit2 = glmval(beta2,DesignMat_part','log','constant','off');
            yfitLK           =[yfitLK  ; yfit2];
        else
            
        end
    end
    
    figure
    plot(yfit, yfitLK,'r.')
    axis square
    set(gca,'xlim',[0 max([max(yfit) max(yfitLK)]) ]);
    set(gca,'ylim',[0 max([max(yfit) max(yfitLK)])]);
    xlabel('Big Model - all LK one Component')
    ylabel('Fixedd Component scaled')
    
    figure
    subplot(2,1,1)
    plot( target_vec)
    title('Trend only')
    hold on
    plot(yfit,'r')
    plot(yfitLK,'g')
    subplot(2,1,2)
    semilogy( target_vec)
    hold on
    semilogy(yfit,'r')
    set(gca,'Ylim',[1 max(yfit)*1.5])
    growth_factor    =  beta(1:size(BasisMAT_Trend,1))'*BasisMAT_Trend;
    
    
end

