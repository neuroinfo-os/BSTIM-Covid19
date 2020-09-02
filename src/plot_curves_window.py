import matplotlib
#matplotlib.use('TkAgg')
from config import *
from plot_utils import *
from shared_utils import *
import pickle as pkl
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from pymc3.stats import quantiles
import os
import pandas as pd
from pathlib import Path

# def curves(use_interactions=True, use_report_delay=True, prediction_day=30, save_plot=False):
# Load only one county
def curves(start, county, n_weeks=3,  model_i=35, save_plot=False):

    with open('../data/counties/counties.pkl', "rb") as f:
        counties = pkl.load(f)
    start = int(start)
    n_weeks = int(n_weeks)
    model_i = int(model_i)
    # with open('../data/comparison.pkl', "rb") as f:
    #     best_model = pkl.load(f)

    # update to day and new limits!
    xlim = (5.5, 15.5)
    ylim = (47, 56) # <- 10 weeks

    #countyByName = OrderedDict(
    #    [('Düsseldorf', '05111'), ('Leipzig', '14713'), ('Nürnberg', '09564'), ('München', '09162')])
    countyByName = make_county_dict()
    # Hier dann das reinspeisen
    plot_county_names = {"covid19": [county]}
    start_day = pd.Timestamp('2020-01-28') + pd.Timedelta(days=start)
    year = str(start_day)[:4]
    month = str(start_day)[5:7]
    day = str(start_day)[8:10]
   # if os.path.exists("../figures/{}_{}_{}/curve_trend_{}.png".format(year, month, day,countyByName[county])):
    #    return

    day_folder_path = "../figures/{}_{}_{}".format(year, month, day)
    Path(day_folder_path).mkdir(parents=True, exist_ok=True)


    # check for metadata file:
    if not os.path.isfile("../figures/{}_{}_{}/metadata.csv".format(year, month, day)):
        ids = []
        for key in counties:
            ids.append(int(key))
        df = pd.DataFrame(data=ids, columns=["countyID"])
        df["probText"] = ""
        df.to_csv("../figures/{}_{}_{}/metadata.csv".format(year, month, day)) 



   # colors for curves
    #red
    C1 = "#D55E00"
    C2 = "#E69F00"
    #C3 = "#0073CF"
    #green
    C4 = "#188500"
    C5 = "#29c706"
    #C6 = "#0073CF"

    # quantiles we want to plot
    qs = [0.25, 0.50, 0.75]

    fig = plt.figure(figsize=(12, 6))
    grid = plt.GridSpec(
        1,
        1,
        top=0.9,
        bottom=0.2,
        left=0.07,
        right=0.97,
        hspace=0.25,
        wspace=0.15,
        )

    # for i, disease in enumerate(diseases):
    i = 0
    disease = "covid19"
    prediction_region = "germany"
    data = load_daily_data_n_weeks(start, n_weeks, disease, prediction_region, counties)

    start_day = pd.Timestamp('2020-01-28') + pd.Timedelta(days=start)
    i_start_day = 0
    day_0 = start_day + pd.Timedelta(days=n_weeks*7+5)
    day_m5 = day_0 - pd.Timedelta(days=5)
    day_p5 = day_0 + pd.Timedelta(days=5)

    _, target, _, _ = split_data(
        data,
        train_start=start_day,
        test_start=day_0,
        post_test=day_p5)

    county_ids = target.columns
    county_id = countyByName[county]

    ### SELECTION CRITERION ###
    #if np.count_non_zero(target[county_id]) < 7: #???
    #    stdd = 10
    #    gaussian = lambda x: np.exp( (-(x)**2) / (2* stdd**2) )
        


    # Load our prediction samples
    res = load_pred_model_window(model_i, start, n_weeks)
    res_trend = load_pred_model_window(model_i, start, n_weeks, trend=True)
    n_days = (day_p5 - start_day).days
    prediction_samples = np.reshape(res['y'], (res['y'].shape[0], -1, 412)) 
    prediction_samples_trend = np.reshape(res_trend['μ'], (res_trend['μ'].shape[0],  -1, 412))
    prediction_samples = prediction_samples[:,i_start_day:i_start_day+n_days,:]
    prediction_samples_trend = prediction_samples_trend[:,i_start_day:i_start_day+n_days,:]
    ext_index = pd.DatetimeIndex([d for d in target.index] + \
            [d for d in pd.date_range(target.index[-1]+timedelta(1),day_p5-timedelta(1))])

    # TODO: figure out where quantiles comes from and if its pymc3, how to replace it
    prediction_quantiles = quantiles(prediction_samples, (5, 25, 75, 95)) 

    prediction_mean = pd.DataFrame(
        data=np.mean(
            prediction_samples,
            axis=0),
        index=ext_index,
        columns=target.columns)
    prediction_q25 = pd.DataFrame(
        data=prediction_quantiles[25],
        index=ext_index,
        columns=target.columns)
    prediction_q75 = pd.DataFrame(
        data=prediction_quantiles[75],
        index=ext_index,
        columns=target.columns)
    prediction_q5 = pd.DataFrame(
        data=prediction_quantiles[5],
        index=ext_index,
        columns=target.columns)
    prediction_q95 = pd.DataFrame(
        data=prediction_quantiles[95],
        index=ext_index,
        columns=target.columns)


    prediction_mean_trend = pd.DataFrame(
        data=np.mean(
            prediction_samples_trend,
            axis=0),
        index=ext_index,
        columns=target.columns)

    # Unnecessary for-loop
    for j, name in enumerate(plot_county_names[disease]):
        ax = fig.add_subplot(grid[j, i])

        county_id = countyByName[name]
        dates = [pd.Timestamp(day) for day in ext_index]
        days = [ (day - min(dates)).days for day in dates]

        # plot our predictions w/ quartiles
        p_pred = ax.plot_date(
            dates,
            prediction_mean[county_id],
            "-",
            color=C1,
            linewidth=2.0,
            zorder=4)
        # plot our predictions w/ quartiles
        
        p_quant = ax.fill_between(
            dates,
            prediction_q25[county_id],
            prediction_q75[county_id],
            facecolor=C2,
            alpha=0.5,
            zorder=1)
        ax.plot_date(
            dates,
            prediction_q25[county_id],
            ":",
            color=C2,
            linewidth=2.0,
            zorder=3)
        ax.plot_date(
            dates,
            prediction_q75[county_id],
            ":",
            color=C2,
            linewidth=2.0,
            zorder=3)


        # plot ground truth
        p_real = ax.plot_date(dates[:-5], target[county_id], "k.")
        print(dates[-5]-pd.Timedelta(12, unit='h'))
        # plot 30week marker
        ax.axvline(dates[-5]-pd.Timedelta(12,unit='h'),ls='-', lw=2, c='dodgerblue')
        ax.axvline(dates[-10]-pd.Timedelta(12,unit='h'),ls='--', lw=2, c='lightskyblue')

     
        ax.set_ylabel("Fallzahlen/Tag nach Meldedatum", fontsize=16)
        ax.tick_params(axis="both", direction='out',
                    size=6, labelsize=16, length=6
                    )
        ticks = [start_day+pd.Timedelta(days=i) for i in [0,5,10,15,20,25,30,35,40]]
        labels = ["{}.{}.{}".format(str(d)[8:10], str(d)[5:7], str(d)[:4]) for d in ticks]
        
        plt.xticks(ticks,labels)        
        #new_ticks = plt.get_xtickslabels()
        plt.setp(ax.get_xticklabels()[-4], color="red")
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        ax.autoscale(True)
        p_quant2 = ax.fill_between(
            dates,
            prediction_q5[county_id],
            prediction_q95[county_id],
            facecolor=C2,
            alpha=0.25,
            zorder=0)
        ax.plot_date(dates, prediction_q5[county_id], ":",
                    color=C2, alpha=0.5, linewidth=2.0, zorder=1)
        ax.plot_date(dates, prediction_q95[county_id], ":",
                    color=C2, alpha=0.5, linewidth=2.0, zorder=1)

        # Plot the trend.
        '''
        p_pred_trend = ax.plot_date(
                        dates,
                        prediction_mean_trend[county_id],
                        "-",
                        color="green",
                        linewidth=2.0,
                        zorder=4)
        '''
        # Compute probability of increase/decreas
        i_county =  county_ids.get_loc(county_id)
        trace = load_trace_window(disease, model_i, start, n_weeks)
        trend_params = pm.trace_to_dataframe(trace, varnames=["W_t_t"]).values
        trend_w2 = np.reshape(trend_params, newshape=(1000,412,2))[:,i_county,1]
        prob2 = np.mean(trend_w2>0)
        
        # Set axis limits.
        ylimmax = max(3*(target[county_id]).max(),10)
        ax.set_ylim([-(1/30)*ylimmax,ylimmax])
        ax.set_xlim([start_day,day_p5-pd.Timedelta(days=1)])

        if (i == 0) & (j == 0):
            ax.legend([p_real[0], p_pred[0], p_quant, p_quant2],
                    ["Daten RKI", "Modell",  
                        "25\%-75\%-Quantil", "5\%-95\%-Quantil"],
                    fontsize=16, loc="upper left")

        # Not perfectly positioned.
        print("uheufbhwio")
        print(ax.get_xticks()[-5])
        print(ax.get_ylim()[1])
        pos1 = tuple(ax.transData.transform((ax.get_xticks()[-3], ax.get_ylim()[1])))
        pos1 = (ax.get_xticks()[-5], ax.get_ylim()[1]) 
        print(pos1)
        fontsize_bluebox = 18
        fig.text(ax.get_xticks()[-5]+0.65, ax.get_ylim()[1],"Nowcast",ha="left",va="top",fontsize=fontsize_bluebox,bbox=dict(facecolor='lightskyblue', boxstyle='rarrow'), transform=ax.transData)
        # fig.text(pos1[0]/1200, pos1[1]/600,"Nowcast",fontsize=fontsize_bluebox,bbox=dict(facecolor='cornflowerblue'))
        fig.text(ax.get_xticks()[-4]+0.65, ax.get_ylim()[1],"Forecast",ha="left", va="top",fontsize=fontsize_bluebox,bbox=dict(facecolor='dodgerblue', boxstyle='rarrow'), transform=ax.transData)

        ''' 
        fig.text(0,
                1 + 0.025,
                r"$\textbf{"  + plot_county_names["covid19"][j]+ r"}$",
                fontsize=22,
                transform=ax.transAxes)
        '''
       
        #plt.yticks(ax.get_yticks()[:-1], ax.get_yticklabels()[:-1]) 
        # Store text in csv.
        #fontsize_probtext = 14
        if prob2 >=0.5:
            #fig.text(0.865, 0.685, "Die Fallzahlen \n werden mit einer \n Wahrscheinlichkeit \n von {:2.1f}\% steigen.".format(prob2*100), fontsize=fontsize_probtext,bbox=dict(facecolor='white'))
            probText = "Die Fallzahlen werden mit einer Wahrscheinlichkeit von {:2.1f}\% steigen.".format(prob2*100)
        else:
            probText = "Die Fallzahlen werden mit einer Wahrscheinlichkeit von {:2.1f}\% fallen.".format(100-prob2*100)
            #fig.text(0.865, 0.685, "Die Fallzahlen \n werden mit einer \n Wahrscheinlichkeit \n von {:2.1f}\% fallen.".format(100-prob2*100), fontsize=fontsize_probtext ,bbox=dict(facecolor='white'))
        
        print(county_id)
        df = pd.read_csv("../figures/{}_{}_{}/metadata.csv".format(year, month, day), index_col=0)
        county_ix = df["countyID"][df["countyID"]==int(county_id)].index[0]
        if prob2 >=0.5:
            probVal = prob2*100
        else:
            probVal = -(100-prob2*100)
        df.iloc[county_ix, 1] = probVal#= probText
        df.to_csv("../figures/{}_{}_{}/metadata.csv".format(year, month, day))
        print(probVal)

  
        plt.tight_layout()
    if save_plot:
        year = str(start_day)[:4]
        month = str(start_day)[5:7]
        day = str(start_day)[8:10]
        day_folder_path = "../figures/{}_{}_{}".format(year, month, day)
        Path(day_folder_path).mkdir(parents=True, exist_ok=True)
      
        plt.savefig("../figures/{}_{}_{}/curve_{}.png".format(year, month, day,countyByName[county]), dpi=200)

    plt.close()
    return fig


if __name__ == "__main__": 

    import sys

    start = sys.argv[2]
    county = sys.argv[4]

    _ = curves(start, county ,save_plot=True)

