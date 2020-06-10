import matplotlib
matplotlib.use('TkAgg')
from config import *
from plot_utils import *
from shared_utils import *
import pickle as pkl
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from pymc3.stats import quantiles

# def curves(use_interactions=True, use_report_delay=True, prediction_day=30, save_plot=False):
# Load only one county
def curves(model_i, start, n_weeks, county, save_plot=False):

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

    countyByName = OrderedDict(
        [('Düsseldorf', '05111'), ('Leipzig', '14713'), ('Nürnberg', '09564'), ('München', '09162')])
    # Hier dann das reinspeisen
    plot_county_names = {"covid19": [county]}

    # colors for curves
    C1 = "#D55E00"
    C2 = "#E69F00"
    C3 = "#0073CF"

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
        #height_ratios=[
        #    1,
        #    1,
        #    1.75]
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

    # Load our prediction samples
    res = load_pred_model_window(model_i, start, n_weeks)
    res_trend = load_pred_model_window(model_i, start, n_weeks, trend=True)
    #res_test = load_pred_by_i(disease, model_i)
   # print(res_train['y'].shape)
    #print(res_test['y'].shape)
    n_days = (day_p5 - start_day).days
    #print(res['y'].shape)
    prediction_samples = np.reshape(res['y'], (res['y'].shape[0], -1, 412)) 
    prediction_samples_trend = 20 * np.reshape(res_trend['y'], (res_trend['y'].shape[0],  -1, 412))
    
    #print(prediction_samples.shape)
    #print(target.index)
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

    '''
    map_ax = fig.add_subplot(grid[2, i])
    map_ax.set_position(grid[2, i].get_position(fig).translated(0, -0.05))
    map_ax.set_xlabel(
        "{}.{}.{}".format(
            prediction_mean.index[-5].day,
            prediction_mean.index[-5].month,
            prediction_mean.index[-5].year),
        fontsize=22)
    

    
    # plot the chloropleth map
    plot_counties(map_ax,
                counties,
                prediction_mean.iloc[-10].to_dict(),
                edgecolors=dict(zip(map(countyByName.get,
                                        plot_county_names[disease]),
                                    ["red"] * len(plot_county_names[disease]))),
                xlim=xlim,
                ylim=ylim,
                contourcolor="black",
                background=False,
                xticks=False,
                yticks=False,
                grid=False,
                frame=True,
                ylabel=False,
                xlabel=False,
                lw=2)

    map_ax.set_rasterized(True)
    '''
    for j, name in enumerate(plot_county_names[disease]):
        ax = fig.add_subplot(grid[j, i])

        county_id = countyByName[name]
    #     dates = [n.wednesday() for n in target.index.values]
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

        # plot 30week marker
        ax.axvline(dates[-5],ls='-', lw=2, c='cornflowerblue')
        ax.axvline(dates[-10],ls='--', lw=2, c='cornflowerblue')

        #ax.set_title(["campylobacteriosis" if disease == "campylobacter" else disease]
        #            [0] + "\n" + name if j == 0 else name, fontsize=22)
     
        #ax.set_xlabel("Time", fontsize=20)
        ax.tick_params(axis="both", direction='out',
                    size=6, labelsize=16, length=6
                    )
        ticks = [start_day+pd.Timedelta(days=i) for i in [0,5,10,15,20,25,30,35,40]]
        labels = ["{}.{}.{}".format(str(d)[8:10], str(d)[5:7], str(d)[:4]) for d in ticks]
        
        # ticks = ['2020-03-02','2020-03-12','2020-03-22','2020-04-01','2020-04-11','2020-04-21','2020-05-1','2020-05-11','2020-05-21']
        #labels = ['02.03.2020','12.03.2020','22.03.2020','01.04.2020','11.04.2020','21.04.2020','01.05.2020','11.05.2020','21.05.2020']
        plt.xticks(ticks,labels)
        #plt.xlabel(ticks)
        #ax.set_ylim([0,500])
        plt.setp(ax.get_xticklabels(), rotation=45)
        

        '''
        cent = np.array(counties[county_id]["shape"].centroid.coords[0])
        txt = map_ax.annotate(
            name,
            cent,
            cent + 0.5,
            color="white",
            arrowprops=dict(
                facecolor='white',
                shrink=0.001,
                headwidth=3),
            fontsize=26,
            fontweight='bold',
            fontname="Arial")
        txt.set_path_effects(
            [PathEffects.withStroke(linewidth=2, foreground='black')])
        '''

        ax.set_xlim([start_day,day_p5-pd.Timedelta(1)])
        #ax.set_ylim()
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

        
        '''
        _, target, _, _ = split_data(
            data,
            train_start=start_day,
            test_start=day_p5+pd.Timedelta(days=1),
            post_test=day_p5+pd.Timedelta(days=2))

        county_ids = target.columns
        print("HEYYEYYEY")
        print(target.index)
        print(ext_index)
        tspan = (target.index[0], target.index[-1])

    

        model = BaseModel(tspan,
                        counties,
                        ["../data/ia_effect_samples/{}_{}.pkl".format(disease,
                                                                        i) for i in range(100)],
                        include_ia=True,
                        include_report_delay=False,
                        include_demographics=True,
                        trend_poly_order=1,
                        periodic_poly_order=4)


        features = model.evaluate_features(
            ext_index, target.columns)

        trend_features = features["temporal_trend"].swaplevel(0, 1).loc[county_id]

        trace = load_trace_window(disease, model_i, start, n_weeks)
        trend_params = pm.trace_to_dataframe(trace, varnames=["W_t_t"])
        #trend_params = trend_params.swaplevel(0, 1).loc[county_id]

        iii = list(target.columns).index(county_id)
        #print(iii)
        ##print(trend_params)
        #print(trend_params.values.shape)
        trend_params = trend_params.values[:,2*iii:2*iii+2]
        #TT = trend_params.values.dot(trend_features.values.T)
        #print(trend_features.values.shape)

        #print(trend_params)
        #print(trend_features.values)
        TT = np.exp(np.dot(trend_params, trend_features.values.T))
        
        p_pred_trend = ax.plot_date(
                        dates,
                        TT.mean(axis=0),
                        "-",
                        color="green",
                        linewidth=2.0,
                        zorder=4)
        '''
        p_pred_trend = ax.plot_date(
                        dates,
                        prediction_mean_trend[county_id],
                        "-",
                        color="green",
                        linewidth=2.0,
                        zorder=4)
        
        



        if (i == 0) & (j == 0):
            ax.legend([p_real[0], p_pred[0], p_pred_trend[0], p_quant, p_quant2],
                    ["reported", "predicted", "trend", 
                        "25\%-75\% quantile", "5\%-95\% quantile"],
                    fontsize=12, loc="upper right")
        fig.text(0,
                1 + 0.025,
                r"$\textbf{"  + plot_county_names["covid19"][j]+ r"}$",
                fontsize=22,
                transform=ax.transAxes)
    #fig.text(0, 0.95, r"$\textbf{" + str(i + 1) + r"C}$",
    #        fontsize=22, transform=map_ax.transAxes)

    #fig.text(0.01, 1.6, "Reported/predicted infections",
    #        va='center', rotation='vertical', fontsize=16)
    



    

    if save_plot:
        plt.savefig("../figures/curve_{}_{}_{}.pdf".format(county,start,n_weeks))

    plt.close()
    return fig


if __name__ == "__main__": 

    import sys

    model_i = sys.argv[2]
    start = sys.argv[4]
    weeks = sys.argv[6]
    county = sys.argv[8]


    _ = curves(model_i,start, weeks, "Leipzig" ,save_plot=True)

