import matplotlib
matplotlib.use('TkAgg')
import pymc3 as pm
import pandas as pd
import matplotlib
import numpy as np
import pickle as pkl
import datetime
from BaseModel import BaseModel
import isoweek
from matplotlib import rc
from shared_utils import *
from pymc3.stats import quantiles
from matplotlib import pyplot as plt
from config import * # <-- to select the right model

# from pandas import register_matplotlib_converters
# register_matplotlib_converters() # the fk python

def temporal_contribution(model_i=15, combinations=combinations, save_plot=False):

    use_ia, use_report_delay, use_demographics, trend_order, periodic_order = combinations[model_i]

    plt.style.use('ggplot')

    with open('../data/counties/counties.pkl', "rb") as f:
        county_info = pkl.load(f)

    C1 = "#D55E00"
    C2 = "#E69F00"
    C3 = C2  # "#808080"

    if use_report_delay:
        fig = plt.figure(figsize=(25, 10))
        grid = plt.GridSpec(4, 1, top=0.93, bottom=0.12,
                            left=0.11, right=0.97, hspace=0.28, wspace=0.30)
    else:
        fig = plt.figure(figsize=(16, 10))
        grid = plt.GridSpec(3, 1, top=0.93, bottom=0.12,
                            left=0.11, right=0.97, hspace=0.28, wspace=0.30)


    disease = "covid19"
    prediction_region = "germany"


    data = load_daily_data(disease, prediction_region, county_info)
    first_day = pd.Timestamp('2020-04-01')
    last_day = data.index.max()

    _, target_train, _, _ = split_data(
        data,
        train_start=first_day,
        test_start=last_day - pd.Timedelta(days=1),
        post_test=last_day + pd.Timedelta(days=1)
    )

    tspan = (target_train.index[0], target_train.index[-1])

    model = BaseModel(tspan,
                      county_info,
                      ["../data/ia_effect_samples/{}_{}.pkl".format(disease,
                                                                    i) for i in range(100)],
                      include_ia=use_ia,
                      include_report_delay=use_report_delay,
                      include_demographics=True,
                      trend_poly_order=4,
                      periodic_poly_order=4)


    features = model.evaluate_features(
        target_train.index, target_train.columns)

    trend_features = features["temporal_trend"].swaplevel(0, 1).loc["09162"]
    periodic_features = features["temporal_seasonal"].swaplevel(0, 1).loc["09162"]
    #t_all = t_all_b if disease == "borreliosis" else t_all_cr

    trace = load_final_trace()
    trend_params = pm.trace_to_dataframe(trace, varnames=["W_t_t"])
    periodic_params = pm.trace_to_dataframe(trace, varnames=["W_t_s"])

    TT = trend_params.values.dot(trend_features.values.T)
    TP = periodic_params.values.dot(periodic_features.values.T)
    TTP = TT + TP

    # add report delay if used
    #if use_report_delay:
    #    delay_features = features["temporal_report_delay"].swaplevel(0,1).loc["09162"]
    #    delay_params = pm.trace_to_dataframe(trace,varnames=["W_t_d"])
    #    TD =delay_params.values.dot(delay_features.values.T)

     #   TTP += TD
     #   TD_quantiles = quantiles(TD, (25, 75))


    TT_quantiles = quantiles(TT, (25, 75))
    TP_quantiles = quantiles(TP, (25, 75))
    TTP_quantiles = quantiles(TTP, (2.5,25, 75,97.5))

    dates = [pd.Timestamp(day) for day in target_train.index.values]
    days = [ (day - min(dates)).days for day in dates]



    # Temporal trend+periodic effect
    if use_report_delay:
        ax_tp = fig.add_subplot(grid[0, 0])
    else:
        ax_tp = fig.add_subplot(grid[0, 0])

    ax_tp.fill_between(days, np.exp(TTP_quantiles[25]), np.exp(
        TTP_quantiles[75]), alpha=0.5, zorder=1, facecolor=C1)
    ax_tp.plot(days, np.exp(TTP.mean(axis=0)),
                    "-", color=C1, lw=2, zorder=5)
    ax_tp.plot(days, np.exp(
        TTP_quantiles[25]), "-", color=C2, lw=2, zorder=3)
    ax_tp.plot(days, np.exp(
        TTP_quantiles[75]), "-", color=C2, lw=2, zorder=3)
    ax_tp.plot(days, np.exp(
        TTP_quantiles[2.5]), "--", color=C2, lw=2, zorder=3)
    ax_tp.plot(days, np.exp(
        TTP_quantiles[97.5]), "--", color=C2, lw=2, zorder=3)
    #ax_tp.plot(days, np.exp(TTP[:25, :].T),
    #                "--", color=C3, lw=1, alpha=0.5, zorder=2)

    ax_tp.tick_params(axis="x", rotation=45)

    # Temporal trend effect
    ax_t = fig.add_subplot(grid[1, 0], sharex=ax_tp)

    #ax_t.fill_between(days, np.exp(TT_quantiles[25]), np.exp(
    #    TT_quantiles[75]), alpha=0.5, zorder=1, facecolor=C1)
    ax_t.plot(days, np.exp(TT.mean(axis=0)),
                "-", color=C1, lw=2, zorder=5)
    #ax_t.plot(days, np.exp(
    #    TT_quantiles[25]), "-", color=C2, lw=2, zorder=3)
    #ax_t.plot(days, np.exp(
    #    TT_quantiles[75]), "-", color=C2, lw=2, zorder=3)
    #ax_t.plot(days, np.exp(TT[:25, :].T),
    #            "--", color=C3, lw=1, alpha=0.5, zorder=2)
    

    ax_t.tick_params(axis="x", rotation=45)
    ax_t.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    # Temporal periodic effect
    ax_p = fig.add_subplot(grid[2, 0], sharex=ax_tp)

    #ax_p.fill_between(days, np.exp(TP_quantiles[25]), np.exp(
     #   TP_quantiles[75]), alpha=0.5, zorder=1, facecolor=C1)
    ax_p.plot(days, np.exp(TP.mean(axis=0)),
                "-", color=C1, lw=2, zorder=5)
    ax_p.set_ylim([-0.0001,0.001])
    #ax_p.plot(days, np.exp(
      #  TP_quantiles[25]), "-", color=C2, lw=2, zorder=3)
    #ax_p.plot(days, np.exp(
      #  TP_quantiles[75]), "-", color=C2, lw=2, zorder=3)
    #ax_p.plot(days, np.exp(TP[:25, :].T),
      #          "--", color=C3, lw=1, alpha=0.5, zorder=2)

    ax_p.tick_params(axis="x", rotation=45)
    ax_p.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    ticks = ['2020-03-02','2020-03-12','2020-03-22','2020-04-01','2020-04-11','2020-04-21','2020-05-1','2020-05-11','2020-05-21']
    labels = ['02.03.2020','12.03.2020','22.03.2020','01.04.2020','11.04.2020','21.04.2020','01.05.2020','11.05.2020','21.05.2020']
    

    


    

    #if use_report_delay:
    #    ax_td = fig.add_subplot(grid[2, 0], sharex=ax_p)
#
#        ax_td.fill_between(days, np.exp(TD_quantiles[25]), np.exp(
#            TD_quantiles[75]), alpha=0.5, zorder=1, facecolor=C1)
#        ax_td.plot(days, np.exp(TD.mean(axis=0)),
#                    "-", color=C1, lw=2, zorder=5)
#        ax_td.plot(days, np.exp(
#            TD_quantiles[25]), "-", color=C2, lw=2, zorder=3)
#        ax_td.plot(days, np.exp(
#            TD_quantiles[75]), "-", color=C2, lw=2, zorder=3)
#        ax_td.plot(days, np.exp(TD[:25, :].T),
#                    "--", color=C3, lw=1, alpha=0.5, zorder=2)

#        ax_td.tick_params(axis="x", rotation=45)

    #ax_tp.set_title("campylob." if disease ==
    #            "campylobacter" else disease, fontsize=22)
    ax_p.set_xlabel("time [days]", fontsize=22)

    ax_p.set_ylabel("periodic\ncontribution", fontsize=22)
    ax_t.set_ylabel("trend\ncontribution", fontsize=22)
    ax_tp.set_ylabel("combined\ncontribution", fontsize=22)

    #if use_report_delay:
    #    ax_td.set_ylabel("r.delay\ncontribution", fontsize=22)

    ax_t.set_xlim(days[0], days[-1])
    ax_t.tick_params(labelbottom=False, labelleft=True, labelsize=18, length=6)
    ax_p.tick_params(labelbottom=True, labelleft=True, labelsize=18, length=6)
    ax_tp.tick_params(labelbottom=False, labelleft=True, labelsize=18, length=6)

    #ax_p.set_xticks(ticks)#,labels)

    if save_plot:
        fig.savefig("../figures/temporal_contribution_{}.pdf".format(model_i))

    #return fig

if __name__ == "__main__":

    _ = temporal_contribution(15, combinations,save_plot=True)
