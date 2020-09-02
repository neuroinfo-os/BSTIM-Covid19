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
import pandas as pd
from pathlib import Path
# def curves(use_interactions=True, use_report_delay=True, prediction_day=30, save_plot=False):
# Load only one county
def curves(start, n_weeks=3, model_i=35,save_plot=False):

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
    # Hier dann das reinspeisen
    # Fake for loop to work
    #plot_county_names = {"covid19": ["Leipzig"]}

    # colors for curves
    C1 = "#D55E00"
    C2 = "#E69F00"
    C3 = "#0073CF"

    # quantiles we want to plot
    qs = [0.25, 0.50, 0.75]

    fig = plt.figure(figsize=(6, 8))
    grid = plt.GridSpec(
        1,
        1,
        #top=0.99,
        #bottom=0.01,
        #left=0.1,
        #right=0.9,
        #hspace=0.02,
        #wspace=0.15,
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
    n_days = (day_p5 - start_day).days
    prediction_samples = np.reshape(res['y'], (res['y'].shape[0], -1, 412))
    prediction_samples = prediction_samples[:,i_start_day:i_start_day+n_days,:]
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

    
    map_ax = fig.add_subplot(grid[0, i])
    #map_ax.set_position(grid[0, i].get_position(fig).translated(0, -0.05))
  
     
    # relativize prediction mean.
    map_vals = prediction_mean.iloc[-10]
    map_rki = data.iloc[-1].values.astype('float64')
    map_keys = []
    # ik= 0
    for (ik, (key, _)) in enumerate(counties.items()):
        n_people = counties[key]['demographics'][('total',2018)]
        map_vals[ik] = (map_vals[ik] / n_people) * 100000
        map_rki[ik] = (map_rki[ik] / n_people) * 100000
        # ik = ik+1
        map_keys.append(key)

    map_df = pd.DataFrame(index=None)
    map_df["countyID"] = map_keys
    map_df["newInf100k"] = list(map_vals)
    map_df["newInf100k_RKI"] = list(map_rki)
    
    # plot the chloropleth map
    plot_counties(map_ax,
                counties,
                map_vals.to_dict(),
                #prediction_mean.iloc[-10].to_dict(),
                #edgecolors=dict(zip(map(countyByName.get,
                #                        plot_county_names[disease]),
                #                    ["red"] * len(plot_county_names[disease]))),
                edgecolors=None,
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
    #plt.colorbar()
    map_ax.set_rasterized(True)
    
    '''
    for j, name in enumerate(plot_county_names[disease]):
        ax = fig.add_subplot(grid[j, i])

        county_id = countyByName[name]
    #     dates = [n.wednesday() for n in target.index.values]
        dates = [pd.Timestamp(day) for day in ext_index]
        days = [ (day - min(dates)).days for day in dates]
    '''

    fig.text(0.71,0.17,"Neuinfektionen \n pro 100.000 \n Einwohner", fontsize=14, color=[0.3,0.3,0.3])


    if save_plot:
        year = str(start_day)[:4]
        month = str(start_day)[5:7]
        day = str(start_day)[8:10]
        day_folder_path = "../figures/{}_{}_{}".format(year, month, day)
        Path(day_folder_path).mkdir(parents=True, exist_ok=True)

        plt.savefig("../figures/{}_{}_{}/map.png".format(year, month, day), dpi=300)
        map_df.to_csv("../figures/{}_{}_{}/map.csv".format(year, month, day))
    plt.close()
    return fig


if __name__ == "__main__": 

    import sys    
    start = sys.argv[2]
    _ = curves(start ,save_plot=True)

