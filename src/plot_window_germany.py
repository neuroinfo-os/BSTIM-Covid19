import matplotlib
#matplotlib.use('TkAgg')
from config import *
from plot_utils import *
from shared_utils import *
import pickle as pkl
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
# def curves(use_interactions=True, use_report_delay=True, prediction_day=30, save_plot=False):
# Load only one county
def curves(start, n_weeks=3, model_i=35,save_csv=False):

    with open('../data/counties/counties.pkl', "rb") as f:
        counties = pkl.load(f)
    start = int(start)
    n_weeks = int(n_weeks)
    model_i = int(model_i)
    
    disease = "covid19"
    prediction_region = "germany"
    data = load_daily_data_n_weeks(start, n_weeks, disease, prediction_region, counties)

    start_day = pd.Timestamp('2020-01-28') + pd.Timedelta(days=start)
    day_0 = start_day + pd.Timedelta(days=n_weeks*7+5)
    day_p5 = day_0 + pd.Timedelta(days=5)


    _, target, _, _ = split_data(
        data,
        train_start=start_day,
        test_start=day_0,
        post_test=day_p5)

    # Load our prediction samples
    res = load_pred_model_window(model_i, start, n_weeks, trend=True)
    prediction_samples = np.reshape(res['μ'], (res['μ'].shape[0], -1, 412))
    #prediction_samples = prediction_samples[:,i_start_day:i_start_day+n_days,:]
    ext_index = pd.DatetimeIndex([d for d in target.index] + \
            [d for d in pd.date_range(target.index[-1]+timedelta(1),day_p5-timedelta(1))])

    prediction_mean = pd.DataFrame(
        data=np.mean(
            prediction_samples,
            axis=0),
        index=ext_index,
        columns=target.columns)

    # relativize prediction mean.
    ref_date = target.iloc[-1].name # this comes out of a function that splits the data for us!
#     map_vals = prediction_mean.iloc[-10] # THIS IF OFF!
#     map_rki = target.iloc[-1]
    nowcast_vals = prediction_mean.loc[prediction_mean.index == ref_date]
    rki_vals = target.iloc[-1] # this is the same as data[data.index==ref_date]
    #SORRY QUICK AND DIRTY FIXES.
    map_nowcast = []
    map_rki = []
    map_keys = []
    for (ik, (key, _)) in enumerate(counties.items()):
        n_people = counties[key]['demographics'][('total',2018)]
        map_nowcast.append(nowcast_vals[key].item() / n_people * 100000)
        map_rki.append((rki_vals[key].item() / n_people) * 100000)
        map_keys.append(key)

    map_df = pd.DataFrame(index=None)
    map_df["countyID"] = map_keys
    map_df["newInf100k"] = map_nowcast
    map_df["newInf100k_RKI"] = map_rki

    if save_csv:
        day_dpath = "../figures/{}_{}_{}".format(start_day.year, start_day.month, start_day.day)
        if not os.path.isdir(day_dpath):
            os.mkdir(day_dpath)
        map_df.to_csv(os.path.join(day_dpath, "map.csv"))
    
    return map_df

if __name__ == "__main__": 

    import sys
    start = sys.argv[2]
    _ = curves(start ,save_csv=True)

