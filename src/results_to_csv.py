from shared_utils import *
import pickle as pkl
import numpy as np
import pandas as pd
import argparse
from collections import OrderedDict
from pymc3.stats import quantiles
from pathlib import Path


def main(start):
    start = int(os.environ["SGE_DATE_ID"])

    with open('../data/counties/counties.pkl', "rb") as f:
        counties = pkl.load(f)

    # default
    n_weeks = 3

    countyByName = make_county_dict()
    start_day = pd.Timestamp('2020-01-28') + pd.Timedelta(days=start)
    year = str(start_day)[:4]
    month = str(start_day)[5:7]
    day = str(start_day)[8:10]

    day_folder_path = "../figures/{}_{}_{}".format(year, month, day)
    Path(day_folder_path).mkdir(parents=True, exist_ok=True)

    prediction_region = "germany"
    data = load_data_n_weeks(start, n_weeks, prediction_region, counties)

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

    # Load our prediction samples
    res = load_predictions(start, n_weeks)
    res_trend = load_trend_predictions(start, n_weeks)

    n_days = (day_p5 - start_day).days

    prediction_samples = np.reshape(res['y'], (res['y'].shape[0], -1, 412)) 
    prediction_samples = prediction_samples[:,i_start_day:i_start_day+n_days,:]

    prediction_samples_trend = np.reshape(res_trend['y'], (res_trend['y'].shape[0],  -1, 412))
    prediction_samples_trend = prediction_samples_trend[:,i_start_day:i_start_day+n_days,:]

    prediction_samples_trend_mu = np.reshape(res_trend['μ'], (res_trend['μ'].shape[0],  -1, 412))
    prediction_samples_trend_mu = prediction_samples_trend[:,i_start_day:i_start_day+n_days,:]

    ext_index = pd.DatetimeIndex([d for d in target.index] + \
            [d for d in pd.date_range(target.index[-1]+timedelta(1),day_p5-timedelta(1))])
            
    # TODO: figure out where quantiles comes from and if its pymc3, how to replace it
    prediction_quantiles = quantiles(prediction_samples, (5, 25, 75, 95)) 
    prediction_quantiles_trend = quantiles(prediction_samples_trend, (5, 25, 75, 95)) 

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
            prediction_samples_trend_mu,
            axis=0),
        index=ext_index,
        columns=target.columns)
    prediction_q25_trend = pd.DataFrame(
        data=prediction_quantiles_trend[25],
        index=ext_index,
        columns=target.columns)
    prediction_q75_trend = pd.DataFrame(
        data=prediction_quantiles_trend[75],
        index=ext_index,
        columns=target.columns)
    prediction_q5_trend = pd.DataFrame(
        data=prediction_quantiles_trend[5],
        index=ext_index,
        columns=target.columns)
    prediction_q95_trend = pd.DataFrame(
        data=prediction_quantiles_trend[95],
        index=ext_index,
        columns=target.columns)

    for (county, county_id) in countyByName.items():
        county_data = pd.DataFrame({
            'Raw Prediction Mean' : prediction_mean.loc[:,county_id].values,
            'Raw Prediction Q25' : prediction_q25.loc[:,county_id].values,
            'Raw Prediction Q75' : prediction_q75.loc[:,county_id].values,
            'Raw Prediction Q5' : prediction_q5.loc[:,county_id].values,
            'Raw Prediction Q95' : prediction_q95.loc[:,county_id].values,
            'Trend Prediction Mean' : prediction_mean_trend.loc[:,county_id].values,
            'Trend Prediction Q25' : prediction_q25_trend.loc[:,county_id].values,
            'Trend Prediction Q75' : prediction_q75_trend.loc[:,county_id].values,
            'Trend Prediction Q5' : prediction_q5_trend.loc[:,county_id].values,
            'Trend Prediction Q95' : prediction_q95_trend.loc[:,county_id].values,
            'RKI Meldedaten' : np.append(target.loc[:,county_id].values, np.repeat(np.nan, 5)),
            'is_nowcast' : (day_m5 <= ext_index) & (ext_index < day_0),
            'is_prediction' : (day_0 <= ext_index)},
            index = ext_index
        )
        county_data.to_csv("../csv/{}_{}_{}/{}.csv".format(year, month, day, countyByName[county]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize model samples as mean and \
                                     quantiles, writing results into csv files")

    parser.add_argument("--start",
                        nargs=1,
                        dest="start",
                        type=int,
                        required=True,
                        help="start day calculated by Jan 28 2020 + start days")

    args = parser.parse_args()
    
    main(args.start[0])