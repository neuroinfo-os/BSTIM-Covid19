import argparse
import os
import pickle as pkl
import shutil
import uuid
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pymc3.stats import quantiles
from shared_utils import *


def metadata_csv(start, n_weeks, counties, output_dir):
    trace = load_trace(start, n_weeks)
    p_linear_slope = np.mean((trace["W_t_t"] > 0)[:, :, 1], axis=0)
    prob_text = [
        p * 100 if p >= 0.5 else -(100 - p*100) for p in p_linear_slope
    ]
    n_people = [
        counties[key]["demographics"][("total", 2018)] for key in counties.keys()
    ]

    metadata = pd.DataFrame(
        {
            "countyID": [int(county_id) for county_id in counties.keys()],
            "LKType": [
                counties[county_id]["name"].split(" ")[0]
                for county_id in counties.keys()
            ],
            "LKName": [
                counties[county_id]["name"].split(" ")[1]
                for county_id in counties.keys()
            ],
            "probText": prob_text,
            "n_people": n_people,
        }
    )
    metadata = metadata.sort_values(["LKName", "countyID"])

    fpath = os.path.join(output_dir, "metadata.csv")
    metadata.to_csv(fpath, index=False)


def sample_x_days_incidence_by_county(samples, x):
    offset = x-1 if x >= 1 else 0
    num_sample = len(samples)
    timesteps = len(samples[0])
    counties = len(samples[0][0])
    incidence = np.empty((num_sample, timesteps - offset, counties), dtype=np.float64)
    for sample in range(num_sample):
        for interval in range(timesteps - offset):
            incidence[sample][interval] = np.sum(samples[sample][interval : interval + x], axis=0)
    return incidence


def plotdata_csv(start, n_weeks, csv_path, counties, output_dir):
    countyByName = make_county_dict()
    data = load_data_n_weeks(start, n_weeks, csv_path)
    start_day = pd.Timestamp("2020-01-28") + pd.Timedelta(days=start)
    day_0 = start_day + pd.Timedelta(days=n_weeks * 7 + 5)
    day_m5 = day_0 - pd.Timedelta(days=5)
    day_p5 = day_0 + pd.Timedelta(days=5)
    _, target, _, _ = split_data(
        data, train_start=start_day, test_start=day_0, post_test=day_p5
    )

    # Load our prediction samples
    res = load_predictions(start, n_weeks)
    res_trend = load_trend_predictions(start, n_weeks)

    prediction_samples = np.reshape(res["y"], (res["y"].shape[0], -1, 412))
    prediction_samples_mu = np.reshape(res["μ"], (res["μ"].shape[0], -1, 412))
    prediction_samples_trend = np.reshape(
        res_trend["y"], (res_trend["y"].shape[0], -1, 412)
    )
    prediction_samples_trend_mu = np.reshape(
        res_trend["μ"], (res_trend["μ"].shape[0], -1, 412)
    )
    predictions_7day_inc = sample_x_days_incidence_by_county(
        prediction_samples_trend, 7
    )
    predictions_7day_inc_mu = sample_x_days_incidence_by_county(
        prediction_samples_trend_mu, 7
    )
    ext_index = pd.DatetimeIndex(
        [d for d in target.index]
        + [
            d
            for d in pd.date_range(
                target.index[-1] + timedelta(1), day_p5 - timedelta(1)
            )
        ]
    )
    # TODO: figure out if we want to replace quantiles function (newer pymc3 versions don't support it)
    prediction_quantiles = quantiles(prediction_samples, (5, 25, 75, 95))
    prediction_quantiles_trend = quantiles(prediction_samples_trend, (5, 25, 75, 95))
    prediction_quantiles_7day_inc = quantiles(predictions_7day_inc, (5, 25, 75, 95))

    prediction_mean = pd.DataFrame(
        data=np.mean(prediction_samples_mu, axis=0),
        index=ext_index,
        columns=target.columns,
    )
    prediction_q25 = pd.DataFrame(
        data=prediction_quantiles[25], index=ext_index, columns=target.columns
    )
    prediction_q75 = pd.DataFrame(
        data=prediction_quantiles[75], index=ext_index, columns=target.columns
    )
    prediction_q5 = pd.DataFrame(
        data=prediction_quantiles[5], index=ext_index, columns=target.columns
    )
    prediction_q95 = pd.DataFrame(
        data=prediction_quantiles[95], index=ext_index, columns=target.columns
    )

    prediction_mean_trend = pd.DataFrame(
        data=np.mean(prediction_samples_trend_mu, axis=0),
        index=ext_index,
        columns=target.columns,
    )
    prediction_q25_trend = pd.DataFrame(
        data=prediction_quantiles_trend[25], index=ext_index, columns=target.columns
    )
    prediction_q75_trend = pd.DataFrame(
        data=prediction_quantiles_trend[75], index=ext_index, columns=target.columns
    )
    prediction_q5_trend = pd.DataFrame(
        data=prediction_quantiles_trend[5], index=ext_index, columns=target.columns
    )
    prediction_q95_trend = pd.DataFrame(
        data=prediction_quantiles_trend[95], index=ext_index, columns=target.columns
    )

    prediction_mean_7day = pd.DataFrame(
        data=np.pad(
            np.mean(predictions_7day_inc_mu, axis=0),
            ((6, 0), (0, 0)),
            "constant",
            constant_values=np.nan,
        ),
        index=ext_index,
        columns=target.columns,
    )
    prediction_q25_7day = pd.DataFrame(
        data=np.pad(
            prediction_quantiles_7day_inc[25].astype(float),
            ((6, 0), (0, 0)),
            "constant",
            constant_values=np.nan,
        ),
        index=ext_index,
        columns=target.columns,
    )
    prediction_q75_7day = pd.DataFrame(
        data=np.pad(
            prediction_quantiles_7day_inc[75].astype(float),
            ((6, 0), (0, 0)),
            "constant",
            constant_values=np.nan,
        ),
        index=ext_index,
        columns=target.columns,
    )
    prediction_q5_7day = pd.DataFrame(
        data=np.pad(
            prediction_quantiles_7day_inc[5].astype(float),
            ((6, 0), (0, 0)),
            "constant",
            constant_values=np.nan,
        ),
        index=ext_index,
        columns=target.columns,
    )
    prediction_q95_7day = pd.DataFrame(
        data=np.pad(
            prediction_quantiles_7day_inc[95].astype(float),
            ((6, 0), (0, 0)),
            "constant",
            constant_values=np.nan,
        ),
        index=ext_index,
        columns=target.columns,
    )

    rki_7day = target.rolling(7).sum()

    ref_date = target.iloc[-1].name
    nowcast_vals = prediction_mean.loc[prediction_mean.index == ref_date]
    nowcast7day_vals = prediction_mean_7day.loc[prediction_mean.index == ref_date]
    rki_vals = target.iloc[-1]
    rki_7day_vals = rki_7day.iloc[-1]

    map_nowcast = []
    map_nowcast100k = []
    map_nowcast_7day = []
    map_nowcast_7day100k = []
    map_rki = []
    map_rki100k = []
    map_rki_7day = []
    map_rki_7day100k = []
    map_keys = []

    for (county, county_id) in countyByName.items():
        rki_data = np.append(target.loc[:, county_id].values, np.repeat(np.nan, 5))
        rki_data7day = np.append(
            rki_7day.loc[:, county_id].values, np.repeat(np.nan, 5)
        )
        n_people = counties[county_id]["demographics"][("total", 2018)]

        map_nowcast.append(nowcast_vals[county_id].item())
        map_nowcast100k.append(nowcast_vals[county_id].item() / n_people * 100000)
        map_nowcast_7day.append(nowcast7day_vals[county_id].item())
        map_nowcast_7day100k.append(
            nowcast7day_vals[county_id].item() / n_people * 100000
        )
        map_rki.append(rki_vals[county_id].item())
        map_rki100k.append(rki_vals[county_id].item() / n_people * 100000)
        map_rki_7day.append(rki_7day_vals[county_id].item())
        map_rki_7day100k.append(rki_7day_vals[county_id].item() / n_people * 100000)
        map_keys.append(county_id)

        county_data = pd.DataFrame(
            {
                "Raw Prediction Mean": prediction_mean.loc[:, county_id].values,
                "Raw Prediction Mean 100k": np.multiply(
                    np.divide(prediction_mean.loc[:, county_id].values, n_people),
                    100000,
                ),
                "Raw Prediction Q25": prediction_q25.loc[:, county_id].values,
                "Raw Prediction Q25 100k": np.multiply(
                    np.divide(prediction_q25.loc[:, county_id].values, n_people),
                    100000,
                ),
                "Raw Prediction Q75": prediction_q75.loc[:, county_id].values,
                "Raw Prediction Q75 100k": np.multiply(
                    np.divide(prediction_q75.loc[:, county_id].values, n_people),
                    100000,
                ),
                "Raw Prediction Q5": prediction_q5.loc[:, county_id].values,
                "Raw Prediction Q5 100k": np.multiply(
                    np.divide(prediction_q5.loc[:, county_id].values, n_people), 100000,
                ),
                "Raw Prediction Q95": prediction_q95.loc[:, county_id].values,
                "Raw Prediction Q95 100k": np.multiply(
                    np.divide(prediction_q95.loc[:, county_id].values, n_people),
                    100000,
                ),
                "Trend Prediction Mean": prediction_mean_trend.loc[:, county_id].values,
                "Trend Prediction Mean 100k": np.multiply(
                    np.divide(prediction_mean_trend.loc[:, county_id].values, n_people),
                    100000,
                ),
                "Trend Prediction Q25": prediction_q25_trend.loc[:, county_id].values,
                "Trend Prediction Q25 100k": np.multiply(
                    np.divide(prediction_q25_trend.loc[:, county_id].values, n_people),
                    100000,
                ),
                "Trend Prediction Q75": prediction_q75_trend.loc[:, county_id].values,
                "Trend Prediction Q75 100k": np.multiply(
                    np.divide(prediction_q75_trend.loc[:, county_id].values, n_people),
                    100000,
                ),
                "Trend Prediction Q5": prediction_q5_trend.loc[:, county_id].values,
                "Trend Prediction Q5 100k": np.multiply(
                    np.divide(prediction_q5_trend.loc[:, county_id].values, n_people),
                    100000,
                ),
                "Trend Prediction Q95": prediction_q95_trend.loc[:, county_id].values,
                "Trend Prediction Q95 100k": np.multiply(
                    np.divide(prediction_q95_trend.loc[:, county_id].values, n_people),
                    100000,
                ),
                "Trend 7Day Prediction Mean": prediction_mean_7day.loc[
                    :, county_id
                ].values,
                "Trend 7Day Prediction Mean 100k": np.multiply(
                    np.divide(prediction_mean_7day.loc[:, county_id].values, n_people),
                    100000,
                ),
                "Trend 7Day Prediction Q25": prediction_q25_7day.loc[
                    :, county_id
                ].values,
                "Trend 7Day Prediction Q25 100k": np.multiply(
                    np.divide(prediction_q25_7day.loc[:, county_id].values, n_people),
                    100000,
                ),
                "Trend 7Day Prediction Q75": prediction_q75_7day.loc[
                    :, county_id
                ].values,
                "Trend 7Day Prediction Q75 100k": np.multiply(
                    np.divide(prediction_q75_7day.loc[:, county_id].values, n_people),
                    100000,
                ),
                "Trend 7Day Prediction Q5": prediction_q5_7day.loc[
                    :, county_id
                ].values,
                "Trend 7Day Prediction Q5 100k": np.multiply(
                    np.divide(prediction_q5_7day.loc[:, county_id].values, n_people),
                    100000,
                ),
                "Trend 7Day Prediction Q95": prediction_q95_7day.loc[
                    :, county_id
                ].values,
                "Trend 7Day Prediction Q95 100k": np.multiply(
                    np.divide(prediction_q95_7day.loc[:, county_id].values, n_people),
                    100000,
                ),
                "RKI Meldedaten": rki_data,
                "RKI 7Day Incidence": rki_data7day,
                "is_nowcast": (day_m5 <= ext_index) & (ext_index < day_0),
                "is_high": np.less(
                    prediction_q95_trend.loc[:, county_id].values, rki_data
                ),
                "is_prediction": (day_0 <= ext_index),
            },
            index=ext_index,
        )
        fpath = os.path.join(output_dir, "{}.csv".format(countyByName[county]))
        county_data.to_csv(fpath)

    map_df = pd.DataFrame(index=None)
    map_df["countyID"] = map_keys
    map_df["newInf100k"] = map_nowcast100k
    map_df["7DayInf100k"] = map_nowcast_7day100k
    map_df["newInf100k_RKI"] = map_rki100k
    map_df["7DayInf100k_RKI"] = map_rki_7day100k
    map_df["newInfRaw"] = map_nowcast
    map_df["7DayInfRaw"] = map_nowcast_7day
    map_df["newInfRaw_RKI"] = map_rki
    map_df["7DayInfRaw_RKI"] = map_rki_7day
    map_df.to_csv(os.path.join(output_dir, "map.csv"))


def export(start, outputdir, exportrootdir, exportoffset=0):
    exportday = (
        pd.Timestamp("2020-01-28")
        + pd.Timedelta(days=start)
        + pd.Timedelta(days=exportoffset)
    )
    export_year = str(exportday)[:4]
    export_month = str(exportday)[5:7]
    export_day = str(exportday)[8:10]
    exportdir = os.path.join(
        exportrootdir, "{}_{}_{}".format(export_year, export_month, export_day)
    )
    Path(os.path.basename(exportdir)).mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(outputdir, exportdir)
    except FileExistsError:
        uuidcode = uuid.uuid4().hex
        exportdir2 = "{}{}".format(exportdir, uuidcode)
        print("{} already exists. Try {}".format(exportdir, exportdir2))
        shutil.copytree(outputdir, exportdir2)


def main(start, csv_path, output_root_dir, exportdir, exportoffset):
    # default
    n_weeks = 3
    with open("../data/counties/counties.pkl", "rb") as f:
        counties = pkl.load(f)
    start_day = pd.Timestamp("2020-01-28") + pd.Timedelta(days=start)
    year = str(start_day)[:4]
    month = str(start_day)[5:7]
    day = str(start_day)[8:10]
    output_dir = os.path.join(output_root_dir, "{}_{}_{}".format(year, month, day))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metadata_csv(start, n_weeks, counties, output_dir)
    plotdata_csv(start, n_weeks, csv_path, counties, output_dir)
    if exportdir:
        export(start, output_dir, exportdir, exportoffset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize model samples as mean and \
                                     quantiles, writing results into csv files"
    )

    parser.add_argument(
        "start",
        nargs=1,
        type=int,
        help="start day calculated by Jan 28 2020 + start days",
    )

    parser.add_argument(
        "--csvinputfile",
        nargs=1,
        dest="csvinputfile",
        type=str,
        default=["../data/diseases/covid19.csv"],
        help="path to input csv file",
    )

    parser.add_argument(
        "--outputrootdir",
        nargs=1,
        dest="outputrootdir",
        type=str,
        default=["../csv/"],
        help="path to output root folder",
    )

    parser.add_argument(
        "--exportdir",
        nargs=1,
        dest="exportdir",
        type=str,
        help="path to export csv files",
    )

    parser.add_argument(
        "--exportoffset",
        nargs=1,
        dest="exportoffset",
        type=int,
        help="Define an offset for export directory",
    )

    args = parser.parse_args()

    exportdir = args.exportdir[0] if args.exportdir else None
    exportoffset = args.exportoffset[0] if args.exportoffset else 0

    main(
        args.start[0],
        args.csvinputfile[0],
        args.outputrootdir[0],
        exportdir,
        exportoffset,
    )
