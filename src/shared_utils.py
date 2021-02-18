from collections import defaultdict
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from descartes import PolygonPatch
import seaborn as sns
import re
import theano.tensor as tt
import scipy.stats
from itertools import product
import datetime
import pickle as pkl
import numpy as np
import pandas as pd
import pymc3 as pm
# from pymc3.stats import quantiles
from collections import OrderedDict
import isoweek
import itertools as it
import os
from datetime import timedelta

yearweek_regex = re.compile(r"([0-9]+)-KW([0-9]+)")

def make_county_dict():
    with open('../data/counties/counties.pkl', "rb") as f:
        counties = pkl.load(f)

    county_list = []
    #print(counties)
    for key, _ in counties.items():
        county_name = counties[key]['name']
        encoded_name = counties[key]['name'].encode('utf-8')
        if b'\xc2\x96' in encoded_name:
            ix = encoded_name.index(b'\xc2\x96')
            county_name = counties[key]['name'][:ix]+'-'+counties[key]['name'][ix+1:]
        county_list.append((county_name, key))
    return OrderedDict(county_list)

def _parse_yearweek(yearweek):
    """Utility function to convert internal string representations of calender weeks into datetime objects. Uses strings of format `<year>-KW<week>`. Weeks are 1-based."""
    year, week = yearweek_regex.search(yearweek).groups()
    # datetime.combine(isoweek.Week(int(year), int(week)).wednesday(),time(0))
    return isoweek.Week(int(year), int(week))


parse_yearweek = np.frompyfunc(_parse_yearweek, 1, 1)

def load_data(prediction_region, counties, csv_path, seperator=",", pad=None):
    data = pd.read_csv(csv_path,
                       sep=seperator, encoding='iso-8859-1', index_col=0)

    if "99999" in data.columns:
        data.drop("99999", inplace=True, axis=1)

    data = data.loc[:, list(
        filter(lambda cid: prediction_region in counties[cid]["region"], data.columns))]

    if pad is not None:
        # get last date
        last_date = pd.Timestamp(data.iloc[:, -1].index[-1])
        extra_range = pd.date_range(
            last_date+timedelta(1), last_date+timedelta(pad))
        for x in extra_range:
            data = data.append(pd.Series(name=str(x)[:11]))

    data.index = [pd.Timestamp(date) for date in data.index]
    return data

def load_data_n_weeks(
    start,
    n_weeks,
    csv_path,
    seperator=",",
    pad = None
):

    data = pd.read_csv(csv_path, sep=seperator, encoding='iso-8859-1', index_col=0)

    if "99999" in data.columns:
        data.drop("99999", inplace=True, axis=1)

    data.index = [pd.Timestamp(date) for date in data.index]
    start_day = pd.Timestamp('2020-01-28') + pd.Timedelta(days=start)
    data = data.loc[start_day <= data.index]

    if pad is not None:
        last_date = data.index[-1]
        extended_index = pd.date_range(last_date + pd.Timedelta(days=1),
                                       last_date + pd.Timedelta(days=pad))
        for x in extended_index:
            data = data.append(pd.Series(name=x))

    data.index = [pd.Timestamp(date) for date in data.index]

    return data



def split_data(
    data,
    train_start=parse_yearweek("2011-KW01"),
    test_start=parse_yearweek("2016-KW01"),
    post_test=parse_yearweek("2018-KW01")
):
    """
        split_data(data,data_start,train_start,test_start)

    Utility function that splits the dataset into training and testing data as well as the corresponding target values.

    Returns:
    ========
        data_train:     training data (from beginning of records to end of training phase)
        target_train:   target values for training data
        data_test:      testing data (from beginning of records to end of testing phase = end of records)
        target_test:    target values for testing data
    """

    target_train = data.loc[(train_start <= data.index)
                            & (data.index < test_start)]
    target_test = data.loc[(test_start <= data.index)
                           & (data.index < post_test)]

    data_train = data.loc[data.index < test_start]
    data_test = data

    return data_train, target_train, data_test, target_test

def load_trace(start, n_weeks):
    filename = "../data/mcmc_samples_backup/parameters_covid19_{}".format(start)
    model = load_model(start, n_weeks)
    with model:
        trace = pm.load_trace(filename)
    del model
    return trace

def load_model(start, n_weeks):
    filename = "../data/mcmc_samples_backup/model_covid19_{}.pkl".format(start)
    with open(filename, "rb") as f:
        model = pkl.load(f)
    return model

def load_predictions(start, n_weeks):
    filename = "../data/mcmc_samples_backup/predictions_covid19_{}.pkl".format(start)
    with open(filename, "rb") as f:
        res = pkl.load(f)
    return res

def load_trend_predictions(start, n_weeks):
    filename = "../data/mcmc_samples_backup/predictions_trend_covid19_{}.pkl".format(start)
    with open(filename, "rb") as f:
        res = pkl.load(f)
    return res
