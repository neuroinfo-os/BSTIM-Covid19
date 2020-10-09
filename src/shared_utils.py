from config import *
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
from sampling_utils import *
from collections import OrderedDict
import isoweek
import gc
from BaseModel import BaseModel
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


def load_data(disease, prediction_region, counties, separator=";"):
    """
        load_data(disease, prediction_region, counties, separator)

    Utility to load the data for a given disease and prediction_region. Only counties are selected that lie in the prediction_region.

    Arguments:
    ==========
        disease:            name of the disease
        prediction_region:  name of the region for which to make a prediction
        counties:           dictionary containing county information
        separator:          string delimiter
        """
    # load raw data from csv file into a pandas dataframe
    data = pd.read_csv("../data/diseases/{}.csv".format(disease),
                       sep=separator, encoding='iso-8859-1', index_col=0)
    # exclude data reported for unknown counties
    if "99999" in data.columns:
        data.drop("99999", inplace=True, axis=1)
    # exclude data for regions outside of region of interest
    data = data.loc[:, list(
        filter(lambda cid: prediction_region in counties[cid]["region"], data.columns))]
    data.index = parse_yearweek(data.index)
    return data


def load_daily_data(disease, prediction_region, counties, seperator=",", pad=None):
    data = pd.read_csv("../data/diseases/{}.csv".format(disease),
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

def load_daily_data_n_weeks(start, n_weeks,disease, prediction_region, counties, seperator=",", pad=None, permute=False):
    data = pd.read_csv("../data/diseases/{}.csv".format(disease),
                       sep=seperator, encoding='iso-8859-1', index_col=0)

    if "99999" in data.columns:
        data.drop("99999", inplace=True, axis=1)

    data.index = [pd.Timestamp(date) for date in data.index]
    start_day = pd.Timestamp('2020-01-28') + pd.Timedelta(days=start)
    data = data.loc[start_day <= data.index]

    data = data.loc[:, list(
        filter(lambda cid: prediction_region in counties[cid]["region"], data.columns))]
    print("kwbcuf")
    if permute:
        print(data.head())
        data_stored = data.to_numpy()
        print(data_stored[:5,:])
        np.random.shuffle(data_stored.T)
        data_new = pd.DataFrame(data_stored, index=data.index, columns=data.columns)
        print(data_new.head())
    
    if pad is not None:
        # get last date
        last_date = pd.Timestamp(data.iloc[:, -1].index[-1])
        extra_range = pd.date_range(
            last_date+timedelta(1), last_date+timedelta(pad))
        for x in extra_range:
            data = data.append(pd.Series(name=str(x)[:11]))

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


def quantile_negbin(qs, mean, dispersion=0):
    """ For `n` values in `qs` and `m` values in `mean`, computes the m
    by n matrix of `qs` quantiles for distributions with means `mean`.
    If dispersion is set to 0, a Poisson distribution is assumed, 
    otherwise a Negative Binomial with corresponding dispersion parameter is used."""
    qs = np.array(qs).ravel()
    mean = np.array(mean).ravel()
    res = np.empty((len(mean), len(qs)))
    for i, mu in enumerate(mean):
        if dispersion == 0:
            # plot Poisson quantiles
            dist = scipy.stats.poisson(mu)
        else:
            # k = y
            n = 1 / dispersion
            p = 1 / (1 + dispersion * mu)
            dist = scipy.stats.nbinom(n, p)
        for j, q in enumerate(qs):
            res[i, j] = dist.ppf(q)
    return res


def deviance_negbin(y, μ, α, saturated="NegativeBinomial"):
    if saturated == "NegativeBinomial":
        logp_sat = tt.where(
            y == 0, np.zeros_like(
                y, dtype=np.float32), pm.NegativeBinomial.dist(
                mu=y, alpha=α).logp(y))
    elif saturated == "Poisson":
        logp_sat = tt.where(y == 0, np.zeros_like(
            y, dtype=np.float32), pm.Poisson.dist(mu=y).logp(y))
    else:
        raise NotImplementedError()
    logp_mod = pm.NegativeBinomial.dist(mu=μ, alpha=α).logp(y)
    return (2 * (logp_sat - logp_mod)).eval()


def dss(y, μ, var):
    return np.log(var) + (y - μ)**2 / var


def pit_negbin(y, μ, α):
    return scipy.stats.nbinom.cdf(y, α, μ / (α + μ))


def make_axes_stack(
        ax,
        num,
        xoff,
        yoff,
        fig=None,
        down=True,
        sharex=False,
        sharey=False):
    if isinstance(ax, plt.Axes):
        bbox = ax.get_position()
        if fig is None:
            fig = ax.figure
        ax.set_zorder(num if down else 0)
        axes = [ax]
    elif isinstance(ax, SubplotSpec):
        if fig is None:
            fig = plt.gcf()
        bbox = ax.get_position(fig)
        axes = [fig.add_axes(bbox, zorder=num if down else 0)]
    else:
        if fig is None:
            fig = plt.gcf()
        bbox = transforms.Bbox.from_bounds(
            ax[0], ax[1], ax[2] - ax[0], ax[3] - ax[1])
        axes = [fig.add_axes(bbox, zorder=num if down else 0)]

    for i in range(1, num):
        bbox = bbox.translated(xoff, yoff)
        ax = fig.add_axes(
            bbox,
            zorder=num - i if down else i,
            sharex=axes[0] if sharex else None,
            sharey=axes[0] if sharey else None)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_visible(False)
        axes.append(ax)
    return axes


def set_file_permissions(filename, uid, gid, permissions=0o660):
    os.chmod(filename, permissions)
    os.chown(filename, uid, gid)


def load_model(disease, use_age, use_eastwest):
    filename_model = "../data/mcmc_samples_backup/model_{}_{}_{}.pkl".format(
        disease, use_age, use_eastwest)

    with open(filename_model, "rb") as f:
        model = pkl.load(f)
    return model

def load_model_by_i(disease, i):
    """ specialized to enumeratable experiments """
    filename_model = "../data/mcmc_samples_backup/model_{}_{}.pkl".format(disease, i)

    with open(filename_model, "rb") as f:
        model = pkl.load(f)
    return model

def load_model_window(disease, model_i, start, n_weeks):
    filename_model = "../data/mcmc_samples_backup/model_{}_{}.pkl".format(disease, start)
    with open(filename_model, "rb") as f:
        model = pkl.load(f)
    return model

def load_final_model(no_rd=False):
    if no_rd:
        filename_model = "../data/mcmc_samples_backup/model_covid19_final_no_rd.pkl"
    else:
        filename_model = "../data/mcmc_samples_backup/model_covid19_final.pkl"

    with open(filename_model, "rb") as f:
        model = pkl.load(f)
    return model    


def load_trace(disease, use_age, use_eastwest):
    filename_params = "../data/mcmc_samples_backup/parameters_{}_{}_{}".format(
        disease, use_age, use_eastwest)

    model = load_model(disease, use_age, use_eastwest)
    with model:
        trace = pm.load_trace(filename_params)

    del model
    return trace

def load_trace_by_i(disease, i):
    filename_params = "../data/mcmc_samples_backup/parameters_{}_{}".format(disease, i)

    model = load_model_by_i(disease, i)
    with model:
        trace = pm.load_trace(filename_params)
    del model
    return trace

def load_trace_window(disease, model_i, start, n_weeks):
    filename_params = "../data/mcmc_samples_backup/parameters_{}_{}".format(disease, start)
    model = load_model_window(disease, model_i, start, n_weeks)
    with model:
        trace = pm.load_trace(filename_params)
    del model
    return trace


def load_final_trace(no_rd=False):
    if no_rd:
        filename_params = "../data/mcmc_samples_backup/parameters_covid19_final_no_rd"
    else:
        filename_params = "../data/mcmc_samples_backup/parameters_covid19_final"
    model = load_final_model(no_rd=no_rd)
    with model:
        trace = pm.load_trace(filename_params)
    del model
    return trace

def load_pred(disease, use_age, use_eastwest):
    # Load our prediction samples
    filename_pred = "../data/mcmc_samples_backup/predictions_{}_{}_{}.pkl".format(
            disease, use_age, use_eastwest)
    with open(filename_pred, "rb") as f:
        res = pkl.load(f)
    return res

def load_pred_by_i(disease, i):
    filename_pred = "../data/mcmc_samples_backup/predictions_{}_{}.pkl".format(disease, i)
    with open(filename_pred, "rb") as f:
        res = pkl.load(f)
    return res

def load_final_pred(no_rd=False):
    if no_rd:
        filename_pred = "../data/mcmc_samples_backup/predictions_covid19_final_no_rd.pkl"
    else:
        filename_pred = "../data/mcmc_samples_backup/predictions_covid19_final.pkl"
    with open(filename_pred, "rb") as f:
        res = pkl.load(f)
    return res

def load_pred_model_window(model_i,start, n_weeks,nowcast=False,trend=False):
    if nowcast:
        filename_pred = "../data/mcmc_samples_backup/predictions_nowcast_covid19_model_{}_window_{}_{}.pkl".format(model_i,start, n_weeks)
    else:
        filename_pred = "../data/mcmc_samples_backup/predictions_covid19_{}.pkl".format(start)
    if trend:
        filename_pred = "../data/mcmc_samples_backup/predictions_trend_covid19_{}.pkl".format(start)

    with open(filename_pred, "rb") as f:
        res = pkl.load(f)
    return res


def load_final_nowcast_pred(no_rd=False):
    if no_rd:
        filename_pred = "../data/mcmc_samples_backup/predictions_nowcast_covid19_final_no_rd.pkl"
    else:
        filename_pred = "../data/mcmc_samples_backup/predictions_nowcast_covid19_final.pkl"
    with open(filename_pred, "rb") as f:
        res = pkl.load(f)
    return res
