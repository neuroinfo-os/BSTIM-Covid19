# -*- coding: utf-8 -*-
import itertools as it
import pickle as pkl
import os
from collections import OrderedDict
from sampling_utils import *
from shared_utils import *
import sys 
import pandas as pd
from config_window import combinations

CID = int(os.environ["SGE_TASK_ID"])
start = CID[1]
num_sample = CID[0]
number_of_weeks = 3
start_date = pd.Timestamp("2020-01-28") + pd.Timedelta(start)

disease = "covid19"
nums_sample = range(100)
#GID = int(os.environ["SGE_TASK_ID"])


filename = "../data/ia_effect_samples/start{}_weeks{}_{}_{}.pkl".format(start, number_of_weeks,disease, num_sample)

print("Running task {} - disease: {} - sample: {} - startdate: {} - number of weeks: {} y\nWill create file {}".\
                                            format(GID,disease, num_sample, start_date, number_of_weeks, filename ))

with open('../data/counties/counties.pkl', "rb") as f:
    counties = pkl.load(f)


prediction_region = "germany"
parameters = OrderedDict()

# Load data
data = load_daily_data_n_weeks(start, number_of_weeks, disease, prediction_region, counties)
print("DaysTest")
print(data.index)
# samples random times --> check the data conversion carefully
# check if correct
times = uniform_times_by_day(data.index)
locs = uniform_locations_by_county(counties)

#NOTE: Do we want basis functions with a longer temporal horizon? // we may want to weight them around fixed days?!
#NOTE: Split this up, so we can get multiple basis functions!
def temporal_bfs(x): return bspline_bfs(x, np.array(
    [0, 0, 1, 2, 3, 4, 5]) * 24 * 3600.0, 2) 


def spatial_bfs(x): return [gaussian_bf(x, σ)
                            for σ in [6.25, 12.5, 25.0, 50.0]]


samp = IAEffectSampler(data, times, locs, temporal_bfs,
                       spatial_bfs, num_tps=10, time_horizon=5)
res = samp(data.index, data.columns)
results = {"ia_effects": res, "predicted day": data.index,
           "predicted county": data.columns}

with open(filename, "wb") as file:
    pkl.dump(results, file)
