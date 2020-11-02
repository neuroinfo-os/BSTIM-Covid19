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

from pathlib import Path

def main():
    start = int(os.environ["SGE_DATE_ID"])
    print(start)
    num_sample = int(os.environ["SGE_TASK_ID"])
    number_of_weeks = 3
    start_date = pd.Timestamp("2020-01-28") + pd.Timedelta(start, unit='d')

    # maybe not necessary
    disease = "covid19"
    prediction_region = "germany"
    nums_sample = range(100)

    year = str(start_date)[:4]
    month = str(start_date)[5:7]
    day = str(start_date)[8:10]
    day_folder_path = "../data/ia_effect_samples/{}_{}_{}".format(year, month, day)
    Path(day_folder_path).mkdir(parents=True, exist_ok=True)

    filename = "../data/ia_effect_samples/{}_{}_{}/{}_{}.pkl".format(year, month, day, disease, num_sample)

    print("Running task {} - disease: {} - sample: {} - startdate: {} - number of weeks: {} y\nWill create file {}".\
                                                format(num_sample, disease, num_sample, start_date, number_of_weeks, filename ))

    with open('../data/counties/counties.pkl', "rb") as f:
        counties = pkl.load(f)


    data = load_data_n_weeks(start, number_of_weeks, disease, prediction_region, counties, permute=PERMUTATION_STUDY)
    print("DaysTest")
    print(data.index)
    # RNGenerators
    rnd_tsel = np.random.Generator(np.random.PCG64())
    times = uniform_times_by_day(data.index, rnd_tsel)
    rnd_csel = np.random.Generator(np.random.PCG64())
    locs = uniform_locations_by_county(counties, rnd_csel)

    res = iaeffect_sampler(data, times, locs, temporal_bfs, spatial_bfs)
    results = {"ia_effects": res, "predicted day": data.index,
               "predicted county": data.columns}

    with open(filename, "wb") as file:
        pkl.dump(results, file)
