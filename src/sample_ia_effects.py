# -*- coding: utf-8 -*-
import itertools as it
import pickle as pkl
import os
from collections import OrderedDict
from sampling_utils import *
from shared_utils import *
import sys
import pandas as pd

from pathlib import Path
import argparse

def main(start, task_id, num_weeks, csv_path):
    start_date = pd.Timestamp("2020-01-28") + pd.Timedelta(start, unit='d')
    disease = "covid19"

    year = start_date.year
    month = start_date.month
    day = start_date.day
    day_folder_path = "../data/ia_effect_samples/{}_{}_{}".format(year, month, day)

    Path(day_folder_path).mkdir(parents=True, exist_ok=True)
    filename = "../data/ia_effect_samples/{}_{}_{}/{}_{}.pkl".format(year, 
                                                                     month, 
                                                                     day, 
                                                                     disease, 
                                                                     task_id)

    print("Running task {} - disease: {} - sample: {} -\
            startdate: {} - number of weeks: {} y\nWill create file {}".format(task_id, 
                                                                               disease,
                                                                               task_id,
                                                                               start_date,
                                                                               num_weeks,
                                                                               filename ))

    with open('../data/counties/counties.pkl', "rb") as f:
        counties = pkl.load(f)


    data = load_data_n_weeks(start, num_weeks, csv_path)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Samples IAEffects starting at\
                                Jan 28 2020 plus a number of days")

    parser.add_argument("start",
                        nargs=1,
                        type=int,
                        help="start day calculated by Jan 28 2020 + start days")
    parser.add_argument("--task_id",
                        nargs=1,
                        type=int,
                        dest="task_id")
    parser.add_argument("--num_weeks",
                        nargs=1,
                        dest="num_weeks",
                        type=int,
                        default=[3])
    parser.add_argument("--csv_path",
                        nargs=1,
                        dest="csv_path",
                        type=str,
                        default=["../data/diseases/covid19.csv"],
                        help="path to csv containing reported cases")

    args = parser.parse_args()

    print(args.csv_path)

    main(args.start[0],
         args.task_id[0],
         args.num_weeks[0],
         args.csv_path[0])
