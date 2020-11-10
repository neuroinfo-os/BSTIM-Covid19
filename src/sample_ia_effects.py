# -*- coding: utf-8 -*-
import argparse
import itertools as it
import os
import pickle as pkl
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from sampling_utils import *
from shared_utils import *


def main(start, task_id, num_weeks, csv_path, ia_effect_root):
    start_date = pd.Timestamp("2020-01-28") + pd.Timedelta(start, unit="d")
    disease = "covid19"

    year = str(start_date)[:4]
    month = str(start_date)[5:7]
    day = str(start_date)[8:10]
    day_folder_path = os.path.join(ia_effect_root, "{}_{}_{}".format(year, month, day))

    Path(day_folder_path).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(day_folder_path, "{}_{}.pkl".format(disease, task_id))

    print(
        "Running task {} - disease: {} - sample: {} -\
            startdate: {} - number of weeks: {} y\nWill create file {}".format(
            task_id, disease, task_id, start_date, num_weeks, filename
        )
    )

    with open("../data/counties/counties.pkl", "rb") as f:
        counties = pkl.load(f)

    data = load_data_n_weeks(start, num_weeks, csv_path)
    day_0 = start_date + pd.Timedelta(days=num_weeks * 7 + 5)
    _, target, _, _ = split_data(
        data, train_start=start_date, test_start=day_0, post_test=day_0
    )

    # RNGenerators
    rnd_tsel = np.random.Generator(np.random.PCG64())
    times = uniform_times_by_day(target.index, rnd_tsel)
    rnd_csel = np.random.Generator(np.random.PCG64())
    locs = uniform_locations_by_county(counties, rnd_csel)

    res = iaeffect_sampler(target, times, locs, temporal_bfs, spatial_bfs)
    results = {
        "ia_effects": res,
        "predicted day": data.index,
        "predicted county": data.columns,
    }

    with open(filename, "wb") as file:
        pkl.dump(results, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Samples IAEffects starting at\
                                Jan 28 2020 plus a number of days"
    )

    parser.add_argument(
        "start",
        nargs=1,
        type=int,
        help="start day calculated by Jan 28 2020 + start days",
    )
    parser.add_argument("--task_id", nargs=1, type=int, dest="task_id")
    parser.add_argument("--num_weeks", nargs=1, dest="num_weeks", type=int, default=[3])
    parser.add_argument(
        "--csvinputfile",
        nargs=1,
        dest="csvinputfile",
        type=str,
        default=["../data/diseases/covid19.csv"],
        help="path to csv containing reported cases",
    )

    parser.add_argument(
        "--ia_effect_root",
        nargs=1,
        dest="ia_effect_root",
        type=str,
        default=["../data/ia_effect_samples"],
        help="Path to directory with IA effect sample files in date folders",
    )

    args = parser.parse_args()

    main(
        args.start[0],
        args.task_id[0],
        args.num_weeks[0],
        args.csvinputfile[0],
        args.ia_effect_root[0],
    )
