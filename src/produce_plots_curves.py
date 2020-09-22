from plot_curves_window import curves as curves_window
from plot_curves_window_trend import curves as curves_window_trend
from shared_utils import make_county_dict

import os
import pandas as pd
import argparse
import shutil
import numpy as np

parser = argparse.ArgumentParser(description='NYI')

parser.add_argument(
    "--njobs",
    nargs=1,
    type=int,
    dest="njobs")

parser.add_argument(
    "--jobid",
    nargs=1,
    type=int,
    dest="jobid")

args = parser.parse_args()

# start = int(os.environ["SGE_DATE_ID"]) 
start = 1
county_dict = make_county_dict()
start_day = pd.Timestamp('2020-01-28') + pd.Timedelta(days=start)
year = str(start_day)[:4]
month = str(start_day)[5:7]
day = str(start_day)[8:10]

figures_path = "/p/project/covid19dynstat/autostart/BSTIM-Covid19_Window_Final/figures/{}_{}_{}/".format(year, month, day)
shared_path = "/p/project/covid19dynstat/shared_assets/figures/{}_{}_{}/".format(year, month, day)

counties = np.array_split(list(county_dict.keys()), args.njobs[0])[args.jobid[0]]
for c in counties:
    curves_window(start, c, n_weeks=3, model_i=35, save_plot=True)
    curves_window_trend(start, c, save_plot=True)