import pymc3 as pm
import pandas as pd
import matplotlib
import numpy as np
import pickle as pkl
import datetime
import os
from BaseModel import BaseModel
from matplotlib.gridspec import SubplotSpec, GridSpec, GridSpecFromSubplotSpec
import matplotlib.patheffects as PathEffects
import gc
import isoweek
from collections import OrderedDict
from matplotlib import rc
from sampling_utils import *
from shared_utils import *
from pymc3.stats import quantiles
from config import *
from matplotlib import pyplot as plt
plt.style.use('ggplot')

# age_eastwest_by_name = dict(zip(["A","B","C"],combinations_age_eastwest))
disease = "covid19"
prediction_region = "germany"

with open('../data/counties/counties.pkl', "rb") as f:
    county_info = pkl.load(f)

print("Evaluating model for {}...".format(disease))

#TODO: change the dates || import them from somewhere sensible like config
data = load_daily_data(disease, prediction_region, county_info)
data_train, target_train, data_test, target_test = split_data(
    data, train_start=pd.Timestamp(
        2020, 1, 28), test_start=pd.Timestamp(
        2020, 4, 22), post_test=pd.Timestamp(
        2020, 4, 23))

# NOTE: I think the tspan in BaseModel is actually never used?!
tspan = (target_train.index[0], target_train.index[-1])

name = "dev"
use_interactions = False
use_report_delay = True
# load sample trace
trace = load_trace(disease, use_interactions, use_report_delay)

model = BaseModel(tspan,
                  county_info,
                  ["../data/ia_effect_samples/{}_{}.pkl".format(disease,
                                                                i) for i in range(100)],
                  include_ia=use_interactions,
                  include_report_delay=use_report_delay)

filename_pred = "../data/mcmc_samples_backup/predictions_{}_{}_{}.pkl".format(
    disease, use_interactions, use_report_delay)
print("Sampling predictions on the testing set.")
pred = model.sample_predictions(target_train.index, target_train.columns, trace)
with open(filename_pred, 'wb') as f:
    pkl.dump(pred, f)

del trace
del model
