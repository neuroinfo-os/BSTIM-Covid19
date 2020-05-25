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


with open('../data/counties/counties.pkl', "rb") as f:
    county_info = pkl.load(f)

disease = "covid19"
best_model = {}

print("Evaluating model for {}...".format(disease))

prediction_region = "germany"

#data = load_daily_data(disease, prediction_region, county_info)
#data_train, target_train, data_test, target_test = split_data(data)

#tspan = (target_train.index[0], target_train.index[-1])
waics = {}
# reintroduce combinations as we have the right set of models! // use_eastwest is dummy!
# for (name, (use_interaction, use_report_delay)) in ia_delay_by_name.items():
for (i, _) in enumerate(combinations):
    # load sample trace
    try:
        trace = load_trace_by_i(disease, i)
    except:
        print("Model nr. {} does not exist, skipping...\n".format(i))
        continue
    # load model
    model = load_model_by_i(disease, i)

    with model:
        waics[str(i)] = pm.waic(trace).WAIC

with open('../data/waics.pkl', "wb") as f:
    pkl.dump(waics, f)
