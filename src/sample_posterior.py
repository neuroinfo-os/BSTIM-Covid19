from shared_utils import *
from BaseModel import BaseModel
from config import *
import pymc3 as pm
import pickle as pkl
import pandas as pd
import os

# SGE_TASK_ID=4 python sample_posterior.py
i = int(os.environ["SGE_TASK_ID"])-1

#NOTE: for jureca, extend to the number of available cores (chains and cores!)
num_samples = 250
# num_sample = 1
num_chains = 4
# num_chains = 1
num_cores = num_chains

# whether to sample the parameters or load them 
SAMPLE_PARAMS = False

# whether to sample predictions on training, test or both
SAMPLE_PREDS = "both" # can be "train", "test" or "both"

model_complexity, disease = combinations[i]
use_interactions, use_report_delay = combinations_ia_report[model_complexity]
prediction_region = "germany"

filename_params = "../data/mcmc_samples_backup/parameters_{}_{}_{}".format(
    disease, use_interactions, use_report_delay)
filename_pred = "../data/mcmc_samples_backup/predictions_{}_{}_{}.pkl".format(
    disease, use_interactions, use_report_delay)
filename_model = "../data/mcmc_samples_backup/model_{}_{}_{}.pkl".format(
    disease, use_interactions, use_report_delay)

# Load data
with open('../data/counties/counties.pkl', "rb") as f:
    county_info = pkl.load(f)

data = load_daily_data(disease, prediction_region, county_info)
data_train, target_train, data_test, target_test = split_data(
    data,
    train_start=pd.Timestamp(2020, 1, 28),
    test_start=pd.Timestamp(2020, 4, 12),
    post_test=pd.Timestamp(2020, 4, 23)
)

tspan = (target_train.index[0], target_train.index[-1])

print("training for {} in {} with model complexity {} from {} to {}\nWill create files {}, {} and {}".format(
    disease, prediction_region, model_complexity, *tspan, filename_params, filename_pred, filename_model))

model = BaseModel(tspan,
                  county_info,
                  ["../data/ia_effect_samples/{}_{}.pkl".format(disease,
                                                                i) for i in range(100)],
                  include_ia=use_interactions,
                  include_report_delay=use_report_delay)

if SAMPLE_PARAMS:
    print("Sampling parameters on the training set.")
    trace = model.sample_parameters(
        target_train,
        samples=num_samples,
        tune=100,
        target_accept=0.95,
        max_treedepth=15,
        chains=num_chains,
        cores=num_cores)
else:
    print("Load parameters.")
    trace = load_trace(disease, use_interactions, use_report_delay)

with open(filename_model, "wb") as f:
    pkl.dump(model.model, f)

with model.model:
    pm.save_trace(trace, filename_params, overwrite=True)

if SAMPLE_PREDS == "train":
    print("Sampling predictions on the training set.")
    filename_pred = "../data/mcmc_samples_backup/predictions_train_{}_{}_{}.pkl".format(
                                            disease, use_interactions, use_report_delay)
    pred = model.sample_predictions(target_train.index, target_train.columns, trace)
    with open(filename_pred, 'wb') as f:
        pkl.dump(pred, f)

elif SAMPLE_PREDS == "test":
    print("Sampling predictions on the test set.")
    filename_pred = "../data/mcmc_samples_backup/predictions_test_{}_{}_{}.pkl".format(
                                            disease, use_interactions, use_report_delay)
    pred = model.sample_predictions(target_test.index, target_test.columns, trace)
    with open(filename_pred, 'wb') as f:
        pkl.dump(pred, f)

elif SAMPLE_PREDS == "both":
    print("Sampling predictions on all data.")
    filename_pred = "../data/mcmc_samples_backup/predictions_total_{}_{}_{}.pkl".format(
                                            disease, use_interactions, use_report_delay)

    # Split data new.
    data_train, target_train, data_test, target_test = split_data(
    data,
    train_start=pd.Timestamp(2020, 1, 28),
    test_start=pd.Timestamp(2020, 4, 22),
    post_test=pd.Timestamp(2020, 4, 23)
)
    pred = model.sample_predictions(target_train.index, target_train.columns, trace)
    with open(filename_pred, 'wb') as f:
        pkl.dump(pred, f)

else:
    print("Couldn't sample predictions. Unknown flag.")
    

