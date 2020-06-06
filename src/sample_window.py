from shared_utils import *
from BaseModel import BaseModel
from config import *
import pymc3 as pm
import pickle as pkl
import pandas as pd
import os
import sys

start = int(sys.argv[2])
number_of_weeks = int(sys.argv[4])
model_i = int(sys.argv[6])

start_date = pd.Timestamp("2020-01-28") + pd.Timedelta(start)

#NOTE: for jureca, extend to the number of available cores (chains and cores!)
num_samples = 250
num_chains = 4
num_cores = num_chains

# whether to sample the parameters or load them 
SAMPLE_PARAMS = True

# whether to sample predictions on training, test or both
# SAMPLE_PREDS = "both" # can be "train", "test" or "both"

disease = "covid19"
prediction_region = "germany"

# model 15 selected by WAICS
# model 35 ohne report delay und mit trend order 1
# model 47 mit trend 4
use_ia, use_report_delay, use_demographics, trend_order, periodic_order = combinations[model_i]
# Print Model Eigenschaften
print("Model {} - IA: {} - RD: {} - DEMO: {} - Trend: {} - Per: {}".format(
    model_i, use_ia, use_report_delay, use_demographics, trend_order, periodic_order

))

# use_interactions, use_report_delay = combinations_ia_report[model_complexity]

filename_params = "../data/mcmc_samples_backup/parameters_{}_model_{}_window_{}_{}".format(disease, model_i, start, number_of_weeks)
filename_pred = "../data/mcmc_samples_backup/predictions_{}_model_{}_window_{}_{}.pkl".format(disease, model_i, start, number_of_weeks)
filename_pred_nowcast = "../data/mcmc_samples_backup/predictions_nowcast_{}_model_{}_window_{}_{}.pkl".format(disease, model_i, start, number_of_weeks)
filename_model = "../data/mcmc_samples_backup/model__{}_model_{}_window_{}_{}.pkl".format(disease, model_i, start, number_of_weeks)

# Load data
with open('../data/counties/counties.pkl', "rb") as f:
    county_info = pkl.load(f)

# pad = days to look into the future
#days_into_future = 5
#data = load_daily_data(disease, prediction_region, county_info, pad=days_into_future)

days_into_future = 5
data = load_daily_data_n_weeks(start, number_of_weeks, disease, prediction_region, county_info, pad=days_into_future)
print("DaysTest")
print(data.index)

first_day = data.index.min()
last_day = data.index.max()

data_train, target_train, data_test, target_test = split_data(
    data,
    train_start=first_day,
    test_start=last_day - pd.Timedelta(days=days_into_future+4),
    post_test=last_day + pd.Timedelta(days=1)
)


tspan = (target_train.index[0], target_train.index[-1])

print("training for {} in {} with final model from {} to {}\nWill create files {}, {} and {}".format(
    disease, prediction_region, *tspan, filename_params, filename_pred, filename_model))

model = BaseModel(tspan,
                  county_info,
                  ["../data/ia_effect_samples/{}_{}.pkl".format(
                        disease, i) for i in range(100)],
                  include_ia=use_ia,
                  include_report_delay=use_report_delay,
                  include_demographics=use_demographics,
                  trend_poly_order=trend_order,
                  periodic_poly_order=periodic_order)

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

    with open(filename_model, "wb") as f:
    	pkl.dump(model.model, f)

    with model.model:
        pm.save_trace(trace, filename_params, overwrite=True)
else:
    print("Load parameters.")
    trace = load_trace_by_i(disease, i)

print("Sampling predictions on the training and test set.")

pred = model.sample_predictions(target_train.index, 
                                target_train.columns, 
                                trace, 
                                target_test.index, 
                                average_periodic_feature=False)

#pred_nowcast = model.sample_predictions(target_train.index, 
#                                        target_train.columns, 
#                                        trace, 
#                                        target_test.index,
#                                        average_periodic_feature=True)

with open(filename_pred, 'wb') as f:
    pkl.dump(pred, f)

#with open(filename_pred_nowcast, "wb") as f:
#    pkl.dump(pred_nowcast, f)
