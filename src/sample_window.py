from shared_utils import *
from BaseModel import BaseModel
from config import *
import pymc3 as pm
import pickle as pkl
import pandas as pd
import os
import sys
from config_window import start as startixs

start = int(os.environ["SGE_DATE_ID"])
number_of_weeks = 3# int(sys.argv[4])
model_i = 35

start_date = pd.Timestamp("2020-01-28") + pd.Timedelta(days=start)

#NOTE: for jureca, extend to the number of available cores (chains and cores!)
num_samples = 250
num_chains = 4
num_cores = num_chains

# whether to sample the parameters or load them 
SAMPLE_PARAMS = True

PERMUTATION_STUDY=False

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

filename_params = "../data/mcmc_samples_backup/parameters_{}_{}".format(disease,start)
filename_pred = "../data/mcmc_samples_backup/predictions_{}_{}.pkl".format(disease, start)
#filename_pred_nowcast = "../data/mcmc_samples_backup/predictions_nowcast_{}_model_{}_window_{}_{}.pkl".format(disease, model_i, start, number_of_weeks)
filename_pred_trend = "../data/mcmc_samples_backup/predictions_trend_{}_{}.pkl".format(disease, start )
filename_model = "../data/mcmc_samples_backup/model_{}_{}.pkl".format(disease, start)
if PERMUTATION_STUDY:
    filename_params = "../data/mcmc_samples_backup/permutation_studies/parameters_{}_{}".format(disease,start)
    filename_pred = "../data/mcmc_samples_backup/permutation_studies/predictions_{}_{}.pkl".format(disease, start)
    #filename_pred_nowcast = "../data/mcmc_samples_backup/predictions_nowcast_{}_model_{}_window_{}_{}.pkl".format(disease, model_i, start, number_of_weeks)
    filename_pred_trend = "../data/mcmc_samples_backup/permutation_studies/predictions_trend_{}_{}.pkl".format(disease, start )
    filename_model = "../data/mcmc_samples_backup/permutation_studies/model_{}_{}.pkl".format(disease, start)


import os
print(os.getcwd())
print('../data/counties/counties.pkl')

# Load data
with open('../data/counties/counties.pkl', "rb") as f:
    county_info = pkl.load(f)

# pad = days to look into the future
#days_into_future = 5
#data = load_daily_data(disease, prediction_region, county_info, pad=days_into_future)

days_into_future = 5
data = load_daily_data_n_weeks(start, number_of_weeks, disease, prediction_region, county_info, pad=days_into_future, permute=PERMUTATION_STUDY)
print(data)
first_day = data.index.min()
last_day = data.index.max()

data_train, target_train, data_test, target_test = split_data(
    data,
    train_start=first_day,
    test_start=last_day - pd.Timedelta(days=days_into_future+4),
    post_test=last_day + pd.Timedelta(days=1)
)

tspan = (target_train.index[0], target_train.index[-1])
print("CHECK")
print(start)
print(start_date)
print(first_day)
print(target_train)
print("training for {} in {} with final model from {} to {}\nWill create files {}, {} and {}".format(
    disease, prediction_region, *tspan, filename_params, filename_pred, filename_model))

print(os.getcwd())
print('../data/iaeffect')

year = str(start_date)[:4]
month = str(start_date)[5:7]
day = str(start_date)[8:10]


model = BaseModel(tspan,
                  county_info,
                  ["../data/ia_effect_samples/{}_{}_{}/{}_{}.pkl".format(year, month, day,disease, i) for i in range(100)],
                  include_ia=use_ia,
                  include_report_delay=use_report_delay,
                  include_demographics=use_demographics,
                  trend_poly_order=trend_order,
                  periodic_poly_order=periodic_order)
if PERMUTATION_STUDY:
    model = BaseModel(tspan,
                  county_info,
                  ["../data/ia_effect_samples/permutation_studies/{}_{}_{}/{}_{}.pkl".format(year, month, day,disease, i) for i in range(100)],
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
        cores=num_cores,
        window=True)

    with open(filename_model, "wb") as f:
    	pkl.dump(model.model, f)

    with model.model:
        pm.save_trace(trace, filename_params, overwrite=True)
else:
    print("Load parameters.")
    trace = load_trace_window(disease,model_i, start, number_of_weeks ) 

print("Sampling predictions on the training and test set.")

pred = model.sample_predictions(target_train.index, 
                                target_train.columns, 
                                trace, 
                                target_test.index, 
                                average_periodic_feature=False,
                                average_all=False,
                                window=True)

pred_trend = model.sample_predictions(target_train.index, 
                                target_train.columns, 
                                trace, 
                                target_test.index, 
                                average_periodic_feature=False,
                                average_all=True,
                                window=True)


with open(filename_pred, 'wb') as f:
    pkl.dump(pred, f)

with open(filename_pred_trend, "wb") as f:
    pkl.dump(pred_trend, f)
