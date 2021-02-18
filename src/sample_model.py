import argparse
import os
import pickle as pkl
import sys
import warnings

import pandas as pd
import pymc3 as pm
from BaseModel import BaseModel
from shared_utils import *


def main(
    start,
    csv_path,
    ia_effect_root,
    use_interactions,
    use_demographics,
    trend_order,
    periodic_order,
    num_samples,
    num_chains,
):

    # Default Values
    number_of_weeks = 3
    use_report_delay = False

    start_date = pd.Timestamp("2020-01-28") + pd.Timedelta(days=start)
    num_cores = num_chains
    disease = "covid19"
    prediction_region = "germany"

    """ Model: """
    use_demographics = True
    trend_order = 1
    periodic_order = 4
    use_report_delay = False

    print(
        "Model Configuration: \n Demographics: {} \n Trend order: {} \n Periodic order: {} \n Report Delay: {}\n".format(
            use_demographics, trend_order, periodic_order, use_report_delay
        )
    )

    filename_params = "../data/mcmc_samples_backup/parameters_{}_{}".format(
        disease, start
    )
    filename_pred = "../data/mcmc_samples_backup/predictions_{}_{}.pkl".format(
        disease, start
    )
    filename_pred_trend = "../data/mcmc_samples_backup/predictions_trend_{}_{}.pkl".format(
        disease, start
    )
    filename_model = "../data/mcmc_samples_backup/model_{}_{}.pkl".format(
        disease, start
    )

    year = str(start_date)[:4]
    month = str(start_date)[5:7]
    day = str(start_date)[8:10]
    ia_effect_path = os.path.join(ia_effect_root, "{}_{}_{}".format(year, month, day))
    ia_effect_files = [
        os.path.join(ia_effect_path, ia_file) for ia_file in os.listdir(ia_effect_path)
    ]

    if len(ia_effect_files) < 100:
        warnings.warn("Currently using less then 100 sample sets for IA Effects")

    # Load data: This must be in the correct place at the moment!
    with open("../data/counties/counties.pkl", "rb") as f:
        county_info = pkl.load(f)

    # Prediciton is 5 days into the future
    days_into_future = 5
    data = load_data_n_weeks(start, number_of_weeks, csv_path, pad=days_into_future)

    # For the simple model, only targets are required!
    _, target_train, _, target_test = split_data(
        data,
        train_start=start_date,
        # test_start=last_day - pd.Timedelta(days=days_into_future+4),
        test_start=start_date + pd.Timedelta(days=number_of_weeks * 7),
        post_test=start_date + pd.Timedelta(days=number_of_weeks * 7 + 10),
    )

    tspan = (target_train.index[0], target_train.index[-1])
    print("CHECK")
    print(start)
    print(start_date)
    print(tspan)
    print(target_train)
    print(
        "training for {} in {} with final model from {} to {}\nWill create files {}, {} and {}".format(
            disease,
            prediction_region,
            *tspan,
            filename_params,
            filename_pred,
            filename_model
        )
    )

    print(os.getcwd())

    model = BaseModel(
        tspan,
        county_info,
        ia_effect_files,
        include_ia=use_interactions,
        include_report_delay=use_report_delay,
        include_demographics=use_demographics,
        trend_poly_order=trend_order,
        periodic_poly_order=periodic_order,
    )

    print("Sampling parameters on the training set.")
    trace = model.sample_parameters(
        target_train,
        samples=num_samples,
        tune=100,
        target_accept=0.95,
        max_treedepth=15,
        chains=num_chains,
        cores=num_cores,
    )

    with open(filename_model, "wb") as f:
        pkl.dump(model.model, f)

    with model.model:
        pm.save_trace(trace, filename_params, overwrite=True)

    print("Sampling predictions on the training and test set.")

    pred = model.sample_predictions(
        target_train.index,
        target_train.columns,
        trace,
        target_test.index,
        average_periodic_feature=False,
        average_all=False,
    )

    pred_trend = model.sample_predictions(
        target_train.index,
        target_train.columns,
        trace,
        target_test.index,
        average_periodic_feature=False,
        average_all=True,
    )

    with open(filename_pred, "wb") as f:
        pkl.dump(pred, f)

    with open(filename_pred_trend, "wb") as f:
        pkl.dump(pred_trend, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Samples a model and its \
                                     predictions starting on Jan 28 2020 \
                                     plus a number of days."
    )

    # this argument is required!
    parser.add_argument(
        "start",
        nargs=1,
        type=int,
        help="start day calculated by Jan 28 2020 + start days",
    )

    parser.add_argument(
        "--csvinputfile",
        nargs=1,
        dest="csvinputfile",
        type=str,
        default=["../data/diseases/covid19.csv"],
        help="Number of chains in MCMC",
    )

    parser.add_argument(
        "--ia_effect_root",
        nargs=1,
        dest="ia_effect_root",
        type=str,
        default=["../data/ia_effect_samples"],
        help="Path to directory with IA effect sample files in date folders",
    )

    parser.add_argument(
        "--use_interactions",
        nargs=1,
        dest="use_interactions",
        type=bool,
        default=[True],
        help="Use IA effects feature",
    )

    parser.add_argument(
        "--use_demographics",
        nargs=1,
        dest="use_demographics",
        type=bool,
        default=[True],
        help="Use demographic feature",
    )

    parser.add_argument(
        "--trend_order",
        nargs=1,
        dest="trend_order",
        type=int,
        default=[1],
        help="Order of local trend polynomial",
    )

    parser.add_argument(
        "--periodic_order",
        nargs=1,
        dest="periodic_order",
        type=int,
        default=[4],
        help="Order of global periodic polynomial",
    )

    parser.add_argument(
        "--num_samples",
        nargs=1,
        dest="num_samples",
        type=int,
        default=[250],
        help="Number of samples per chain in MCMC",
    )

    parser.add_argument(
        "--num_chains",
        nargs=1,
        dest="num_chains",
        type=int,
        default=[4],
        help="Number of chains in MCMC",
    )

    args = parser.parse_args()

    main(
        args.start[0],
        args.csvinputfile[0],
        args.ia_effect_root[0],
        args.use_interactions[0],
        args.use_demographics[0],
        args.trend_order[0],
        args.periodic_order[0],
        args.num_samples[0],
        args.num_chains[0],
    )
