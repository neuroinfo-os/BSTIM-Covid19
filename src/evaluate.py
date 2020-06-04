from config import *
from shared_utils import *
import pickle as pkl
from collections import OrderedDict

measures = {
    "deviance": (
        lambda target_val,
        pred_val,
        alpha_val: deviance_negbin(
            target_val,
            pred_val,
            alpha_val)),
    "DS score": (
        lambda target_val,
        pred_val,
        alpha_val: dss(
            target_val,
            pred_val,
            pred_val +
            pred_val**2 /
            alpha_val))}

with open('../data/counties/counties.pkl', "rb") as f:
    counties = pkl.load(f)


# TODO: import the correct last date
# for i, disease in enumerate(diseases):

disease = "covid19"
# use_age = best_model[disease]["use_age"]
# use_eastwest = best_model[disease]["use_eastwest"]
prediction_region = "germany"

# res = load_pred(disease, use_age, use_eastwest)

data = load_daily_data(disease, prediction_region, counties)
first_day = data.index.min()
last_day = data.index.max()

_, _, _, target = split_data(
    data,
    train_start=first_day,
    test_start=last_day - pd.Timedelta(days=1),
    post_test=last_day + pd.Timedelta(days=1)
)

county_ids = target.columns

# summary -> csv only for mean and sd; ohter get seperate files!
summary = {
    "ID": [],
    "interaction effect": [],
    "report delay": [],
    "demographics": [],
    "trend order": [],
    "period order": [],
    "deviance mean": [],
    "deviance sd": [],
    "DS score mean": [],
    "DS score sd": [],
}

measure_data = {}

for (i,_) in enumerate(combinations):

    try:
        res = load_pred_by_i(disease, i)
    except:
        print("Model nr. {} does not exist, skipping...\n".format(i))
        continue

    use_ia, use_report_delay, use_demographics, trend_order, periodic_order = combinations[i]
    summary["ID"].append(i)
    summary["interaction effect"].append(use_ia)
    summary["report delay"].append(use_report_delay)
    summary["demographics"].append(use_demographics)
    summary["trend order"].append(trend_order)
    summary["period order"].append(periodic_order)


    for measure, f in measures.items():
        print("Evaluating model {} for disease {}, measure {}".format(
            i, disease, measure))
        measure_df = pd.DataFrame(
            f(target.values.astype(np.float32).reshape((1, -1)).repeat(res["y"].shape[0], axis=0),
              res["μ"].astype(np.float32),
              res["α"].astype(np.float32).reshape((-1, 1))).mean(axis=0).reshape(target.shape),
              index=target.index, columns=target.columns)

        measure_data[str(i) + "_" + measure] = measure_df

        summary[measure + " mean"].append(np.mean(measure_df.mean()))
        summary[measure + " sd"].append(np.std(measure_df.mean()))

df = pd.DataFrame(summary)
df.to_csv("../data/measures_{}_summary.csv".format(disease), sep=",")

with open("../data/measures_{}_data.pkl".format(disease), "wb") as f:
    pkl.dump(measure_data, f)

del summary
