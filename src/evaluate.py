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

with open('../data/comparison.pkl', "rb") as f:
    best_model = pkl.load(f)

with open('../data/counties/counties.pkl', "rb") as f:
    counties = pkl.load(f)

summary = OrderedDict()

# TODO: import the correct last date
# for i, disease in enumerate(diseases):

disease = "covid19"
use_age = best_model[disease]["use_age"]
use_eastwest = best_model[disease]["use_eastwest"]
prediction_region = "germany"

res = load_pred(disease, use_age, use_eastwest)

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

summary = {}

for measure, f in measures.items():
    print("Evaluating {} for disease {}, measure {}".format(
        name, disease, measure))
    measure_df = pd.DataFrame(
        f(
            target.values.astype(
                np.float32).reshape(
                (1, -1)).repeat(
                res["y"].shape[0], axis=0), res["μ"].astype(
                np.float32), res["α"].astype(
                np.float32).reshape(
                    (-1, 1))).mean(
                        axis=0).reshape(
                            target.shape), index=target.index, columns=target.columns)

    summary[measure] = measure_df
    summary[measure + " mean"] = np.mean(measure_df.mean())
    summary[measure + " sd"] = np.std(measure_df.mean())

with open("../data/measures_{}_summary.pkl".format(disease), "wb") as f:
    pkl.dump(summary, f)

del summary
