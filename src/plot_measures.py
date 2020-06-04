import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.patheffects as PathEffects
from config import *
from plot_utils import *
from shared_utils import *
import matplotlib
from matplotlib import pyplot as plt
plt.style.use("ggplot")
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"


measures = ["deviance", "DS score"]

models = [15, 47]

fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(len(measures), len(models), top=0.92,
                    bottom=0.07, left=0.1, right=0.98, hspace=0.3, wspace=0.24)
sp = np.empty((len(measures), len(models)), dtype=object)

with open('../data/measures_covid19_data.pkl', "rb") as f:
    summary = pkl.load(f)

for i, model_id in enumerate(models):

    for j, measure in enumerate(measures):

        sp[j, i] = plt.subplot(grid[j, i])

        if j == 0:
            plt.title("Covid19 - Model ID: {}".format(model_id))

        if i == 0:
            plt.ylabel(measure + "\n" + " distribution", fontsize=22)

        plt.tick_params(axis="both", direction='out', size=6, labelsize=18)

        sns.distplot(summary[str(model_id) + "_" + measure].iloc[0].values)

#         for k, model in enumerate(summary.keys()):
#             sns.distplot(summary[model][measure].mean(), label=model)

#         if (j, i) == (1, 2):
#             plt.legend(["hhh4 model", "proposed model"], fontsize=15)

# plt.show()
plt.savefig("../figures/measures.pdf")
