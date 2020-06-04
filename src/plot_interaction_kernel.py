import matplotlib
matplotlib.use('TkAgg')
from sampling_utils import *
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.patheffects as PathEffects
from config import *
from plot_utils import *
from shared_utils import *
import matplotlib
from matplotlib import pyplot as plt
theano.config.compute_test_value = 'off'

def interaction_kernel(model_i=15, save_plot=False):
    theano.config.compute_test_value = 'off'
    ii=15

    use_ia, use_report_delay, use_demographics, trend_order, periodic_order = combinations[model_i]

    if not use_ia:
        raise ValueError("Model nr. {} does not include an interaction kernel".format(model_i))

    plt.style.use("ggplot")
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    plt.rcParams["font.family"] = "Bitstream Charter"

    # with open('../data/comparison.pkl', "rb") as f:
    #     best_model = pkl.load(f)

    C1 = "#D55E00"
    C2 = "#E69F00"
    C3 = C2  # "#808080"

    disease = 'covid19'
    diseases = ['covid19']
    fig = plt.figure(figsize=(13, 8))
    #fig.suptitle("Learned interaction kernels and temporal contributions", fontsize=20)
    grid = plt.GridSpec(
        1,
        2 * len(diseases),
        top=0.92,
        bottom=0.1,
        left=0.09,
        right=0.97,
        hspace=0.28,
        wspace=0.1,
        width_ratios=[
            10,
            1])
    #         10,
    #         1,
    #         10,
    #         1])

    loc0 = np.array([[11.0767, 49.4521]])
    locs = np.hstack((np.zeros(200).reshape((-1, 1)),
                    np.linspace(-0.67, 0.67, 200).reshape((-1, 1)))) + loc0
    t0 = np.array([0.0])
    ts = np.linspace(0 * 24 * 3600, 5 * 24 * 3600, 200) # plot for 5 weeks / 5 days instead?

    def temporal_bfs(x): return bspline_bfs(
        x, np.array([0, 0, 1, 2, 3, 4, 5]) * 24 * 3600.0, 2)


    def spatial_bfs(x): return [gaussian_bf(x, σ)
                                for σ in [6.25, 12.5, 25.0, 50.0]]

    basis_kernels = build_ia_bfs(temporal_bfs, spatial_bfs)

    res = np.empty((200, 200, 16))
    for j in range(200):
        for k in range(200):
            res[j, k, :] = basis_kernels(ts[k:k + 1], locs[j:j + 1, :], t0, loc0)


    i = 0
    

    #trace = load_trace_by_i(disease, ii)
    trace = load_final_trace()

    kernel_samples = res.dot(trace["W_ia"].T)

    K_mean = kernel_samples.mean(axis=2)
    K_samp1 = kernel_samples[:, :, np.random.randint(kernel_samples.shape[-1])]
    K_samp2 = kernel_samples[:, :, np.random.randint(kernel_samples.shape[-1])]

    vmax = np.max([K_mean.max(), -K_mean.max(),])
    
    cNorm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap="RdBu_r")
    scalarMap.set_array([])

    ax_mean = fig.add_subplot(grid[0, i * 2])
    ax_mean.contourf(ts / (3600 * 24),
                    (locs - loc0)[:,
                                1] * 111,
                    K_mean,
                    50,
                    cmap="RdBu_r",
                    vmin=-vmax,
                    vmax=vmax)

    #ax_sample1 = fig.add_subplot(grid[1, i * 2], sharex=ax_mean)
    #ax_sample1.contourf(ts /
    #                    (3600 *
    #                    24), (locs -
    #                        loc0)[:, 1] *
    #                    111, K_samp1, 50, cmap="RdBu_r", vmin=-
    #                    vmax, vmax=vmax)

    #ax_sample2 = fig.add_subplot(grid[2, i * 2], sharex=ax_mean)
    #ax_sample2.contourf(ts /
    #                    (3600 *
    #                    24), (locs -
    #                        loc0)[:, 1] *
    #                    111, K_samp2, 50, cmap="RdBu_r", vmin=-
    #                    vmax, vmax=vmax)

    ax_c = fig.add_subplot(grid[:, i * 2 + 1])
    # shift colorbar by 2% to the left
    ax_c.set_position(matplotlib.transforms.Bbox(
        ax_c.get_position() + np.array([-0.03, 0])))
    ax_c.tick_params(labelsize=18, length=6)
    plt.colorbar(mappable=scalarMap, cax=ax_c)

    ax_mean.set_aspect("auto")

    #ax_mean.set_title("campylob." if disease ==
    #               "campylobacter" else disease, fontsize=22)
    ax_mean.set_xlabel("time [days]", fontsize=22)


    if i == 0:
        ax_mean.set_ylabel("mean kernel [km]", fontsize=22)
        #ax_sample1.set_ylabel("sampled kernel [km]", fontsize=22)
        #ax_sample2.set_ylabel("sampled kernel [km]", fontsize=22)

    ax_mean.tick_params(labelbottom=True, labelleft=(
        i == 0), labelsize=18, length=6)
    #ax_sample1.tick_params(labelbottom=False, labelleft=(
       # i == 0), labelsize=18, length=6)
    #ax_sample2.tick_params(labelbottom=True, labelleft=(
       # i == 0), labelsize=18, length=6)

    #fig.text(0, 1 + 0.025, r"$\textbf{" + str(i + 1) + r"A}$",
    #        fontsize=22, transform=ax_mean.transAxes)
    #fig.text(0, 1 + 0.025, r"$\textbf{" + str(i + 1) + r"B}$",
    #        fontsize=22, transform=ax_sample1.transAxes)
    #fig.text(0, 1 + 0.025, r"$\textbf{" + str(i + 1) + r"C}$",
    #x        fontsize=22, transform=ax_sample2.transAxes)

    if save_plot:
        fig.savefig("../figures/interaction_kernels_{}.pdf".format(ii))

    #return fig

if __name__ == "__main__":

    _ = interaction_kernel(True, save_plot=True)
    #_ = interaction_kernel(False, save_plot=True)
