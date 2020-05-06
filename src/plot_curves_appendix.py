from matplotlib import pyplot as plt
from config import *
from shared_utils import *
from plot_utils import *
import pickle as pkl
import numpy as np
from collections import OrderedDict
from pymc3.stats import quantiles

def curves_appendix(use_interactions=True, use_report_delay=True, save_plot=False):
    with open('../data/counties/counties.pkl', "rb") as f:
        counties = pkl.load(f)

    countyByName = OrderedDict([('Düsseldorf', '05111'), ('Recklinghausen', '05562'),
                                ("Hannover", "03241"), ("Hamburg", "02000"),
                                ("Berlin-Mitte", "11001"), ("Osnabrück", "03404"),
                                ("Frankfurt (Main)", "06412"),
                                ("Görlitz", "14626"), ("Stuttgart", "08111"),
                                ("Potsdam", "12054"), ("Köln", "05315"),
                                ("Aachen", "05334"), ("Rostock", "13003"),
                                ("Flensburg", "01001"), ("Frankfurt (Oder)", "12053"),
                                ("Lübeck", "01003"), ("Münster", "05515"),
                                ("Berlin Neukölln", "11008"), ('Göttingen', "03159"),
                                ("Cottbus", "12052"), ("Erlangen", "09562"),
                                ("Regensburg", "09362"), ("Bayreuth", "09472"),
                                ("Bautzen", "14625"), ('Nürnberg', '09564'),
                                ('München', '09162'), ("Würzburg", "09679"),
                                ("Deggendorf", "09271"), ("Ansbach", "09571"),
                                ("Rottal-Inn", "09277"), ("Passau", "09275"),
                                ("Schwabach", "09565"), ("Memmingen", "09764"),
                                ("Erlangen-Höchstadt",
                                "09572"), ("Nürnberger Land", "09574"),
                                ('Roth', "09576"), ('Starnberg', "09188"),
                                ('Berchtesgadener Land',
                                "09172"), ('Schweinfurt', "09678"),
                                ("Augsburg", "09772"), ('Neustadt a.d.Waldnaab', "09374"),
                                ("Fürstenfeldbruck", "09179"), ('Rosenheim', "09187"),
                                ("Straubing", "09263"), ("Erding", "09177"),
                                ("Tirschenreuth", "09377"), ('Miltenberg', "09676"),
                                ('Neumarkt i.d.OPf.', "09373"),
                                ('Heinsberg', "05370"), ('Landsberg am Lech', "09181"),
                                ('Rottal-Inn', "09277"), ("Tübingen", "08416"), 
                                ("Augsburg", "09772"), ("Bielefeld", "05711")])

    plot_county_names = {
        "covid19": [
            "Düsseldorf",
            "Heinsberg",
            "Hannover",
            "München",
            "Hamburg",
            "Berlin-Mitte",
            "Osnabrück",
            "Frankfurt (Main)",
            "Görlitz",
            "Stuttgart",
            "Landsberg am Lech",
            "Köln",
            "Rottal-Inn",
            "Rostock",
            "Flensburg",
            "Frankfurt (Oder)",
            "Lübeck",
            "Münster",
            "Berlin Neukölln",
            "Göttingen",
            "Bielefeld",
            "Tübingen",
            "Augsburg",
            "Bayreuth",
            "Nürnberg"]}

    # colors for curves
    C1 = "#D55E00"
    C2 = "#E69F00"
    C3 = "#0073CF"

    i = 0
    disease = "covid19"
    prediction_region = "germany"

    data = load_daily_data(disease, prediction_region, county_info)
    first_day = data.index.min()
    last_day = data.index.max()

    _, target, _, _ = split_data(
        data,
        train_start=first_day,
        test_start=last_day - pd.Timedelta(days=1),
        post_test=last_day + pd.Timedelta(days=1))

    county_ids = target.columns

    # Load our prediction samples
    res = load_pred(disease, use_interactions, use_report_delay)
    n_days = ((last_day - pd.Timedelta(days=1)) - first_day).days

    prediction_samples = np.reshape(res['y'], (res['y'].shape[0], n_days, -1)) 
    prediction_quantiles = quantiles(prediction_samples, (5, 25, 75, 95))

    prediction_mean = pd.DataFrame(
        data=np.mean(
            prediction_samples,
            axis=0),
        index=target.index,
        columns=target.columns)
    prediction_q25 = pd.DataFrame(
        data=prediction_quantiles[25],
        index=target.index,
        columns=target.columns)
    prediction_q75 = pd.DataFrame(
        data=prediction_quantiles[75],
        index=target.index,
        columns=target.columns)
    prediction_q5 = pd.DataFrame(
        data=prediction_quantiles[5],
        index=target.index,
        columns=target.columns)
    prediction_q95 = pd.DataFrame(
        data=prediction_quantiles[95],
        index=target.index,
        columns=target.columns)

    fig = plt.figure(figsize=(12, 12))
    grid = plt.GridSpec(5, 5, top=0.90, bottom=0.11,
                        left=0.07, right=0.92, hspace=0.2, wspace=0.3)

    for j, name in enumerate(plot_county_names[disease]):
        # TODO: this should be incapsulated as plot_curve(county) (, days)

        ax = fig.add_subplot(
            grid[np.unravel_index(list(range(25))[j], (5, 5))])

        county_id = countyByName[name]
        dates = [pd.Timestamp(day) for day in target.index.values]
        days = [ (day - min(dates)).days for day in dates]

        # plot our predictions w/ quartiles
        p_pred = ax.plot(
            days,
            prediction_mean[county_id],
            "-",
            color=C1,
            linewidth=2.0,
            zorder=4)
        p_quant = ax.fill_between(
            days,
            prediction_q25[county_id],
            prediction_q75[county_id],
            facecolor=C2,
            alpha=0.5,
            zorder=1)
        ax.plot(
            days,
            prediction_q25[county_id],
            ":",
            color=C2,
            linewidth=2.0,
            zorder=3)
        ax.plot(
            days,
            prediction_q75[county_id],
            ":",
            color=C2,
            linewidth=2.0,
            zorder=3)

        # plot ground truth
        p_real = ax.plot(days, target[county_id], "k.")

        ax.set_title(name, fontsize=18)
        ax.set_xticks(days[::5])
        ax.tick_params(axis="both", direction='out', size=2, labelsize=14)
        plt.setp(ax.get_xticklabels(), visible=False)
        if j >= 19:
            plt.setp(ax.get_xticklabels(), rotation=60)
            plt.setp(ax.get_xticklabels()[::2], visible=True)

        ax.autoscale(False)
        p_quant2 = ax.fill_between(
            days,
            prediction_q5[county_id],
            prediction_q95[county_id],
            facecolor=C2,
            alpha=0.25,
            zorder=0)
        ax.plot(days, prediction_q5[county_id], ":",
                    color=C2, alpha=0.5, linewidth=2.0, zorder=1)
        ax.plot(days, prediction_q95[county_id], ":",
                    color=C2, alpha=0.5, linewidth=2.0, zorder=1)

    plt.legend([p_real[0], p_pred[0], p_quant, p_quant2],
            ["reported", "predicted",
                "25\%-75\% quantile", "5\%-95\% quantile"],
            fontsize=16, ncol=5, loc="upper center", bbox_to_anchor=(0, -0.01, 1, 1),
            bbox_transform=plt.gcf().transFigure)
    fig.text(0.5, 0.02, "Time [days since Jan. 28]", ha='center', fontsize=22)
    fig.text(0.01, 0.46, "Reported/predicted infections",
            va='center', rotation='vertical', fontsize=22)
    
    if save_plot:
        plt.savefig("../figures/curves_{}_appendix_{}_{}.pdf".format(disease, use_interactions, use_report_delay))

    return plt

if __name__ == "__main__":

    combinations_ia_report = [(False,False), (False,True), (True,False), (True,True)]
    for i in range(4):
        _ = curves_appendix(combinations_ia_report[i][0], combinations_ia_report[i][1],save_plot=True)

