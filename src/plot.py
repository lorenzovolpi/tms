import itertools as IT
import os
from argparse import ArgumentParser
from collections import defaultdict
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cap.error import vanilla_acc
from cap.plot.utils import save_figure
from cap.utils.commons import contingency_table
from pandas.core.algorithms import rank

import env
import main
from config import get_acc_names, get_evaluation_acc, get_existing_dataset_names
from data import PretainInfo, load_info_paths
from results import Results
from util import decorate_dataset

method_map = {
    "LR": "$\\emptyset$-LR",
    "kNN": "$\\emptyset$-$k$NN",
    "SVM": "$\\emptyset$-SVM",
    "MLP": "$\\emptyset$-MLP",
    "SVM-t": "$\\emptyset$-TSVM",
    "Naive-LR": "IMS-LR",
    "Naive-kNN": "IMS-$k$NN",
    "Naive-SVM": "IMS-SVM",
    "Naive-MLP": "IMS-MLP",
    "Naive": "IMS-All",
    "O-LEAP(KDEy)": "TMS-All",
    "oracle": "Oracle",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
    "stanfordnlp__imdb": "imdb",
    "fancyzhx__yelp_polarity": "yelp-polarity",
    "stanfordnlp__sst2": "sst2",
    "fancyzhx__ag_news": "ag-news",
    "fancyzhx__dbpedia_14": "dbpedia-14",
    "community-datasets__yahoo_answers_topics": "yahoo-answers-topics",
    "yelp_reviews": "yelp-reviews",
    "ag_news-lt": "ag-news-lt",
    "dbpedia_14-lt": "dbpedia-14-lt",
    "ethz__food101": "food101",
}

clsf_map = {
    "microsoft__resnet-50": "resnet-50",
    "facebook__convnext-tiny-224": "convnext-tiny",
    "google__efficientnet-b0": "efficientnet-b0",
    "google__vit-base-patch16-224": "vit-base",
    "microsoft__swin-tiny-patch4-window7-224": "swin-tiny",
    "google-bert__bert-base-uncased": "bert",
    "FacebookAI__roberta-base": "roberta",
    "distilbert__distilbert-base-uncased": "distilbert",
    "microsoft__deberta-v3-base": "deberta-v3",
    "google__electra-base-discriminator": "electra",
}


def get_binned_values(shifts, n_bins):
    # sh_min, sh_max = np.min(df.loc[:, "shifts"]), np.max(df.loc[:, "shifts"])
    # val_min, val_max = 0, 1
    val_min, val_max = np.min(shifts), np.max(shifts)
    bins = np.linspace(val_min, val_max, n_bins + 1)
    binwidth = (val_max - val_min) / n_bins
    vals_bin_idx = np.digitize(shifts, bins, right=True)
    vals_bin_idx = np.where(vals_bin_idx == 0, 1, vals_bin_idx) - 1
    bins = bins[1:] - binwidth / 2
    # bins = np.hstack([bins, [val_max]])
    return bins[vals_bin_idx]


def get_binned_values_from_df(df, val_name, n_bins):
    # sh_min, sh_max = np.min(df.loc[:, "shifts"]), np.max(df.loc[:, "shifts"])
    val_min, val_max = 0, 1
    bins = np.linspace(val_min, val_max, n_bins + 1)
    binwidth = (val_max - val_min) / n_bins
    vals_bin_idx = np.digitize(df.loc[:, val_name], bins, right=True)
    vals_bin_idx = np.where(vals_bin_idx == 0, 1, vals_bin_idx) - 1
    bins = bins[1:] - binwidth / 2
    # bins = np.hstack([bins, [val_max]])
    return bins[vals_bin_idx]


def get_shifts(test_accs: np.ndarray, train_acc: np.ndarray):
    return np.linalg.norm(test_accs - train_acc.reshape(1, -1), axis=1) / np.sqrt(2)


def get_palette():
    color_map = {
        "Naive-LR": 0,
        "Naive-kNN": 1,
        "Naive-SVM": 2,
        "Naive-MLP": 6,
        "Naive": 3,
        "O-LEAP(KDEy)": 5,
        "oracle": "black",
    }
    # palette = sns.color_palette()
    # return palette[:5] + palette[9:]
    _palette = sns.color_palette("Paired")
    return {method_map.get(n, n): (_palette[id] if isinstance(id, int) else id) for n, id in color_map.items()}


def plots(experiment: Literal["transd", "hoptim"]):
    domain = "classic"
    if experiment == "transd":
        ea_label = "estim_accs"
    else:
        print(f"Invalid experiment '{experiment}'; aborting.")
        return

    accs = get_acc_names()
    datasets = get_existing_dataset_names(experiment, domain)
    methods = [
        "oracle",
        "Naive-LR",
        "Naive-kNN",
        "Naive-SVM",
        "Naive-MLP",
        "Naive",
        "O-LEAP(KDEy)",
    ]

    base_dir = os.path.join(env.root_dir, experiment)
    plot_dir = os.path.join(env.root_dir, "plots", experiment)
    os.makedirs(plot_dir, exist_ok=True)

    dashes_dict = {method_map.get(_m, _m): "" for _m in methods}
    for _m in ["oracle"]:
        dashes_dict[method_map.get(_m, _m)] = (4, 2)

    for acc in accs:
        res = []
        for dataset in datasets:
            res.append(
                Results.load(base_dir=base_dir, acc_name=acc, dataset=dataset, domain=domain)
                # .split_by_shift(prevs=0.5)
                # TODO: split model_selection in submethods
                .model_selection(oracle=True, rank_label=ea_label)
                .filter_column_values("method", "isin", methods)
                .select_columns(["method", "dataset", "true_accs", "shifts"])
            )
            print(f"[{acc}] {dataset} loaded")

        res = Results.concat(res, axis=0)
        res = (
            res.add_column("shift_bins", get_binned_values_from_df(res.df, "shifts", n_bins=20))
            .map_column_values("method", method_map)
            .map_column_values("dataset", dataset_map)
            .apply_to_column("dataset", decorate_dataset)
        )

        _methods = [method_map.get(m, m) for m in methods if m != "oracle"]
        _datasets = [decorate_dataset(dataset_map.get(d, d)) for d in datasets]

        plot = sns.lineplot(
            res.filter_column_values("method", "ne", "Oracle").df,
            x="shift_bins",
            y="true_accs",
            hue="method",
            hue_order=_methods,
            # style="method",
            errorbar="se",
            err_style="bars",
            err_kws=dict(capsize=2, capthick=1),
            linewidth=2,
            palette=get_palette(),
        )

        # add oracle line
        oracle_label = method_map.get("oracle", "oracle")
        oracle_df = (
            res.filter_column_values("method", "eq", "Oracle")
            .select_columns(["shift_bins", "true_accs"])
            .df.groupby("shift_bins")
            .mean()
            .reset_index()
        )
        plt.plot(
            oracle_df["shift_bins"],
            oracle_df["true_accs"],
            color="black",
            linestyle="--",
            linewidth=1,
            label=oracle_label,
        )

        # Add oracle to the legend at the top
        handles, labels = plt.gca().get_legend_handles_labels()
        oracle_index = labels.index(oracle_label)
        handles = [handles[oracle_index]] + handles[:oracle_index] + handles[oracle_index + 1 :]
        labels = [labels[oracle_index]] + labels[:oracle_index] + labels[oracle_index + 1 :]
        plt.legend(handles=handles, labels=labels)

        # add bin density bars to the background
        _bin_arr = res.filter_column_values("method", "ne", "Oracle").df["shift_bins"].to_numpy()
        bin_vals, bin_counts = np.unique(_bin_arr, return_counts=True)
        true_vals = (
            res.select_columns(["method", "shifts", "true_accs"])
            .df.groupby(["method", "shifts"])
            .mean()["true_accs"]
            .to_numpy()
        )
        v_min, v_max = 0.25, 1.0
        bin_densities = v_min + bin_counts / bin_counts.max() * (v_max - v_min)
        # v_max, v_min, d_max, d_min = true_vals.max(), true_vals.min(), bin_densities.max(), bin_densities.min()
        # scaled_densities = v_min + (bin_densities - d_min) / (d_max - d_min) * (v_max - v_min)

        # print(v_max, v_min, scaled_densities)
        # print(list(zip(bin_vals, scaled_densities)))
        plot.bar(
            bin_vals,
            bin_densities,
            width=(bin_vals[1] - bin_vals[0]) * 0.95,
            alpha=0.1,
            color="blue",
            label="density",
        )

        plot.set_ylim(v_min - 0.02, v_max + 0.02)

        # Set plot axis ratio
        plt.gca().set_aspect(0.6)

        # Set axes labels
        plot.set_xlabel("Amount of PPS")
        plot.set_ylabel("Accuracy")

        sns.move_legend(plot, "center left", bbox_to_anchor=(1.05, 0.5), title=None, frameon=False)

        save_figure(plot, plot_dir, f"{acc}_shift_{experiment}")


def trueacc_by_shift(pargs):
    experiment = "tas"
    domain = pargs.domain

    plot_dir = os.path.join(env.root_dir, "plots", experiment)
    os.makedirs(plot_dir, exist_ok=True)

    info_paths = load_info_paths(domain)
    dc_map = defaultdict(list)
    for path in info_paths:
        p = PretainInfo.load(path, fast=True)
        dc_map[p.d_info.name].append(p)

    for dataset in list(dc_map.keys()):
        d_name = dataset_map.get(dataset, dataset)
        ps = dc_map[dataset]
        data = defaultdict(list)
        for p in ps:
            D = p.load_dataset_bundle()
            h = p.load_pretrained_classifier(D)

            true_cts = [contingency_table(Ui.y, np.argmax(h.predict_proba(Ui.X), axis=-1)) for Ui in D.test_prot()]
            shifts = get_shifts(np.array([Ui.prevalence() for Ui in D.test_prot()]), D.L_prevalence)
            shift_bins = get_binned_values(shifts, n_bins=20)

            for acc_name in get_acc_names():
                acc = get_evaluation_acc(acc_name, ps[0].d_info.n_classes > 2)
                true_accs = [acc(ct) for ct in true_cts]

                c_name = clsf_map.get(p.h_info.name, p.h_info.name)
                p_data_list = [
                    dict(true_accs=ta, shift_bins=sb, classifier=c_name) for ta, sb in zip(true_accs, shift_bins)
                ]
                data[acc_name].extend(p_data_list)

        for acc_name, acc_data in data.items():
            df = pd.DataFrame(acc_data)

            plot = sns.lineplot(
                df,
                x="shift_bins",
                y="true_accs",
                hue="classifier",
                # style="method",
                errorbar="se",
                err_style="bars",
                err_kws=dict(capsize=2, capthick=1),
                linewidth=2,
                # palette=get_palette(),
            )

            # add bin density bars to the background
            _bin_arr = df["shift_bins"].to_numpy()
            bin_vals, bin_counts = np.unique(_bin_arr, return_counts=True)
            # v_min, v_max = 0.25, 1.0
            gb_df = df.groupby(["shift_bins", "classifier"]).mean()
            v_min, v_max = gb_df["true_accs"].min(), gb_df["true_accs"].max()
            v_min = max(0, v_min - 0.2 * (v_max - v_min))
            v_max = min(1, v_max + 0.2 * (v_max - v_min))
            bin_densities = v_min + bin_counts / bin_counts.max() * (v_max - v_min)
            plot.bar(
                bin_vals,
                bin_densities,
                width=(bin_vals[1] - bin_vals[0]) * 0.95,
                alpha=0.1,
                color="blue",
                label="density",
            )

            plot.set_ylim(v_min, v_max)

            # Set plot axis ratio
            # plt.gca().set_aspect(0.6)

            # Set axes labels
            plot.set_xlabel("Amount of PPS")
            plot.set_ylabel("Accuracy")

            sns.move_legend(plot, "center left", bbox_to_anchor=(1.05, 0.5), title=None, frameon=False)

            save_figure(plot, plot_dir, f"{domain}_{acc_name}_{d_name}")
            print(f"[{acc_name}] {d_name} done.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        action="store",
        help="The type of experiment for which to generate tables",
        choices=["transd"],
        default="transd",
    )
    parser.add_argument("--tas", action="store_true", help="Plot the True Accuracy by Shift results")
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    parser.add_argument("--classic", action="store_const", dest="domain", const="classic")
    pargs = parser.parse_args()

    if pargs.domain is None:
        raise ValueError("Please specify a domain.")

    if pargs.tas:
        trueacc_by_shift(pargs)
    else:
        plots(pargs.experiment)
