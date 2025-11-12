import os
from argparse import ArgumentParser
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cap.plot.utils import save_figure

import env
from config import get_acc_names, get_all_dataset_names
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
}


def get_binned_values(df, val_name, n_bins):
    # sh_min, sh_max = np.min(df.loc[:, "shifts"]), np.max(df.loc[:, "shifts"])
    val_min, val_max = 0, 1
    bins = np.linspace(val_min, val_max, n_bins + 1)
    binwidth = (val_max - val_min) / n_bins
    vals_bin_idx = np.digitize(df.loc[:, val_name], bins, right=True)
    vals_bin_idx = np.where(vals_bin_idx == 0, 1, vals_bin_idx) - 1
    bins = bins[1:] - binwidth / 2
    # bins = np.hstack([bins, [val_max]])
    return bins[vals_bin_idx]


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
    if experiment == "transd":
        ea_label = "estim_accs"
    else:
        print(f"Invalid experiment '{experiment}'; aborting.")
        return

    accs = get_acc_names()
    datasets = get_all_dataset_names()
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
                Results.load(base_dir=base_dir, acc_name=acc, dataset=dataset, set_problem=False)
                # .split_by_shift(prevs=0.5)
                .model_selection(oracle=True, only_default=True, ea_label=ea_label)
                .filter_column_values("method", "isin", methods)
                .select_columns(["method", "dataset", "true_accs", "shifts"])
            )
            print(f"[{acc}] {dataset} loaded")

        res = Results.concat(res, axis=0)
        res = (
            res.add_column("shift_bins", get_binned_values(res.df, "shifts", n_bins=20))
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
    args = parser.parse_args()

    plots(args.experiment)
