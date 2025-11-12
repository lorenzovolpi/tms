import os
import pickle
from argparse import ArgumentParser
from typing import Literal

import pandas as pd
from pandatex import Format, Table

import env
from config import get_acc_names, get_all_dataset_names
from results import Results
from util import decorate_dataset

method_map = {
    "LR": "\\nomslr",
    "kNN": "\\nomsknn",
    "SVM": "\\nomssvm",
    "MLP": "\\nomsmlp",
    "SVM-t": "\\nomstsvm",
    "Naive-LR": "\\imslr",
    "Naive-kNN": "\\imsknn",
    "Naive-SVM": "\\imssvm",
    "Naive-MLP": "\\imsmlp",
    "Naive": "\\imsall",
    "O-LEAP(KDEy)": "\\tmsall",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
}


def gen_tables(experiment: Literal["transd", "hoptim"]):
    if experiment == "transd":
        ea_label = "estim_accs"
    else:
        print(f"Invalid experiment '{experiment}'; aborting.")
        return

    base_dir = os.path.join(env.root_dir, experiment)

    def add_to_table(tbl: Table, df: pd.DataFrame, dataset, methods):
        for method in methods:
            values = df.loc[df["method"] == method, "true_accs"].to_numpy()
            for v in values:
                tbl.add(dataset, method, v)
        return tbl

    tbls = []
    accs = get_acc_names()
    datasets = get_all_dataset_names()
    for acc in accs:
        name = f"{experiment}_{acc}"
        tbl = Table(name=name)
        tbl.format = Format(
            lower_is_better=False,
            mean_prec=3,
            show_std=True,
            remove_zero=True,
            with_rank_mean=False,
            with_mean=True,
            mean_macro=False,
            color=True,
            color_mode="local",
            simple_stat=True,
        )
        for dataset in datasets:
            res = (
                Results.load(base_dir=base_dir, acc_name=acc, dataset=dataset, set_problem=False)
                # .split_by_shift(prevs=0.5)
                .model_selection(oracle=False, only_default=True, ea_label=ea_label)
                .map_column_values("method", method_map)
                .map_column_values("dataset", dataset_map)
                .apply_to_column("dataset", decorate_dataset)
            )
            _methods = [method_map.get(m, m) for m in res.unique_column_values("method")]
            _dataset = decorate_dataset(dataset_map.get(dataset, dataset))
            tbl = add_to_table(tbl, res.df, _dataset, _methods)
            print(f"[{acc}] {dataset} done")

        tbls.append(tbl)
        print(f"table {name} genned")

    table_dir = os.path.join(env.root_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    pickle_path = os.path.join(table_dir, f"{experiment}.pickle")
    with open(pickle_path, "wb") as f:
        pickle.dump(tbls, f)

    # pdf_path = os.path.join(table_dir, f"{experiment}.pdf")
    # new_commands = [
    #     "\newcommand{\tmsall}{TMS-All}",
    #     "\newcommand{\imsall}{IMS-All}",
    #     "\\newcommand{\\imslr}{IMS-LR}",
    #     "\\newcommand{\\imsknn}{IMS-$k$NN}",
    #     "\\newcommand{\\imssvm}{IMS-SVM}",
    #     "\\newcommand{\\imsmlp}{IMS-MLP}",
    #     "\\newcommand{\\nomslr}{$\\emptyset$-LR}",
    #     "\\newcommand{\\nomsknn}{$\\emptyset$-$k$NN}",
    #     "\\newcommand{\\nomssvm}{$\\emptyset$-SVM}",
    #     "\\newcommand{\\nomstsvm}{$\\emptyset$-TSVM}",
    #     "\\newcommand{\\nomsmlp}{$\\emptyset$-MLP}",
    # ]
    # column_alignment = [5, 5, 1], "c"
    # additional_headers = [("$\\emptyset$", 5), ("IMS", 5), ("TMS", 1)]
    # Table.LatexPDF(
    #     pdf_path,
    #     tables=tbls,
    #     landscape=False,
    #     new_commands=new_commands,
    #     column_alignment=column_alignment,
    #     additional_headers=additional_headers,
    # )


def gen_pdf(experiment: Literal["transd", "hoptim"]):
    table_dir = os.path.join(env.root_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    pickle_path = os.path.join(table_dir, f"{experiment}.pickle")
    assert os.path.exists(pickle_path), "pickle file does not exist"

    with open(pickle_path, "rb") as f:
        tbls = pickle.load(f)

    pdf_path = os.path.join(table_dir, f"{experiment}.pdf")
    new_commands = [
        "\\newcommand{\\tmsall}{TMS-All}",
        "\\newcommand{\\imsall}{IMS-All}",
        "\\newcommand{\\imslr}{IMS-LR}",
        "\\newcommand{\\imsknn}{IMS-$k$NN}",
        "\\newcommand{\\imssvm}{IMS-SVM}",
        "\\newcommand{\\imsmlp}{IMS-MLP}",
        "\\newcommand{\\nomslr}{$\\emptyset$-LR}",
        "\\newcommand{\\nomsknn}{$\\emptyset$-$k$NN}",
        "\\newcommand{\\nomssvm}{$\\emptyset$-SVM}",
        "\\newcommand{\\nomstsvm}{$\\emptyset$-TSVM}",
        "\\newcommand{\\nomsmlp}{$\\emptyset$-MLP}",
    ]
    column_alignment = [5, 5, 1], "c"
    additional_headers = [("$\\emptyset$", 5), ("IMS", 5), ("TMS", 1)]
    Table.LatexPDF(
        pdf_path,
        tables=tbls,
        landscape=False,
        new_commands=new_commands,
        column_alignment=column_alignment,
        additional_headers=additional_headers,
    )


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
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    args = parser.parse_args()

    if args.tables:
        gen_tables(args.experiment)
    if args.pdf:
        gen_pdf(args.experiment)
