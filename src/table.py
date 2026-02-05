import os
import pickle
from argparse import ArgumentParser
from typing import Literal

import pandas as pd
from pandatex import Format, Table

import env
import main
from config import acc_from_ct, get_acc_names, get_existing_dataset_names
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
    "TMS_LEAP": "\\leapall",
    "TMS_RQBS": "\\rqbsall",
    "TMS_PrediQuant": "\\pqall",
    "TMS_DoC": "\\docall",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
}


def ms_selection():
    return {
        "oracle": True,
        "default": [],
        "method": [
            # ("IMS", "LR"),
            # ("IMS", "kNN"),
            # ("IMS", "SVM"),
            # ("IMS", "MLP"),
            ("IMS", None),
            ("TMS_LEAP", None),
            ("TMS_RQBS", None),
            # ("TMS_PrediQuant", None),
            # ("TMS_DoC", None),
        ],
    }


def gen_tables():
    experiment = main.EXPERIMENT
    domain = "classic"
    rank_label = "ranking_vals"

    main_base_dir = os.path.join(env.root_dir, "main")
    ensamble_base_dir = os.path.join(env.root_dir, "ensamble")

    def add_to_table(tbl: Table, df: pd.DataFrame, dataset, methods):
        for method in methods:
            values = df.loc[df["method"] == method, "true_accs"].to_numpy()
            for v in values:
                tbl.add(dataset, method, v)
        return tbl

    tbls = []
    accs = get_acc_names()
    datasets = get_existing_dataset_names("main", domain)
    for acc in accs:
        name = f"{experiment}_{domain}_{acc}"
        tbl = Table(name=name, oracles=["oracle"])
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
            main_res = (
                Results.load(base_dir=main_base_dir, acc_name=acc, dataset=dataset, domain=domain)
                # .split_by_shift(prevs=0.5)
                .model_selection(selection=ms_selection(), rank_label=rank_label)
                .map_column_values("method", method_map)
                .map_column_values("dataset", dataset_map)
                .apply_to_column("dataset", decorate_dataset)
                .apply_to_column("true_cts", lambda ct: acc_from_ct(acc, ct), new_col="true_accs")
            )
            ensamble_res = (
                Results.load(base_dir=ensamble_base_dir, acc_name=acc, dataset=dataset, domain=domain)
                .map_column_values("method", method_map)
                .map_column_values("dataset", dataset_map)
                .apply_to_column("dataset", decorate_dataset)
                .apply_to_column("true_cts", lambda ct: acc_from_ct(acc, ct), new_col="true_accs")
            )
            res = Results.concat([main_res, ensamble_res])
            _methods = [method_map.get(m, m) for m in res.unique_column_values("method")]
            _dataset = decorate_dataset(dataset_map.get(dataset, dataset))
            tbl = add_to_table(tbl, res.df, _dataset, _methods)
            print(f"[{acc}] {dataset} done")

        tbls.append(tbl)
        print(f"table {name} genned")

    table_dir = os.path.join(env.root_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    pickle_path = os.path.join(table_dir, f"{experiment}_{domain}.pickle")
    with open(pickle_path, "wb") as f:
        pickle.dump(tbls, f)


def gen_pdf():
    experiment = main.EXPERIMENT
    domain = "classic"
    table_dir = os.path.join(env.root_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    pickle_path = os.path.join(table_dir, f"{experiment}_{domain}.pickle")
    assert os.path.exists(pickle_path), "pickle file does not exist"

    with open(pickle_path, "rb") as f:
        tbls = pickle.load(f)

    pdf_path = os.path.join(table_dir, f"{experiment}_{domain}.pdf")
    new_commands = [
        "\\newcommand{\\leapall}{LEAP-All}",
        "\\newcommand{\\rqbsall}{RQBS-All}",
        "\\newcommand{\\pqall}{PQ-All}",
        "\\newcommand{\\docall}{DoC-All}",
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
    column_alignment = [1, 1, 2, 1], "c"
    additional_headers = [("oracle", 1), ("IMS", 1), ("TMS", 2), ("Ens", 1)]
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
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    args = parser.parse_args()

    if args.tables:
        gen_tables()
    if args.pdf:
        gen_pdf()
