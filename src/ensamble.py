import itertools as IT
import os
from argparse import ArgumentParser
from time import time

import numpy as np
import pandas as pd
import quapy as qp
from cap.utils.commons import contingency_table, get_shift
from scipy.special import softmax
from tqdm import tqdm

import env
from config import acc_from_ct, get_acc_names
from data import PretainInfo, load_info_paths
from results import RDF, Results, ResultsDataFrame
from util import get_plain_prev

qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0


def get_ensamble_path(acc: str, dataset: str, method: str):
    base_dir = os.path.join(env.root_dir, "ensamble")
    ensamble_dir = os.path.join(base_dir, env.DOMAIN, acc, dataset)
    os.makedirs(ensamble_dir, exist_ok=True)
    return os.path.join(ensamble_dir, f"{method}.parquet")


def transform_ranking_vals(rvs, temperature=0.01):
    return softmax(rvs / temperature, axis=-1)


def build_ensamble_result(p_infos: list[PretainInfo], dataset: str, acc: str, method: str) -> ResultsDataFrame:
    results_base_dir = os.path.join(env.root_dir, "main")
    res = Results.load(
        base_dir=results_base_dir, acc_name=acc, dataset=dataset, domain=env.DOMAIN, filter_methods=[method]
    )
    f_pinfos = [p for p in p_infos if p.d_info.name == dataset]
    # tqdm.write(f"{acc} - {method} - {dataset}: {len(f_pinfos)}")
    d_bundle = f_pinfos[0].load_dataset_bundle()
    h_dict = {p.h_info.name: p.load_pretrained_classifier(d_bundle) for p in f_pinfos}
    _cts = []
    tinit = time()
    for uid, Ui in tqdm(
        enumerate(d_bundle.test_prot()), desc=f"{acc}-{method}-{dataset}", total=d_bundle.test_prot.total()
    ):
        uid_edf = res.df.loc[res.df["uids"] == uid, :]
        ui_post = np.zeros((Ui.X.shape[0], Ui.n_classes), dtype=np.float32)
        h_list = uid_edf["classifier"].to_list()
        rv_array = uid_edf["ranking_vals"].to_numpy()
        rv_array = transform_ranking_vals(rv_array)
        for h_name, rv in zip(h_list, rv_array):
            _h = h_dict[h_name]
            _h_post = _h.predict_proba(Ui.X)
            ui_post += rv * _h_post
        _cts.append(contingency_table(Ui.y, np.argmax(ui_post, axis=-1), Ui.n_classes))
    t_test_ave = (time() - tinit) / d_bundle.test_prot.total()

    L_prev = get_plain_prev(d_bundle.L_prevalence)
    val_prev = get_plain_prev(d_bundle.V.prevalence())
    test_shift = get_shift(np.array([Ui.prevalence() for Ui in d_bundle.test_prot()]), d_bundle.L_prevalence).tolist()
    df_len = d_bundle.test_prot.total()
    return RDF.from_records(
        df_len,
        uids=np.arange(df_len).tolist(),
        shifts=test_shift,
        true_cts=_cts,
        ranking_vals=np.ones(df_len),
        classifier=["ensamble"] * df_len,
        classifier_class=["ensamble"] * df_len,
        default_c=[True] * df_len,
        ms_ignore=[False] * df_len,
        method=[f"{method}-ens"] * df_len,
        dataset=[dataset] * df_len,
        collection=[f_pinfos[0].d_info.collection] * df_len,
        n_classes=[f_pinfos[0].d_info.n_classes] * df_len,
        acc_name=[acc] * df_len,
        train_prev=[L_prev] * df_len,
        val_prev=[val_prev] * df_len,
        t_train=0,
        t_test_ave=t_test_ave,
    )


def main():
    info_paths = load_info_paths(domain=env.DOMAIN)
    p_infos = [PretainInfo.load(p, fast=True) for p in info_paths]
    datasets = np.unique([p.d_info.name for p in p_infos])
    # datasets = ["german", "mammographic", "semeion", "spambase", "tictactoe"]
    accs = get_acc_names()
    methods = ["TMS_LEAP"]
    for dataset, acc, method in IT.product(datasets, accs, methods):
        _path = get_ensamble_path(acc, dataset, method)
        if os.path.exists(_path):
            tqdm.write(f"{acc}-{method}-{dataset} already exists, skipping.")
            continue
        res = build_ensamble_result(p_infos, dataset, acc, method)
        res.save_result(_path)


def show():
    def ms_selection():
        return {
            "oracle": True,
            "default": [],
            "method": [("TMS_LEAP", None), ("IMS", None)],
        }

    datasets = ["german", "mammographic", "semeion", "spambase", "tictactoe"]
    accs = get_acc_names()
    methods = ["TMS_LEAP"]
    main_base_dir = os.path.join(env.root_dir, "main")
    ensamble_base_dir = os.path.join(env.root_dir, "ensamble")
    ress = []
    for dataset, acc, method in IT.product(datasets, accs, methods):
        _path = get_ensamble_path(acc, dataset, method)
        if not os.path.exists(_path):
            tqdm.write(f"{acc}-{method}-{dataset} does not exist, skipping.")
            continue

        try:
            main_res = (
                Results.load(base_dir=main_base_dir, acc_name=acc, dataset=dataset, domain=env.DOMAIN)
                .model_selection(selection=ms_selection(), rank_label="ranking_vals")
                .apply_to_column("true_cts", lambda ct: acc_from_ct(acc, ct), new_col="true_accs")
            )
        except ValueError as e:
            print(f"main {dataset}-{acc}-{method}: {e}")

        try:
            ensamble_res = Results.load(
                base_dir=ensamble_base_dir, acc_name=acc, dataset=dataset, domain=env.DOMAIN
            ).apply_to_column("true_cts", lambda ct: acc_from_ct(acc, ct), new_col="true_accs")
        except ValueError as e:
            print(f"ensamble {dataset}-{acc}-{method}: {e}")

        ress.append(Results.concat([main_res, ensamble_res]))

    pivot = pd.pivot_table(Results.concat(ress).df, index=["dataset"], columns=["method"], values=["true_accs"])
    print(pivot.to_string())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--show", action="store_true", help="Show results")
    pargs = parser.parse_args()

    if pargs.show:
        show()
    else:
        main()
