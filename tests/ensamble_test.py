import os

import quapy as qp

import env
from config import get_acc_names
from data import PretainInfo, load_info_paths
from results import Results

qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0


def load_results():
    base_dir = os.path.join(env.root_dir, "main")
    dataset = "poker_hand"
    acc_name = "vanilla_accuracy"
    domain = "classic"
    method = "TMS_LEAP"

    return Results.load(
        base_dir=base_dir, acc_name=acc_name, dataset=dataset, domain=domain, filter_methods=[method]
    ), method


def main():
    res, method = load_results()

    info_paths = load_info_paths(domain=env.DOMAIN)
    p_infos = [PretainInfo.load(p, fast=True) for p in info_paths]

    accs = get_acc_names()
    for acc_name in accs:
        edf = res.df.loc[(res.df["method"] == method) & (res.df["acc_name"] == acc_name), :]
        datasets = edf["dataset"].unique()
        for dataset in datasets:
            edf = edf.loc[edf["dataset"] == dataset, :]
            f_pinfos = [p for p in p_infos if p.d_info.name == dataset]
            print(f"n_classifiers: {len(f_pinfos)}")
            d_bundle = f_pinfos[0].load_dataset_bundle()
            h_dict = {p.h_info.name: p.load_pretrained_classifier(d_bundle) for p in f_pinfos}
            for uid, Ui in enumerate([next(d_bundle.test_prot())]):
                edf = edf.loc[edf["uids"] == uid, :]
                h_list = edf["classifier"].to_list()
                rv_array = edf["ranking_vals"].to_numpy()
                for t in edf[["classifier", "ranking_vals"]].itertuples():
                    print(t, type(t))


if __name__ == "__main__":
    main()
