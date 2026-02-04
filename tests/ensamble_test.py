import os
from collections import defaultdict

import numpy as np
import pandas as pd
import quapy as qp
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import env
from config import get_acc_names, get_dataset_names
from data import PretainInfo, load_info_paths
from results import Results

qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0


def load_results(dataset):
    base_dir = os.path.join(env.root_dir, "main")
    acc_name = "vanilla_accuracy"
    domain = "classic"
    method = "TMS_LEAP"

    return Results.load(
        base_dir=base_dir, acc_name=acc_name, dataset=dataset, domain=domain, filter_methods=[method]
    ), method


def transform_ranking_vals(rvs, temperature=0.05):
    return softmax(rvs / temperature, axis=-1)


def main():
    info_paths = load_info_paths(domain=env.DOMAIN)
    p_infos = [PretainInfo.load(p, fast=True) for p in info_paths]
    datasets = np.unique([p.d_info.name for p in p_infos])

    table = defaultdict(list)
    for dataset in datasets:
        res, method = load_results(dataset)
        edf = res.df
        f_pinfos = [p for p in p_infos if p.d_info.name == dataset]
        tqdm.write(f"{dataset}: {len(f_pinfos)}")
        d_bundle = f_pinfos[0].load_dataset_bundle()
        h_dict = {p.h_info.name: p.load_pretrained_classifier(d_bundle) for p in f_pinfos}
        for uid, Ui in tqdm(enumerate(d_bundle.test_prot()), desc=f"{dataset}", total=d_bundle.test_prot.total()):
            _oracle_acc = 0.0
            _method_acc = 0.0
            uid_edf = edf.loc[edf["uids"] == uid, :]
            ui_post = None
            h_list = uid_edf["classifier"].to_list()
            rv_array = uid_edf["ranking_vals"].to_numpy()
            if rv_array.shape[0] != len(h_list):
                continue
            _method_id = int(np.argmax(rv_array))
            rv_array = transform_ranking_vals(rv_array)
            for cl_id, (h_name, rv) in enumerate(zip(h_list, rv_array)):
                _h = h_dict[h_name]
                _h_post = _h.predict_proba(Ui.X)
                _h_acc = accuracy_score(Ui.y, np.argmax(_h_post, axis=-1))
                _oracle_acc = _h_acc if _h_acc > _oracle_acc else _oracle_acc
                if cl_id == _method_id:
                    _method_acc = _h_acc
                if ui_post is None:
                    ui_post = rv * _h_post
                else:
                    ui_post += rv * _h_post
            _ensable_acc = accuracy_score(Ui.y, np.argmax(ui_post, axis=-1))
            table["method"].append("ensable")
            table["acc"].append(_ensable_acc)
            table["method"].append("oracle")
            table["acc"].append(_oracle_acc)
            table["method"].append(method)
            table["acc"].append(_method_acc)
            table["dataset"].extend([dataset] * 3)
            table["uid"].append([uid] * 3)
    pivot = pd.pivot_table(index="dataset", columns="method", values="acc", data=pd.DataFrame.from_dict(table))
    print(pivot.to_string())


if __name__ == "__main__":
    main()
