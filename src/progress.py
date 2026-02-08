import os
import time
from collections import defaultdict
from glob import glob

from tqdm import tqdm

import env
from config import get_acc_names, get_method_names
from data import PretainInfo, load_info_paths

DOMAIN = "text"

if __name__ == "__main__":
    _info_paths = load_info_paths(domain=DOMAIN)
    n_pretrains = len(_info_paths)
    n_methods = len(get_method_names())
    n_accs = len(get_acc_names())

    n_pretrains = defaultdict(lambda: 0)
    for p in _info_paths:
        dataset = PretainInfo.load(p, fast=True).d_info.name
        n_pretrains[dataset] += n_methods * n_accs

    p_bars = {d: tqdm(total=n, desc=str(d), position=i) for i, (d, n) in enumerate(n_pretrains.items())}

    n_total = sum(list(n_pretrains.values())) * n_methods * n_accs
    _compls = {d: 0 for d, _ in n_pretrains.items()}
    while sum(list(_compls.values())) < n_total:
        for dataset, p_bar in p_bars.items():
            _done = len(
                glob(os.path.join(env.root_dir, "main", DOMAIN, "*", dataset, "**", "*.parquet"), recursive=True)
            )
            _delta = _done - _compls[dataset]
            _compls[dataset] = _done
            p_bar.update(_delta)
        time.sleep(1)
