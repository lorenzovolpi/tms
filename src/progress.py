import os
import time
from glob import glob

from tqdm import tqdm

import env
from config import get_acc_names, get_method_names
from data import load_info_paths

if __name__ == "__main__":
    n_pretrains = len(load_info_paths(domain=env.DOMAIN))
    n_methods = len(get_method_names())
    n_accs = len(get_acc_names())

    n_total = n_pretrains * n_methods * n_accs
    last = 0
    with tqdm(total=n_total) as pbar:
        while last < n_total:
            done = len(glob(os.path.join(env.root_dir, "main", env.DOMAIN, "**", "*.parquet"), recursive=True))
            delta = done - last
            last = done
            pbar.update(delta)
            time.sleep(1)
