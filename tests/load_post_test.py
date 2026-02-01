import time

import numpy as np

from data import PretainInfo, load_info_paths

d_name = "isolet"


def main():
    paths = load_info_paths(domain="classic")
    ps = []
    for path in paths:
        p = PretainInfo.load(path, fast=True)
        if p.d_info.name == d_name:
            ps.append(p)

    npzs = []
    for p in ps:
        npzs.append(np.load(p.posteriors_path))

    print(f"loaded {len(npzs)} posteriors")

    time.sleep(10)


if __name__ == "__main__":
    main()
