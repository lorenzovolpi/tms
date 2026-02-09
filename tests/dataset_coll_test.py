from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS

from data import PretainInfo, load_info_paths


def main():
    info_paths = load_info_paths(domain="classic")
    for path in info_paths:
        p = PretainInfo.load(path, fast=True)
        if p.d_info.collection == "uci_binary" and p.d_info.name not in UCI_BINARY_DATASETS:
            print(f"d_name={p.d_info.name}\nh_name={p.h_info.name}\ncoll={p.d_info.collection}\n\n")
        if p.d_info.collection == "uci_multiclass" and p.d_info.name not in UCI_MULTICLASS_DATASETS:
            print(f"d_name={p.d_info.name}\nh_name={p.h_info.name}\ncoll={p.d_info.collection}\n\n")


if __name__ == "__main__":
    main()
