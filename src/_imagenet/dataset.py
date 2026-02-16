import json
import os
from collections import defaultdict

import nltk
from datasets import load_dataset
from nltk.corpus import wordnet as wn

LOC_sysnet_path = "src/_imagenet/LOC_synset_mapping.txt"
lbl_sysnet_path = "src/_imagenet/lbl_synset_mapping.json"

nltk.download("wordnet")


def get_lbl_synset_map():
    lbl_syn_map = None

    if os.path.exists(lbl_sysnet_path):
        with open(lbl_sysnet_path, "r") as f:
            lbl_syn_map = json.load(f)
        lbl_syn_map = {int(k): v for k, v in lbl_syn_map.items()}
        try:
            assert all([i in lbl_syn_map for i in range(1000)])
        except AssertionError:
            lbl_syn_map = None

    if lbl_syn_map is not None:
        return lbl_syn_map

    lbl_syn_map = {}

    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        streaming=True,
    )
    lbl2name = ds.features["label"].int2str

    name_syn_map = {}
    with open(LOC_sysnet_path, "r") as f:
        for line in f.readlines():
            synset, name = line.split(" ", maxsplit=1)
            name_syn_map[name.strip()] = synset

    for i in range(1000):
        lbl_syn_map[i] = name_syn_map[lbl2name(i)]

    with open(lbl_sysnet_path, "w") as f:
        json.dump(lbl_syn_map, f)

    return lbl_syn_map


def get_superclass(ls_map: dict, depth=3):
    super_map = defaultdict(list)
    for s in ls_map.values():
        offset = int(s[1:])
        current = wn.synset_from_pos_and_offset("n", offset)
        paths = current.hypernym_paths()
        if len(paths) > 1:
            print(s, [len(p) for p in paths])
        path = paths[0]
        ancestor = path[-depth]
        super_map[ancestor].append(s)

    _lens = [len(v) for _, v in super_map.items()]
    _max, _min = max(_lens) / 1000, min(_lens) / 1000
    print(len(super_map), _max, _min)


if __name__ == "__main__":
    ls_map = get_lbl_synset_map()
    get_superclass(ls_map, depth=4)
