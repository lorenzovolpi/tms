from collections import defaultdict

from config import gen_classifiers, gen_datasets

map = defaultdict(lambda: {})

if __name__ == "__main__":
    for d, (L, V, U) in gen_datasets():
        n_classes = L.n_classes
        for clsf in gen_classifiers(n_classes):
            assert clsf.file_name not in map[d], f"{clsf.name} already mapped"
            map[d][clsf.file_name] = clsf.name

    print("OK")
    for k in map:
        print(f"{k}: {len(map[k])}")
        for _, c in map[k].items():
            print(f"\t{c}")
