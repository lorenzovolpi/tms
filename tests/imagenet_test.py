import numpy as np
from datasets import load_dataset


def main():
    k = 100
    dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
    freqs = np.zeros(1000)
    for label in dataset["label"]:
        freqs[label] += 1

    top_k_idx = np.argsort(freqs)[::-1][:k]
    top_k = freqs[top_k_idx]
    for i, v in zip(top_k_idx, top_k):
        print(f"{i}: {v}")


if __name__ == "__main__":
    main()
