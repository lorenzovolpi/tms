import itertools as IT
import os
from dataclasses import asdict, dataclass
from itertools import batched
from typing import Any

import numpy as np
import quapy as qp
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from data import ClassifierInfo, DatasetInfo, PretainInfo
from pretrain.dataset import save_sentiment

VERBOSE = True
qp.environ["_R_SEED"] = 0


hf_dataset_map = {}
hf_model_map = {}


def _fdataset(name: str, n: int):
    proper_name = name.replace("/", "__")
    hf_dataset_map[proper_name] = name
    return DatasetInfo(proper_name, "sentiment", n)


def _fmodel(name: str):
    proprer_name = name.replace("/", "__")
    hf_model_map[proprer_name] = name
    return proprer_name


def sout(*args):
    if VERBOSE:
        print(*args)


def gen_datasets():
    yield _fdataset("stanfordnlp/imdb", 2)


def gen_model_args():
    model_params = [
        (
            "google-bert/bert-base-uncased",
            dict(
                train_backbone=[True, False],
            ),
        ),
    ]

    for model_name, param_dict in model_params:
        _keys = list(param_dict.keys())
        _param_combos = IT.product(*list(param_dict.values()))
        for _combo in _param_combos:
            _params = dict(zip(_keys, _combo))
            yield _fmodel(model_name, SentimentArgs(**_params))


def gen_config():
    for d_info, (model_name, args) in IT.product(gen_datasets(), gen_model_args()):
        yield d_info, model_name, args


def get_val_split(dataset):
    _default = 0.5
    _val_splits = {
        "stanfordnlp/imdb": 0.8,
    }
    return _val_splits.get(hf_dataset_map.get(dataset, dataset), _default)


def get_label_tag(dataset):
    _default = "label"
    _label_tags = {
        "stanfordnlp/imdb": "label",
    }
    return _label_tags.get(hf_dataset_map.get(dataset, dataset), _default)


@dataclass
class SentimentArgs:
    max_length: int = 512
    nepochs: int = 3
    train_batchsize: int = 64
    embed_batchsize: int = 512
    lr: float = 5e-4
    train_backbone: bool = False
    device: str = "cuda"
    load_bf16: bool = False

    @property
    def params(self) -> dict[str, Any]:
        # NOTE: asdict makes a deepcopy!
        return asdict(self)


def get_tr_outdir(p_info: PretainInfo):
    outdir = os.path.join("output", "tms", "models", p_info.h_info.full_name, p_info.d_info.name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def get_embed_outdir(args):
    model_name = args.model_name.split("/")[-1]
    dataset_name = args.dataset_name.split("/")[-1]
    outdir = os.path.join("output", "tms", "pretrain", "sentiment", dataset_name, model_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def get_dataset(d_info: DatasetInfo):
    """
    Load dataset and create validation split if does not exist.
    Also check that the number of classes matches the expected number.
    """
    dataset = load_dataset(hf_dataset_map.get(d_info.name, d_info.name))

    if "validation" not in dataset:
        val_split = get_val_split(d_info.name)
        sout("splitting training set into train/validation...")
        _tmp_dataset = dataset["train"].train_test_split(test_size=val_split, seed=qp.environ["_R_SEED"])
        dataset["train"] = _tmp_dataset["train"]
        dataset["validation"] = _tmp_dataset["test"]

    n_inferred_classes = dataset["train"].shape[-1]
    if d_info.n_classes != n_inferred_classes:
        sout(
            f"number of inferred target classes ({n_inferred_classes}) != number of given target classes ({d_info.n_classes})"
        )

    return dataset


def get_classifier(model_name, d_info: DatasetInfo, args: SentimentArgs):
    torch_dtype = torch.bfloat16 if args.load_bf16 else torch.float32
    model_name = hf_model_map.get(model_name, model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=d_info.n_classes, torch_dtype=torch_dtype
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def prepare_model(args: SentimentArgs, model):
    # freeze base model -> train only fresh init classification head
    if not args.train_backbone:
        sout("- freezing base model weights")
        for _, layer_weights in model.base_model.named_parameters():
            layer_weights.requires_grad = False

        # check trainable layers
        trainable_layers = []
        for layer_name, layer_weights in model.named_parameters():
            if layer_weights.requires_grad:
                trainable_layers.append(layer_name)
        sout(f"- trainable layers: {trainable_layers}")

    return model


def tokenize_dataset(args, tokenizer, dataset):
    def _tokenize_helper(sample):
        return tokenizer(
            sample["text"], truncation=True, max_length=args.max_length, padding="max_length", return_tensors="pt"
        )

    # tokenize dataset
    dataset = dataset.map(_tokenize_helper, batched=True, num_proc=8)
    return dataset


def compute_clf_metrics(preds):
    _preds = preds.predictions.argmax(axis=1)
    _labels = preds.label_ids
    acc = accuracy_score(y_true=_labels, y_pred=_preds)
    recall = recall_score(y_true=_labels, y_pred=_preds)
    precision = precision_score(y_true=_labels, y_pred=_preds)
    f1 = f1_score(y_true=_labels, y_pred=_preds, average="micro")
    return {"acc": acc, "recall": recall, "precision": precision, "f1": f1}


def train_model(args: SentimentArgs, p_info: PretainInfo, model, dataset):
    training_outdir = get_tr_outdir(p_info)

    trainer_args = TrainingArguments(
        output_dir=training_outdir,
        do_train=True,
        learning_rate=args.lr,
        num_train_epochs=args.nepochs,
        per_device_train_batch_size=args.train_batchsize,
        per_device_eval_batch_size=args.train_batchsize * 4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=100,
        bf16=True,
        metric_for_best_model="acc",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        eval_on_start=True,
        load_best_model_at_end=True,
    )

    # train model
    sout(f"- storing model in {trainer_args.output_dir}")
    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=trainer_args,
        compute_metrics=compute_clf_metrics,
        callbacks=[],  # early stopping callback goes here, if needed
    )
    last_ckpt = get_last_checkpoint(training_outdir)
    if last_ckpt is None:
        sout("\nTraining...")
    else:
        sout("\nLoading last checkpoint...")
    trainer.train(resume_from_checkpoint=last_ckpt)

    return trainer, trainer_args


def get_cls_bertlike(x):
    """
    Get representation associated with the "CLS" token. In BERT-like models
    this is usually assigned with the first token of the input sequence (idx=0).
    """
    cls_emebds = x[:, 0, :]
    return cls_emebds


def embed(model, tokenizer, data, selection_strategy, args):
    split_logits = []
    split_hidden_states = []
    split_y = []
    for batch in batched(tqdm(data), n=args.embed_batchsize):
        texts, labels = zip(*((d["text"], d["label"]) for d in batch))
        print(type(labels))
        with torch.no_grad():
            model_inputs = tokenizer(
                texts, truncation=True, max_length=args.max_length, padding="max_length", return_tensors="pt"
            )  # pad each batch to max_length
            output = model(**model_inputs.to(args.device), output_hidden_states=True)
        logits = output.logits
        hidden_states = output.hidden_states
        last_hidden_states = hidden_states[-1]

        split_y.append(torch.tensor(labels))
        split_hidden_states.append(selection_strategy(last_hidden_states.cpu().detach()))
        split_logits.append(logits.cpu().detach())

    split_y = torch.vstack(split_y).numpy()
    split_logits = torch.vstack(split_logits).numpy()
    split_hidden_states = torch.vstack(split_hidden_states).numpy()

    return split_y, split_logits, split_hidden_states


def save_dataset(data: dict):
    pass


def pretrain(d_info: DatasetInfo, model_name: str, args: SentimentArgs):
    h_info = ClassifierInfo(class_name=model_name, params=args.params)
    p_info = PretainInfo(domain="sentiment", d_info=d_info, h_info=h_info)
    if p_info.exists:
        return

    sout(f"- model: {args.model_name}")
    sout(f"- dataset: {args.dataset_name}")

    dataset = get_dataset(d_info)

    model, tokenizer = get_classifier(model_name, d_info, args)
    model = prepare_model(args, model)

    dataset = tokenize_dataset(args, tokenizer, dataset)

    train_model(args, model, dataset)

    # Get embedddings and logits
    embeds_outdir = get_embed_outdir(args)
    sout("\nEmbedding...")
    sout(f"- storing embeddings in {embeds_outdir}")
    splits = ["validation", "test"]
    embedddings = {}
    for split in splits:
        split_data = dataset[split]
        split_y, split_logits, split_last_hiddens = embed(
            model, tokenizer, data=split_data, selection_strategy=get_cls_bertlike, args=args
        )
        embedddings[split] = (split_last_hiddens, split_y)

    label_tag = get_label_tag(d_info.name)
    train_labels = np.array(dataset["train"][label_tag])
    classes = np.unique(train_labels)
    train_prev = np.sum(classes.reshape(-1, 1) == train_labels, axis=-1) / train_labels.shape[0]

    save_sentiment(d_info.name, h_info.full_name, classes, train_prev, embedddings)
    # torch.save(split_logits, os.path.join(embeds_outdir, f"logits.{split}.pt"))
    # torch.save(split_last_hiddens, os.path.join(embeds_outdir, f"hidden_states.{split}.pt"))


if __name__ == "__main__":
    if (
        "CUDA_VISIBLE_DEVICES" not in os.environ
        or "TOKENIZERS_PARALLELISM" not in os.environ
        or "HF_HUB_CACHE" not in os.environ
    ):
        raise ValueError("Missing env variables")

    for d_info, model_name, args in gen_config():
        pretrain(d_info, model_name, args)
