import os
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from traceback import print_exception
from typing import Any, Iterator

import numpy as np
import quapy as qp
import torch
from datasets import concatenate_datasets, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint

from data import ClassifierInfo, DatasetInfo, PretainInfo
from env import PROJECT
from pretrain.dataset import save_dataset
from util import get_logger

EXPERIMENT = "pretrain"
DOMAIN = "text"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VERBOSE = True
qp.environ["_R_SEED"] = 0

log = get_logger(id=f"{PROJECT}.{EXPERIMENT}.{DOMAIN}")

hf_dataset_map = {}
hf_model_map = {}


class LoggingCallback(TrainerCallback):
    filter_fields = set(
        ["loss", "learning_rate", "eval_loss", "eval_acc", "eval_runtime", "train_runtime", "train_loss"]
    )

    def __init__(self, p: PretainInfo) -> None:
        self.p = p

    def get_logs_str(self, logs: dict):
        return [f"'{k}': {v:.4f}" if not isinstance(v, int) else f"'{k}': {v}" for k, v in logs.items()]

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return

        logs_with_step = {
            "step": state.global_step,
            "epoch": state.epoch,
            **{k: v for k, v in logs.items() if k in self.filter_fields},
        }
        logs_str = "{" + ", ".join(self.get_logs_str(logs_with_step)) + "}"
        log.info(f"[{self.p.h_info.name}@{self.p.d_info.name}] training log: {logs_str}")


def _fdataset(name: str, n: int) -> DatasetInfo:
    proper_name = name.replace("/", "__")
    hf_dataset_map[proper_name] = name
    return DatasetInfo(proper_name, DOMAIN, n)


def _fmodel(name: str, default: bool = True, **kwargs) -> ClassifierInfo:
    proprer_name = name.replace("/", "__")
    hf_model_map[proprer_name] = name
    return proprer_name


def sout(*args):
    if VERBOSE:
        print(*args)


def gen_datasets() -> Iterator[DatasetInfo]:
    yield _fdataset("stanfordnlp/imdb", 2)
    yield _fdataset("fancyzhx/yelp_polarity", 2)
    yield _fdataset("stanfordnlp/sst2", 2)
    yield _fdataset("fancyzhx/ag_news", 4)
    yield _fdataset("fancyzhx/dbpedia_14", 14)
    yield _fdataset("community-datasets/yahoo_answers_topics", 10)


def gen_model_args(d_info: DatasetInfo) -> Iterator[ClassifierInfo]:
    def ovverride_params(model_name: str, args: SentimentArgs, d_info: DatasetInfo):
        _overrides = {
            ("*", "stanfordnlp/imdb"): dict(
                nepochs=3,
                lr=2e-5,
                warmup_steps=200,
                train_hl=True,
            ),
            ("*", "stanfordnlp/sst2"): dict(
                nepochs=3,
                lr=2e-5,
                warmup_steps=300,
                train_hl=True,
            ),
            ("*", "fancyzhx/yelp_polarity"): dict(
                nepochs=3,
                lr=2e-5,
                warmup_steps=500,
                train_hl=True,
            ),
            ("*", "fancyzhx/ag_news"): dict(
                nepochs=3,
                lr=2e-5,
                warmup_steps=500,
                max_length=256,
                train_hl=True,
            ),
            ("*", "fancyzhx/dbpedia_14"): dict(
                nepochs=3,
                lr=2e-5,
                warmup_steps=1000,
                max_length=256,
                train_hl=True,
            ),
            ("*", "community-datasets/yahoo_answers_topics"): dict(
                nepochs=3,
                lr=2e-5,
                warmup_steps=1000,
                max_length=256,
                train_hl=True,
            ),
        }

        d_name = hf_dataset_map.get(d_info.name, d_info.name)
        or_params = _overrides.get(("*", d_name), {}) | _overrides.get((model_name, d_name), {})
        return args.update(or_params)

    def mp(name: str, default=True, args=None):
        args = args if args else SentimentArgs()
        return dict(name=name, default=default, args=args)

    model_params = [
        mp("google-bert/bert-base-uncased"),
        mp("FacebookAI/roberta-base"),
        mp("distilbert/distilbert-base-uncased"),
        mp("microsoft/deberta-v3-base", args=SentimentArgs(train_bsize=32, embed_bsize=256)),
        mp("google/electra-base-discriminator"),
    ]
    for mp in model_params:
        proper_name = _fmodel(mp["name"])
        args = ovverride_params(mp["name"], mp["args"], d_info)
        yield ClassifierInfo(class_name=proper_name, params=args.params, default=mp["default"])


def gen_config():
    for d_info in gen_datasets():
        for h_info in gen_model_args(d_info):
            yield d_info, h_info


def get_val_split(dataset):
    _default = 0.5
    _val_splits = {
        "stanfordnlp/imdb": 0.6,
        "fancyzhx/yelp_polarity": 0.9,
        "stanfordnlp/sst2": 0.58,
        "fancyzhx/ag_news": 0.5,
        "fancyzhx/dbpedia_14": 0.6,
        "community-datasets/yahoo_answers_topics": 0.82,
    }
    return _val_splits.get(hf_dataset_map.get(dataset, dataset), _default)


@dataclass
class SentimentArgs:
    max_length: int = 512
    nepochs: int = 2
    train_bsize: int = 64
    embed_bsize: int = 512
    lr: float = 2e-5
    train_hl: bool = False
    load_bf16: bool = False
    warmup_steps: int = 200
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    @property
    def params(self) -> dict[str, Any]:
        # NOTE: asdict makes a deepcopy!
        return asdict(self)

    def update(self, params: dict):
        return SentimentArgs(**(self.params | params))


def get_tr_outdir(p_info: PretainInfo):
    outdir = os.path.join("output", "tms", "models", p_info.h_info.full_name, p_info.d_info.name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def get_embed_outdir(args):
    model_name = args.model_name.split("/")[-1]
    dataset_name = args.dataset_name.split("/")[-1]
    outdir = os.path.join("output", "tms", "pretrain", DOMAIN, dataset_name, model_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def fix_dataset_fields(d_info, dataset):
    to_remove = {
        "stanfordnlp/sst2": ["sentence"],
        "fancyzhx/dbpedia_14": ["title", "content"],
        "community-datasets/yahoo_answers_topics": ["question_title", "question_content", "best_answer", "topic"],
    }
    d_name = hf_dataset_map.get(d_info.name, d_info.name)

    def combine_fields(split):
        if d_name == "stanfordnlp/sst2":
            split["text"] = f"{split['sentence']}"
        elif d_name == "fancyzhx/dbpedia_14":
            split["text"] = f"{split['title']}\n{split['content']}"
        elif d_name == "community-datasets/yahoo_answers_topics":
            split["text"] = f"{split['question_title']}\n{split['question_content']}\n{split['best_answer']}"
            split["label"] = split["topic"]

        return split

    dataset = dataset.map(combine_fields)
    dataset = dataset.remove_columns(to_remove.get(d_name, []))

    return dataset


def get_dataset(d_info: DatasetInfo):
    """
    Load dataset and create validation split if does not exist.
    Also check that the number of classes matches the expected number.
    """
    dataset = load_dataset(hf_dataset_map.get(d_info.name, d_info.name))
    dataset = fix_dataset_fields(d_info, dataset)

    if "validation" in dataset:
        trainval = concatenate_datasets([dataset["train"], dataset["validation"]])
        dataset["train"] = trainval

    # for sst2: discard unlabelled test set and split training set using 0.3 ratio for test
    if hf_dataset_map.get(d_info.name, d_info.name) == "stanfordnlp/sst2":
        _tmp_dataset = dataset["train"].train_test_split(test_size=0.3, seed=qp.environ["_R_SEED"])
        dataset["train"] = _tmp_dataset["train"]
        dataset["test"] = _tmp_dataset["test"]

    val_split = get_val_split(d_info.name)
    sout("splitting training set into train/validation...")
    _tmp_dataset = dataset["train"].train_test_split(test_size=val_split, seed=qp.environ["_R_SEED"])
    dataset["train"] = _tmp_dataset["train"]
    dataset["validation"] = _tmp_dataset["test"]

    fe_size = min(int(2e4), dataset["validation"].num_rows)
    if fe_size < dataset["validation"].num_rows:
        _tmp_fe_set = dataset["validation"].train_test_split(test_size=fe_size, seed=qp.environ["_R_SEED"])
        dataset["fast_eval"] = _tmp_fe_set["test"]
    else:
        dataset["fast_eval"] = dataset["validation"]

    n_inferred_classes = np.unique(dataset["train"]["label"]).shape[0]
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
    ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def prepare_model(args: SentimentArgs, model):
    # freeze base model -> train only fresh init classification head
    if not args.train_hl:
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


def tokenize_dataset(args, tokenizer, dataset, dataset_name):
    text_tag = "text"

    def _tokenize_helper(sample):
        return tokenizer(
            sample[text_tag], truncation=True, max_length=args.max_length, padding="max_length", return_tensors="pt"
        )

    # tokenize dataset
    dataset = dataset.map(_tokenize_helper, batched=True, num_proc=8)
    return dataset


def compute_clf_metrics(preds):
    _preds = preds.predictions.argmax(axis=1)
    _labels = preds.label_ids
    acc = accuracy_score(y_true=_labels, y_pred=_preds)
    f1 = f1_score(y_true=_labels, y_pred=_preds, average="micro")
    return {"acc": acc, "f1": f1}


def train_model(args: SentimentArgs, p_info: PretainInfo, model, dataset):
    training_outdir = get_tr_outdir(p_info)

    trainer_args = TrainingArguments(
        output_dir=training_outdir,
        do_train=True,
        learning_rate=args.lr,
        num_train_epochs=args.nepochs,
        per_device_train_batch_size=args.train_bsize,
        per_device_eval_batch_size=args.train_bsize * 4,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        # eval_strategy="steps",
        # eval_steps=100,
        eval_strategy="epoch",
        logging_steps=50,
        bf16=True,
        metric_for_best_model="acc",
        greater_is_better=True,
        save_strategy="epoch",
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
        eval_dataset=dataset["fast_eval"],
        args=trainer_args,
        compute_metrics=compute_clf_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=1e-4),
            LoggingCallback(p_info),
        ],  # early stopping callback goes here, if needed
    )
    # last_ckpt = get_last_checkpoint(training_outdir)
    # if last_ckpt is None:
    #     sout("\nTraining...")
    # else:
    #     sout("\nLoading last checkpoint...")
    # trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.train()

    return trainer, trainer_args


def get_cls_bertlike(x):
    """
    Get representation associated with the "CLS" token. In BERT-like models
    this is usually assigned with the first token of the input sequence (idx=0).
    """
    cls_emebds = x[:, 0, :]
    return cls_emebds


def embed(model, data, selection_strategy, args: SentimentArgs):
    # text_tag = "text"
    # split_logits = []
    split_posteriors = []
    split_hidden_states = []
    split_y = []
    dataloader = DataLoader(
        data.remove_columns(["text"]),
        batch_size=args.embed_bsize,
        shuffle=False,
        collate_fn=DefaultDataCollator(),
    )
    # for batch in batched(tqdm(data), n=args.embed_bsize):
    for batch in tqdm(dataloader, desc="Embedding"):
        # texts, labels = zip(*((d[text_tag], d["label"]) for d in batch))
        # labels = [d["label"] for d in batch]
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].cpu()
        with torch.no_grad():
            # model_inputs = tokenizer(
            #     texts, truncation=True, max_length=args.max_length, padding="max_length", return_tensors="pt"
            # )  # pad each batch to max_length
            # output = model(**model_inputs.to(DEVICE), output_hidden_states=True)
            output = model(**inputs, output_hidden_states=True)
        logits = output.logits
        posteriors = torch.softmax(logits, dim=-1)
        hidden_states = output.hidden_states
        last_hidden_states = hidden_states[-1]

        split_y.append(torch.tensor(labels))
        split_hidden_states.append(selection_strategy(last_hidden_states).cpu().detach())
        # split_logits.append(logits.cpu().detach())
        split_posteriors.append(posteriors.cpu().detach())

    split_y = torch.cat(split_y, dim=0).numpy()
    # split_logits = torch.vstack(split_logits).numpy()
    split_posteriors = torch.vstack(split_posteriors).numpy()
    split_hidden_states = torch.vstack(split_hidden_states).numpy()

    return split_y, split_posteriors, split_hidden_states


def pretrain(d_info: DatasetInfo, h_info: ClassifierInfo, parser_args):
    p_info = PretainInfo(domain=DOMAIN, d_info=d_info, h_info=h_info)
    if p_info.exists and not parser_args.retrain:
        log.info(f"[{h_info.name}@{d_info.name}] already exists, skipping.")
        return

    args = SentimentArgs(**h_info.params)
    model_name = h_info.class_name

    log.info(f"[{h_info.name}@{d_info.name}] started pretrain")
    sout(f"- model: {h_info.name}")
    sout(f"- dataset: {d_info.name}")

    dataset = get_dataset(d_info)
    log.info(f"[{h_info.name}@{d_info.name}] dataset loaded")

    model, tokenizer = get_classifier(model_name, d_info, args)
    model = prepare_model(args, model)
    log.info(f"[{h_info.name}@{d_info.name}] model loaded")

    dataset = tokenize_dataset(args, tokenizer, dataset, d_info.name)
    log.info(f"[{h_info.name}@{d_info.name}] dataset tokinezed")

    train_model(args, p_info, model, dataset)
    log.info(f"[{h_info.name}@{d_info.name}] model trained")

    if parser_args.dry_run:
        return

    # Get embedddings and logits
    splits = ["validation", "test"]
    embedddings = {}
    posteriors = {}
    for split in splits:
        split_data = dataset[split]
        split_y, split_posteriors, split_last_hiddens = embed(
            model, data=split_data, selection_strategy=get_cls_bertlike, args=args
        )
        embedddings[split] = (split_last_hiddens, split_y)
        posteriors[split] = split_posteriors

    label_tag = "label"
    train_labels = np.array(dataset["train"][label_tag])
    classes = np.unique(train_labels)
    train_prev = np.sum(classes.reshape(-1, 1) == train_labels, axis=-1) / train_labels.shape[0]

    save_dataset(DOMAIN, d_info.name, h_info.full_name, classes, train_prev, embedddings)
    log.info(f"[{h_info.name}@{d_info.name}] embeddings saved")
    p_info.dump(V_posteriors=posteriors["validation"], U_posteriors=posteriors["test"])
    log.info(f"[{h_info.name}@{d_info.name}] posteriors saved")


if __name__ == "__main__":
    if (
        "CUDA_VISIBLE_DEVICES" not in os.environ
        or "TOKENIZERS_PARALLELISM" not in os.environ
        or "HF_HUB_CACHE" not in os.environ
    ):
        raise ValueError("Missing env variables")

    parser = ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Retrain existing models")
    parser.add_argument("--dry-run", action="store_true", help="Train the model without saving outputs")
    parser_args = parser.parse_args()

    log.info("-" * 31 + "  start  " + "-" * 31)
    for d_info, h_info in gen_config():
        try:
            pretrain(d_info, h_info, parser_args)
        except Exception as e:
            log.error(e)
            print_exception(e)
    log.info("-" * 32 + "  end  " + "-" * 32)
