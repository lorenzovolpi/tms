import os
from dataclasses import dataclass
from itertools import batched

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

from data import DatasetInfo

VERBOSE = True


def dataset(name: str, n: int):
    return DatasetInfo(name, "sentiment", n)


def sout(*args):
    if VERBOSE:
        print(*args)


model_names = [
    "google-bert/bert-base-uncased",
]

dataset_names = [
    dataset("stanfordnlp/imdb", 2),
]


@dataclass
class SentimentArgs:
    model_name: str
    d_info: DatasetInfo
    max_length: int = 512
    nepochs: int = 3
    train_batchsize: int = 64
    embed_batchsize: int = 512
    val_size: float = 0.2
    lr: float = 5e-4
    train_backbone: bool = False
    device: str = "cuda"

    @property
    def dataset_name(self) -> str:
        return self.d_info.name

    @property
    def num_classes(self) -> int:
        return self.d_info.n_classes


def get_tr_outdir(args: SentimentArgs):
    model_name = args.model_name.split("/")[-1]
    dataset_name = args.dataset_name.split("/")[-1]
    outdir = os.path.join("output", "tms", "models", dataset_name, model_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def get_embed_outdir(args):
    model_name = args.model_name.split("/")[-1]
    dataset_name = args.dataset_name.split("/")[-1]
    outdir = os.path.join("output", "tms", "pretrain", "sentiment", dataset_name, model_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def get_dataset(args: SentimentArgs):
    """
    Load dataset and create validation split if does not exist.
    Also check that the number of classes matches the expected number.
    """
    dataset = load_dataset(args.dataset_name)

    if "validation" not in dataset:
        sout("splitting training set into train/validation...")
        _tmp_dataset = dataset["train"].train_test_split(test_size=args.val_size)
        dataset["train"] = _tmp_dataset["train"]
        dataset["validation"] = _tmp_dataset["test"]

    n_inferred_classes = dataset["train"].shape[-1]
    if args.num_classes != n_inferred_classes:
        sout(
            f"number of inferred target classes ({n_inferred_classes}) != number of given target classes ({args.num_classes})"
        )

    return dataset


def get_classifier(model_name, n_classes=2, use_bfloat16=False, device="cuda"):
    torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes, torch_dtype=torch_dtype
    ).to(device)
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


def train_model(args: SentimentArgs, model, dataset):
    training_outdir = get_tr_outdir(args)

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
        metric_for_best_model="f1",
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

def save_dataset(data:dict):
    

def pretrain(args: SentimentArgs):
    sout(f"- model: {args.model_name}")
    sout(f"- dataset: {args.dataset_name}")

    dataset = get_dataset(args)

    model, tokenizer = get_classifier(model_name=args.model_name, n_classes=args.num_classes, device=args.device)
    model = prepare_model(args, model)

    dataset = tokenize_dataset(args, tokenizer, dataset)

    train_model(args, model, dataset)

    # Get embedddings and logits
    embeds_outdir = get_embed_outdir(args)
    sout("\nEmbedding...")
    sout(f"- storing embeddings in {embeds_outdir}")
    splits = ["validation", "test"]
    for split in splits:
        split_data = dataset[split]
        split_y, split_logits, split_last_hiddens = embed(
            model, tokenizer, data=split_data, selection_strategy=get_cls_bertlike, args=args
        )

        # torch.save(split_logits, os.path.join(embeds_outdir, f"logits.{split}.pt"))
        # torch.save(split_last_hiddens, os.path.join(embeds_outdir, f"hidden_states.{split}.pt"))


args = SentimentArgs(
    model_names[0],
    dataset_names[0],
    train_backbone=False,
)

if __name__ == "__main__":
    if (
        "CUDA_VISIBLE_DEVICES" not in os.environ
        or "TOKENIZERS_PARALLELISM" not in os.environ
        or "HF_HUB_CACHE" not in os.environ
    ):
        raise ValueError("Missing env variables")

    pretrain(args)
