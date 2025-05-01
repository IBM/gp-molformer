from argparse import ArgumentParser
from functools import partial
import os

from datasets import Dataset, DatasetDict, load_from_disk
import networkx as nx
import numpy as np
import pandas as pd
from peft import get_peft_config, get_peft_model
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import qed, MolLogP
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from scripts.DRD2_Score import drd2_scorer
from scripts.SA_Score import sascorer

# suppress warnings
RDLogger.DisableLog("rdApp.*")


def nan_on_error(func):
    def wrapper(*args):
        try:
            return func(*args)
        except:
            return np.nan

    return wrapper


QED = nan_on_error(qed)
drd2 = nan_on_error(drd2_scorer.get_score)


# Copied from https://github.com/wengong-jin/iclr19-graph2graph/blob/master/props/properties.py#L35-L65
@nan_on_error
def penalized_logp(mol):
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle


class FrozenEmbeddingMinusUnk(torch.nn.Module):
    def __init__(self, word_embeddings, unk_token_id=-1):
        super().__init__()
        frozen_weights1 = word_embeddings.weight[:unk_token_id]
        frozen_weights2 = word_embeddings.weight[unk_token_id + 1 :]
        self.frozen1 = torch.nn.Parameter(frozen_weights1, requires_grad=False)
        self.frozen2 = torch.nn.Parameter(frozen_weights2, requires_grad=False)
        self.unk = torch.nn.Parameter(word_embeddings.weight[[unk_token_id]])

    def forward(self, input):
        return F.embedding(input, torch.cat((self.frozen1, self.unk, self.frozen2)))


class DataCollatorForPairTuning(DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors)
        # replace first <eos> with <unk>
        sep = [torch.nonzero(ids == tokenizer.sep_token_id)[0] for ids in batch["input_ids"]]
        sep = [torch.arange(len(sep)), torch.cat(sep)]
        batch["input_ids"][sep] = tokenizer.unk_token_id
        # mask everything except (<unk> +) final molecule
        batch["token_type_ids"][sep] = 1
        batch["labels"] = batch["input_ids"].where(batch["token_type_ids"].to(bool), -100)
        del batch["token_type_ids"]
        return batch


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop("k", 125)
        super().__init__(*args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        max_new_tokens = model.base_model.molformer.config.max_position_embeddings - 2  # 200
        try:
            logits = model.generate(
                **inputs,
                do_sample=True,
                top_k=None,
                num_return_sequences=self.k,
                max_new_tokens=max_new_tokens,
            )
        except RuntimeError as e:
            print(e)
            logits = inputs["input_ids"].repeat_interleave(self.k, dim=0)
        # evaluation_loop truncates last batch based on first dimension and pads based on second
        logits = logits.reshape(len(inputs["input_ids"]), self.k, -1).transpose(1, 2)
        return (None, logits, logits)


def compute_metrics(p, prop=None, k=125):
    # invert transpose and reshaping
    pred_tok = p.predictions.transpose((0, 2, 1)).reshape(-1, p.predictions.shape[1])
    # Trainer pads with -100
    pred_tok = np.where(pred_tok == -100, tokenizer.pad_token_id, pred_tok)

    string = tokenizer.batch_decode(pred_tok)
    init_str = [s.split(tokenizer.unk_token)[0].split(tokenizer.cls_token)[1] for s in string]
    pred_str = [s.split(tokenizer.unk_token)[1].split(tokenizer.sep_token)[0] for s in string]

    init_mol = [Chem.MolFromSmiles(s) for s in init_str[::k]]
    pred_mol = [Chem.MolFromSmiles(s) for s in pred_str]

    pred_valid = [m for m in pred_mol if m is not None]
    nunique = len({Chem.MolToSmiles(m, isomericSmiles=False) for m in pred_valid})

    if prop is None:
        return {"valid": len(pred_valid) / len(pred_mol), "unique": nunique / len(pred_valid)}

    init_prop = np.array([prop(m) for m in init_mol]).reshape(-1, 1)
    pred_prop = np.array([prop(m) for m in pred_mol]).reshape(-1, k)
    best_prop = np.nanmax(pred_prop, axis=1)
    # np.nanargmax throws error if row is all NaN
    pred_prop[np.isnan(pred_prop).all(axis=1)] = 0
    pred_mol = np.array(pred_mol).reshape(-1, k)
    best_mol = pred_mol[np.arange(pred_mol.shape[0]), np.nanargmax(pred_prop, axis=1)]
    best_valid = [m for m in best_mol if m is not None]
    nunique_best = len({Chem.MolToSmiles(m, isomericSmiles=False) for m in best_valid})

    return {
        "valid": len(pred_valid) / pred_mol.size,
        "unique": nunique / len(pred_valid),
        "avg_prop": np.nanmean(pred_prop),
        "avg_prop_diff": np.nanmean(pred_prop - init_prop),
        "valid_best": len(best_valid) / len(best_mol),
        "unique_best": nunique_best / len(best_valid),
        "avg_best_prop": np.nanmean(best_prop),
        "avg_best_prop_diff": np.nanmean(best_prop - init_prop.ravel()),
        "best_prop": np.nanmax(best_prop),
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "task",
        default="qed",
        choices=os.listdir("data/pairtune"),
        help="Pair-tuning task (see data/pairtune/*)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ibm-research/GP-MoLFormer-Uniq",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=125,
        help="Number of generations per validation molecule",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Total batch size (divided amongst all GPUs)",
    )
    parser.add_argument("--lr", type=float, default=3e-2, help="learning rate")
    parser.add_argument("--num_epochs", type=float, default=100, help="Number of training epochs")
    parser.add_argument("--eval_epochs", type=int, default=1, help="Number of epochs between evals")
    parser.add_argument("--lamb", action="store_true", help="use lamb optimizer or not")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, hidden_dropout_prob=0.0, embedding_dropout_prob=0.0, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "ibm-research/MoLFormer-XL-both-10pct", padding_side="left", trust_remote_code=True
    )

    peft_args = {
        "peft_type": "PROMPT_TUNING",
        "task_type": "CAUSAL_LM",
        "num_virtual_tokens": 20,
    }
    peft_config = get_peft_config(peft_args)
    model = get_peft_model(model, peft_config)
    # unfreeze <unk>
    # NOTE: this is used instead of the model's input embedding
    model.word_embeddings = FrozenEmbeddingMinusUnk(model.word_embeddings, tokenizer.unk_token_id)

    data_dir = f"data/pairtune/{args.task}/"
    try:
        dataset = load_from_disk(data_dir)
    except FileNotFoundError:
        data_files = {
            "train": pd.read_csv(data_dir + "/train_pairs.txt", sep=" ", header=None),
            "test": pd.read_csv(data_dir + "/test.txt", header=None),
        }
        dataset = DatasetDict({k: Dataset.from_pandas(v) for k, v in data_files.items()})
        dataset = dataset.map(
            lambda r: tokenizer(
                Chem.CanonSmiles(r["0"], useChiral=0),
                Chem.CanonSmiles(r["1"], useChiral=0) if "1" in r else None,
                return_token_type_ids=True,
            ),
            remove_columns=dataset["test"].column_names,
        )
        dataset["train"] = dataset["train"].remove_columns("1")
        dataset.save_to_disk(data_dir)

    prop = {"drd2": drd2, "logp06": penalized_logp, "qed": QED}.get(args.task)
    n_gpus = max(torch.cuda.device_count(), 1)
    steps_per_epoch = int(np.ceil(len(dataset["train"]) / args.batch_size))
    targs = TrainingArguments(
        output_dir=f"models/pairtune/{args.task}/",
        evaluation_strategy="steps",
        eval_steps=args.eval_epochs * steps_per_epoch,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size // n_gpus,
        per_device_eval_batch_size=8,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        save_strategy="steps",
        save_steps=10 * steps_per_epoch,
        save_total_limit=3,
        remove_unused_columns=False,
        # adam_beta1=0.9,
        # adam_beta2=0.999,
        # load_best_model_at_end=True,
    )
    model.to(targs.device)  # NOTE: FusedLAMB uses params[0].device!
    opt = None  # AdamW by default
    if args.lamb:
        from apex import optimizers

        params = [p for p in model.parameters() if p.requires_grad]
        opt = optimizers.FusedLAMB(
            params,
            lr=args.lr,
            weight_decay=0.0,
            grad_averaging=False,
            max_grad_norm=100,
            bias_correction=False,
        )
    model.print_trainable_parameters()
    print([(n, p.shape) for n, p in model.named_parameters() if p.requires_grad])
    trainer = CustomTrainer(
        model=model,
        args=targs,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=partial(compute_metrics, prop=prop, k=args.k),
        tokenizer=tokenizer,
        data_collator=DataCollatorForPairTuning(tokenizer, mlm=False),
        optimizers=(opt, None),
        k=args.k,
    )
    trainer.train()
