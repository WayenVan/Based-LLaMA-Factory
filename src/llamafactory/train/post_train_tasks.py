from typing import Callable

import numpy as np
from pandas import DataFrame
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer

import wandb

from ..extras.constants import IGNORE_INDEX
from ..extras.east_africa_language_utils import (
    AFRIMMT_LANGUAGES,
    get_dataaset_and_fix_tokenizer_for_metrics,
    monkey_patch_s2strainer_for_metrics,
    swap_metrics,
)
from ..extras.misc import numpify


POST_TRAIN_TASkS: dict[str, Callable] = {}


def register_post_train_task(task_name: str, func: Callable) -> None:
    r"""Register a post-train task function."""
    global POST_TRAIN_TASkS
    if task_name in POST_TRAIN_TASkS:
        raise ValueError(f"Post-train task {task_name} is already registered.")

    POST_TRAIN_TASkS[task_name] = func


def get_post_train_task(task_name: str) -> Callable:
    r"""Get a registered post-train task function."""
    global POST_TRAIN_TASkS
    if task_name not in POST_TRAIN_TASkS:
        raise ValueError(f"Post-train task {task_name} is not registered.")

    return POST_TRAIN_TASkS[task_name]


def afrimmt_metrics_swap(
    trainer: Seq2SeqTrainer,
    model_args,
    data_args,
    training_args,
    finetuning_args,
    generating_args,
):
    if trainer.accelerator.is_main_process:
        print("=================Starting AFRIMMT metrics swap evaluation...===========")
    torch_dataset, tokenizer = get_dataaset_and_fix_tokenizer_for_metrics(
        trainer.tokenizer, data_args, training_args, select_range=None
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=trainer.accelerator.unwrap_model(trainer.model))

    trainer = monkey_patch_s2strainer_for_metrics(trainer, collator=collator)

    results = trainer.predict(torch_dataset, max_new_tokens=128, do_sample=False)

    if trainer.accelerator.is_main_process:
        preds, labels = numpify(results.predictions), numpify(results.label_ids[2])
        source_lang_ids, target_lang_ids = numpify(results.label_ids[0]), numpify(results.label_ids[1])

        preds = np.where(preds != IGNORE_INDEX, preds, tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        source_langs = [AFRIMMT_LANGUAGES[lang_id] for lang_id in source_lang_ids]
        target_langs = [AFRIMMT_LANGUAGES[lang_id] for lang_id in target_lang_ids]

        results = swap_metrics(
            decoded_preds,
            decoded_labels,
            source_langs,
            target_langs,
        )

        df = DataFrame(results)
        wandb.log({"afrimmt_metrics_swap": wandb.Table(dataframe=df)})

    return


register_post_train_task(
    "afrimmt_metrics_swap",
    afrimmt_metrics_swap,
)
