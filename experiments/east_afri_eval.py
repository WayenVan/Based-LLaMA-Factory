import sys


sys.path.append("src")

import numpy as np
from pandas import DataFrame
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.east_africa_language_utils import (
    AFRIMMT_LANGUAGES,
    get_dataaset_and_fix_tokenizer_for_metrics,
    monkey_patch_s2strainer_for_metrics,
    swap_metrics,
)
from llamafactory.extras.misc import numpify
from llamafactory.hparams.data_args import DataArguments
from llamafactory.hparams.finetuning_args import FinetuningArguments
from llamafactory.hparams.training_args import TrainingArguments
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer


DATA_ARGS = DataArguments(preprocessing_num_workers=1)
TRAINING_ARGS = TrainingArguments(
    per_device_eval_batch_size=8,
    report_to="none",
)
FINETUNING_ARGS = FinetuningArguments()
CHECKPOINT_PATH = "outputs/gemma-1b-sft/2025-11-24_01-37-25/checkpoint-127268"


def main():
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    dataset, tokenizer = get_dataaset_and_fix_tokenizer_for_metrics(
        tokenizer, DATA_ARGS, TRAINING_ARGS, select_range=range(100)
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = CustomSeq2SeqTrainer(
        model=model,
        finetuning_args=FINETUNING_ARGS,
        tokenizer=tokenizer,
        processor=None,
        data_collator=collator,
        eval_dataset=dataset,
        args=TRAINING_ARGS,
    )

    trainer = monkey_patch_s2strainer_for_metrics(trainer, collator=collator)

    results = trainer.predict(dataset, max_new_tokens=128, do_sample=False)

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
        df.to_csv("outputs/east_afri_eval_results.csv", index=False)


if __name__ == "__main__":
    main()
