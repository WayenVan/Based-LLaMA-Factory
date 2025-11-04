from typing import Any, Callable, Optional, Union

import datasets
import torch
from bidict import bidict
from transformers import DataCollatorWithPadding, Seq2SeqTrainer

from ..extras.constants import IGNORE_INDEX


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


def _print_data_example(tokenizer, example: dict[str, list[int]]) -> None:
    # valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    # print("label_ids:\n{}".format(example["labels"]))
    # print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


AFRIMMT_LANGUAGES = bidict(
    {
        0: "acholi",
        1: "alur",
        2: "aringa",
        3: "ateso",
        4: "bena",
        5: "borana",
        6: "chaga",
        7: "chonyi",
        8: "chuka",
        9: "duruma",
        10: "embu",
        11: "english",
        12: "giriama",
        13: "gogo",
        14: "gusii",
        15: "gwere",
        16: "haya",
        17: "hehe",
        18: "jita",
        19: "jopadhola",
        20: "kakwa",
        21: "kalenjin",
        22: "kamba",
        23: "kebu",
        24: "kikuyu",
        25: "kinyarwanda",
        26: "kumam",
        27: "langi",
        28: "lango",
        29: "luganda",
        30: "luhya",
        31: "luo",
        32: "maasai",
        33: "makonde",
        34: "marakwet",
        35: "masaba",
        36: "meru",
        37: "nyole",
        38: "nyoro",
        39: "oromo",
        40: "pare",
        41: "pokomo",
        42: "pokot",
        43: "samburu",
        44: "soga",
        45: "somali",
        46: "suba",
        47: "sukuma",
        48: "swahili",
        49: "taita",
        50: "taveta",
        51: "tigrinya",
        52: "zanaki",
    }
)


def afrimmt_metrics_swap(
    trainer: Seq2SeqTrainer,
    model_args,
    data_args,
    training_args,
    finetuning_args,
    generating_args,
):
    dataset_name = "sartifyllc/east_africa_language"

    dataset = datasets.load_dataset(dataset_name, split="test")
    tokenizer = trainer.tokenizer
    tokenizer.add_bos_token = False  # no need because chat template already adds it

    def preprocess_function(example):
        messages = [
            {
                "role": "user",
                "content": f"Translate {example['source_language'].lower()} to {example['translated_language'].lower()}:\nInput: {example['source']}\noutput: ",
            }
        ]
        tokenized_inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors=None,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized_inputs = tokenizer(tokenized_inputs)
        tokenized_inputs["source_language"] = AFRIMMT_LANGUAGES.inv[example["source_language"].lower().strip()]
        tokenized_inputs["translated_language"] = AFRIMMT_LANGUAGES.inv[example["translated_language"].lower().strip()]
        return tokenized_inputs

    # get preprocessed dataset
    with trainer.args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        column_names = list(next(iter(dataset)).keys())
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (trainer.args.local_process_index != 0),
            desc="Converting format of dataset",
        )
        dataset = dataset.map(preprocess_function, batched=False, **kwargs, remove_columns=column_names)
        _print_data_example(tokenizer, next(iter(dataset)))

    torch_dataset = dataset.with_format("torch")
    collator = DataCollatorWithPadding(tokenizer)

    # hack the trainer's data_collator and columns settings
    trainer.data_collator = collator
    trainer.args.remove_unused_columns = False
    trainer.args.predict_with_generate = True
    trainer.compute_metrics = None

    # monkey patch the prediction step to return source and target language ids
    trainer.prediction_step_origin = trainer.prediction_step

    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ):
        source_lang_id = inputs.pop("source_language", None)
        target_lang_id = inputs.pop("translated_language", None)

        _ = inputs.pop("labels", None)

        loss, generated_tokens, _ = self.prediction_step_origin(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        return loss, generated_tokens, (source_lang_id, target_lang_id)

    trainer.prediction_step = prediction_step.__get__(trainer, Seq2SeqTrainer)

    results = trainer.predict(torch_dataset)
    return


register_post_train_task(
    "afrimmt_metrics_swap",
    afrimmt_metrics_swap,
)
