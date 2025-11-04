from typing import TYPE_CHECKING, Callable, Union


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .parser import DatasetAttr


DATA_MAPPINGS: dict[str, Callable] = {}


def register_data_mapping(dataset_name: str, func: Callable) -> None:
    r"""Register a data mapping function."""
    global DATA_MAPPINGS
    if dataset_name in DATA_MAPPINGS:
        raise ValueError(f"Data mapping {dataset_name} is already registered.")

    DATA_MAPPINGS[dataset_name] = func


def apply_data_mapping(
    dataset_name: str,
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
):
    data_mapping = DATA_MAPPINGS.get(dataset_name, None)

    if data_mapping is not None:
        column_names = list(next(iter(dataset)).keys())
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
                desc="Converting format of dataset",
            )

        return dataset.map(
            data_mapping,
            batched=False,
            remove_columns=column_names,
            **kwargs,
        )
    return dataset


register_data_mapping(
    "east_aftrica_pt_source",
    lambda example: {
        # "text": f"Translate {example['source_language']} to {example['translated_language']}:\n input: {example['source']}\noutput: {example['translation']}"
        "text": f"{example['source']}",
    },
)

register_data_mapping(
    "east_aftrica_pt_translated",
    lambda example: {
        "text": f"{example['translation']}",
    },
)


def east_aftrica_sft(example):
    conversations = [
        {
            "from": "human",
            "value": f"Translate {example['source_language']} to {example['translated_language']}:\n input: {example['source']}\noutput: ",
        },
        {
            "from": "gpt",
            "value": f"{example['translation']}",
        },
    ]
    return {"conversations": conversations}


register_data_mapping(
    "east_africa_sft",
    east_aftrica_sft,
)
