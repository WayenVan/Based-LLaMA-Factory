from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import datasets
import torch
from bidict import bidict
from evaluate import load
from transformers import PreTrainedTokenizer, Seq2SeqTrainer

from ..extras.constants import IGNORE_INDEX


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


@dataclass
class EastAfricaLanguageProcessor:
    mode: Literal["pt_source_to_translated", "pt_translated_to_source", "sft_training", "sft_eval_tokenized"]
    tokenizer: Optional[PreTrainedTokenizer] = None

    def __post_init__(self):
        pass

    def _pt_source_to_translated(self, example):
        return {"text": f"{example['source']}"}

    def _pt_translated_to_source(self, example):
        return {
            "text": f"{example['translation']}",
        }

    def _sft_training(self, example):
        conversations = [
            {
                "from": "human",
                "value": f"Translate {example['source_language']} to {example['translated_language']}:\nInput: {example['source']}\nOutput: ",
            },
            {
                "from": "gpt",
                "value": f"{example['translation']}",
            },
        ]
        return {"conversations": conversations}

    def _sft_eval_tokenized(self, example):
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided for sft_eval_tokenized mode.")

        messages = [
            {
                "role": "user",
                "content": f"Translate {example['source_language'].lower()} to {example['translated_language'].lower()}:\nInput: {example['source']}\nOutput: ",
            }
        ]
        tokenized_inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors=None,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized_inputs = self.tokenizer(tokenized_inputs)
        tokenized_inputs["source_language"] = AFRIMMT_LANGUAGES.inv[example["source_language"].lower().strip()]
        tokenized_inputs["translated_language"] = AFRIMMT_LANGUAGES.inv[example["translated_language"].lower().strip()]
        tokenized_inputs["labels"] = self.tokenizer(example["translation"]).input_ids
        return tokenized_inputs

    def __call__(self, example):
        if self.mode == "pt_source_to_translated":
            return self._pt_source_to_translated(example)
        elif self.mode == "pt_translated_to_source":
            return self._pt_translated_to_source(example)
        elif self.mode == "sft_training":
            return self._sft_training(example)
        elif self.mode == "sft_eval_tokenized":
            return self._sft_eval_tokenized(example)


def _print_data_example(tokenizer, example: dict[str, list[int]]) -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print(f"labels:\n{tokenizer.decode(valid_labels, skip_special_tokens=False)}")


def get_dataaset_and_fix_tokenizer_for_metrics(tokenizer, data_args, training_args, select_range=None):
    dataset_name = "sartifyllc/east_africa_language"

    dataset = datasets.load_dataset(dataset_name, split="test")
    tokenizer.add_bos_token = False  # no need because chat template already adds it
    tokenizer.padding_side = "left"

    ea_processor = EastAfricaLanguageProcessor(tokenizer=tokenizer, mode="sft_eval_tokenized")

    # get preprocessed dataset
    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        if select_range is not None:
            dataset = dataset.select(select_range)

        column_names = list(next(iter(dataset)).keys())
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )
        dataset = dataset.map(ea_processor, batched=False, **kwargs, remove_columns=column_names)
        print("Data example after tokenization for metrics:")
        _print_data_example(tokenizer, next(iter(dataset)))

        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels", "source_language", "translated_language"]
        )

    return dataset, tokenizer


def monkey_patch_s2strainer_for_metrics(trainer, collator=None):
    # hack the trainer's data_collator and columns settings
    if collator is not None:
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

        labels = inputs.pop("labels", None)

        loss, generated_tokens, _ = self.prediction_step_origin(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        return loss, generated_tokens, (source_lang_id, target_lang_id, labels)

    trainer.prediction_step = prediction_step.__get__(trainer, Seq2SeqTrainer)

    return trainer


def swap_metrics(predictions: list[str], labels: list[str], source_langs: list[str], target_langs: list[str]):
    grouped_results = defaultdict(lambda: {"preds": [], "labels": []})
    charf = load("chrf")
    # group for each language pair
    for i in range(len(predictions)):
        lang_piar_id = f"{source_langs[i]}->{target_langs[i]}"
        grouped_results[lang_piar_id]["preds"].append(predictions[i])
        grouped_results[lang_piar_id]["labels"].append(labels[i])

    # calcualted the metrics for each language pair
    metrics = []
    for lang_pair, texts in grouped_results.items():
        preds = texts["preds"]
        refs = texts["labels"]

        n_sampls = len(preds)

        # compute BLEU
        import sacrebleu

        bleu = sacrebleu.corpus_bleu(preds, [[ref] for ref in refs])

        # compute chrF
        chrf_score = charf.compute(predictions=preds, references=refs)
        metrics.append(
            {
                "language_pair": lang_pair,
                "n_samples": n_sampls,
                "bleu_score": round(bleu.score, 6),
                "chrf_score": round(chrf_score["score"], 6),
            }
        )
    metrics = sorted(metrics, key=lambda x: x["language_pair"])

    return metrics
