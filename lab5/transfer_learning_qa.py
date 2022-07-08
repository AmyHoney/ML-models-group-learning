import argparse
import json
import logging
import os
import pathlib
import sys
import tarfile
from typing import Tuple

from constants import constants
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from transformers import default_data_collator
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments


root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
root.addHandler(handler)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING")) #dataset
    # parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    # parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    parser.add_argument("--pretrained-model", type=str, default=os.environ.get("SM_CHANNEL_MODEL"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--adam-learning-rate", type=float, default=5e-5)
    return parser.parse_known_args()

def prepare_train_features(examples):

    # remove the left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride.
    # tokenizer = AutoTokenizer.from_pretrained("./pytorch-eqa-distilbert-base-cased")
    model_checkpoint = constants.MODEL_NAME_DIR
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    pad_on_right = tokenizer.padding_side == "right"

    tokenized_train_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=constants.MAX_SEQ_LENGTH,
        stride=constants.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_train_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_train_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_train_examples["start_positions"] = []
    tokenized_train_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_train_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_train_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_train_examples["start_positions"].append(cls_index)
            tokenized_train_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_train_examples["start_positions"].append(cls_index)
                tokenized_train_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_train_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_train_examples["end_positions"].append(token_end_index + 1)

    return tokenized_train_examples

def prepare_validation_features(examples):
    # remove the left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. 
    # tokenizer = AutoTokenizer.from_pretrained("./pytorch-eqa-distilbert-base-cased")
    model_checkpoint = constants.MODEL_NAME_DIR
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    pad_on_right = tokenizer.padding_side == "right"

    tokenized_val_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=constants.MAX_SEQ_LENGTH,
        stride=constants.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_val_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_val_examples["example_id"] = []

    for i in range(len(tokenized_val_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_val_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_val_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_val_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_val_examples["offset_mapping"][i])
        ]

    return tokenized_val_examples
 
def _prepare_data(data_dir: str, tokenizer):
    """Return pytorch data train and test tuple.

    Args:
        data_dir: directory where the .py data file is loaded.
        tokenizer: tokenizer from the huggingface library.

    Returns:
        Tuple: pytorch data objects
    """

    # load dataset from py file
    dataset = load_dataset(
        os.path.join(data_dir, constants.INPUT_DATA_FILENAME_PY),
    )

    # preprocess dataset
    tokenized_dataset = dataset.map(
        prepare_train_features, 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )

    tokenized_eval_dataset = dataset["validation"].map(
        prepare_validation_features,
        batched=True,
        remove_columns=dataset["validation"].column_names
    )

    tokenized_train_dataset = tokenized_dataset["train"]
    tokenized_eval_dataset = tokenized_eval_dataset

    # train_test_split dataset
    print("=================train====================")
    print(tokenized_train_dataset)
    print("=================validation====================")
    print(tokenized_eval_dataset)

    # return preprocessed_dataset[constants.TRAIN], preprocessed_dataset[constants.VALIDATION]
    return tokenized_train_dataset, tokenized_eval_dataset


def _get_model_and_tokenizer(args) -> Tuple[AutoModelForQuestionAnswering, AutoTokenizer]:
    """Extract model files and load model and tokenizer.

    Args:
        args: parser argument
    Returns:
        object: a tuple of tokenizer and model.
    """
    pretrained_model_path = next(pathlib.Path(args.pretrained_model).glob(constants.TAR_GZ_PATTERN))

    # extract model files
    with tarfile.open(pretrained_model_path) as saved_model_tar:
        saved_model_tar.extractall(".")
    # load model and tokenizer
    model_checkpoint = constants.MODEL_NAME_DIR
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    return model, tokenizer


def _compute_metrics(pred) -> dict:
    """Computes accuracy, precision, and recall."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def run_with_args(args):
    """Run training."""
    model, tokenizer = _get_model_and_tokenizer(args=args)

    train_dataset, eval_dataset = _prepare_data(args.train, tokenizer)

    logging.info(f" loaded train_dataset sizes is: {len(train_dataset)}")
    logging.info(f" loaded eval_dataset sizes is: {len(eval_dataset)}")

    # define training args
    training_args = TrainingArguments(
        output_dir=".",
        save_total_limit=1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        logging_dir=".",
        learning_rate=float(args.adam_learning_rate),
        #load_best_model_at_end=False, #是否在训练结束时加载训练期间找到的最佳模型。
        #metric_for_best_model="f1", #The metric to use to compare two different models."
        disable_tqdm=True,
        logging_first_step=True,
        logging_steps=50,
    )
    
    # create Trainer instance (其实一边training 一边比较eval)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=_compute_metrics,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
    )

    # it executes the training loop and saves the best model.
    trainer.train()

    # Saves the model to s3
    trainer.model.save_pretrained(args.model_dir)
    trainer.tokenizer.save_pretrained(args.model_dir)
    with open(os.path.join(args.model_dir, constants.LABELS_INFO), "w") as nf:
        nf.write(json.dumps({constants.LABELS: [0, 1]}))


if __name__ == "__main__":
    args, unknown = _parse_args()
    run_with_args(args)
