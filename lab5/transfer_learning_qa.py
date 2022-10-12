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
from typing import List, Dict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import default_data_collator
from transformers.trainer_utils import EvalPrediction
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments
from process_dataset import _prepare_data

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

def _obtain_data(data_dir: str):
    """Return pytorch data train and test tuple.

    Args:
        data_dir: directory where the .py data file is loaded.

    Returns:
        Tuple: pytorch data objects
    """

    tokenized_train_dataset, tokenized_eval_dataset = _prepare_data(data_dir)

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
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(saved_model_tar, ".")
    # load model and tokenizer
    model_checkpoint = constants.MODEL_NAME_DIR
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    return model, tokenizer


def _compute_metrics(pred) -> dict:
    """Computes accuracy, precision, and recall.
    This function is a callback function that the Trainer calls when evaluating.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    print("=================f1====================")
    print(f1)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def run_with_args(args):
    """Run training."""
    model, tokenizer = _get_model_and_tokenizer(args=args)

    # train_dataset, eval_dataset = _prepare_data(args.train, tokenizer)
    train_dataset, eval_dataset = _obtain_data(args.train) # args.train 指数据集目录


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
        save_strategy = "epoch",
        logging_dir=".",
        learning_rate=float(args.adam_learning_rate),
        # load_best_model_at_end=False, # Load the optimal model after training is complete
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
