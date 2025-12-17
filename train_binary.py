import os
import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

def load(folder):
    tr = pd.read_csv(f"{folder}/binary_train.csv")
    va = pd.read_csv(f"{folder}/binary_validation.csv")
    return tr, va

def prep(df, tok):
    texts = list(df["text"].astype(str))
    enc = tok(texts, truncation=True, padding=False)
    enc["labels"] = df["label"].astype(int).tolist()
    return Dataset.from_dict(enc)

def metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--output_dir", default="models/binary_model")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    if os.path.exists(args.output_dir) and not os.path.isdir(args.output_dir):
        os.remove(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    tr, va = load(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = prep(tr, tokenizer)
    val_ds = prep(va, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Binary model saved to:", args.output_dir)
