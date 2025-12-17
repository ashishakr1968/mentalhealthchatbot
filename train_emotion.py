import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -------------------------------------------------------
# Text Cleaning
# -------------------------------------------------------
def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"[^\w\s.,!?']+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# -------------------------------------------------------
# Custom Trainer (MPS-safe weighted BCE)
# -------------------------------------------------------
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    # Transformers >=4.57 passes num_items_in_batch → include it
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        # ---- FORCE labels to float32 (critical for MPS) ----
        labels = inputs["labels"].to(torch.float32)
        inputs["labels"] = labels  # override so HF won't recast

        outputs = model(**inputs)

        # logits MUST be float32
        logits = outputs.logits.to(torch.float32)

        # weights MUST be float32
        pos_w = self.class_weights.to(logits.device, dtype=torch.float32)

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# -------------------------------------------------------
# Metrics
# -------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    return {"micro_f1": micro_f1}


# -------------------------------------------------------
# Preprocessing
# -------------------------------------------------------
def preprocess(df, tokenizer, label_list):
    df["text"] = df["text"].astype(str).apply(clean_text)

    def normalize_label(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                v = json.loads(x)
                return v if isinstance(v, list) else [int(v)]
            except:
                return []
        if isinstance(x, (int, float)):
            return [int(x)]
        return []

    labels_raw = df["label"].apply(normalize_label).tolist()

    max_label = len(label_list)
    labels_raw = [[y for y in lst if 0 <= y < max_label] for lst in labels_raw]

    mlb = MultiLabelBinarizer(classes=list(range(max_label)))
    Y = mlb.fit_transform(labels_raw).astype("float32")  # float32 for MPS-safe BCE

    enc = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=False,
        max_length=256,
    )

    enc["labels"] = Y.tolist()

    return Dataset.from_dict(enc)


# -------------------------------------------------------
# Main Training Flow
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--output_dir", default="models/emotion_model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load label set
    label_list = json.load(open(f"{args.data_dir}/emotion_labels.json"))
    num_labels = len(label_list)

    # Load dataset CSVs
    train_df = pd.read_csv(f"{args.data_dir}/emotion_train.csv")
    val_df = pd.read_csv(f"{args.data_dir}/emotion_validation.csv")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = preprocess(train_df, tokenizer, label_list)
    val_ds = preprocess(val_df, tokenizer, label_list)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    # Class weights (inverse frequency)
    label_counts = np.sum(np.vstack(train_ds["labels"]), axis=0)
    class_weights = torch.tensor(
        1.0 / (label_counts + 1e-6), dtype=torch.float32
    )  # float32 → MPS compatible

    # TrainingArguments — FULLY UPDATED FOR TRANSFORMERS 4.57
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=50,

        eval_strategy="epoch",       # <-- new name for transformers 4.57
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",

        fp16=False,                  # MPS cannot use fp16
        bf16=False,                  # MUST be disabled for MPS
        gradient_checkpointing=True,

        warmup_ratio=0.1,
        learning_rate=2e-5,

        report_to="none",            # No WandB noise
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n🔥 Training complete! Model saved to:", args.output_dir)
