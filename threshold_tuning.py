import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------
# Load data & model
# ------------------------------
DATA_DIR = "data"
MODEL_DIR = "models/emotion_model"

val_df = pd.read_csv(f"{DATA_DIR}/emotion_validation.csv")
label_list = json.load(open(f"{DATA_DIR}/emotion_labels.json"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# ------------------------------
# Convert labels to proper format
# ------------------------------
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

labels_raw = val_df["label"].apply(normalize_label).tolist()
labels_raw = [[y for y in lst if 0 <= y < len(label_list)] for lst in labels_raw]

# Multi-hot
Y = np.zeros((len(labels_raw), len(label_list)))
for i, lst in enumerate(labels_raw):
    for lab in lst:
        Y[i, lab] = 1

# ------------------------------
# Compute probabilities on val set
# ------------------------------
probs = []

for text in val_df["text"].tolist():
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    p = torch.sigmoid(logits).cpu().numpy()[0]
    probs.append(p)

probs = np.array(probs)

# ------------------------------
# Find best threshold per label
# ------------------------------
best_thresholds = []

for i in range(len(label_list)):
    best_f1 = 0
    best_t = 0.5
    for t in np.linspace(0.1, 0.9, 41):
        pred = (probs[:, i] >= t).astype(int)
        f1 = f1_score(Y[:, i], pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    best_thresholds.append(best_t)

print("\nBest thresholds:", best_thresholds)

json.dump(best_thresholds, open(f"{MODEL_DIR}/thresholds.json", "w"))
print(f"\nSaved → {MODEL_DIR}/thresholds.json")
