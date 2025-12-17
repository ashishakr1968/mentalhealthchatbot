import os
import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


os.makedirs("data", exist_ok=True)


GO_EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "trust", "neutral"
]


RISKY_EMOTIONS = [
    "anger","annoyance","disappointment","disapproval","disgust",
    "fear","grief","remorse","sadness","nervousness"
]

RISKY_IDS = [GO_EMOTIONS.index(e) for e in RISKY_EMOTIONS]


def load_goemotions():
    print("Loading GoEmotions…")
    ds = load_dataset("go_emotions")

    texts = ds["train"]["text"]
    labels = ds["train"]["labels"]  # list of lists

    df = pd.DataFrame({
        "text": texts,
        "labels": labels
    })

    print("Loaded:", len(df))
    return df


def build_binary_dataset(df):
    print("\nBuilding Corrected Binary Dataset…")

    def is_risky(label_list):
        return 1 if any(l in RISKY_IDS for l in label_list) else 0

    df["label"] = df["labels"].apply(is_risky)

    return df[["text", "label"]]


def build_emotion_dataset(df):
    print("\nBuilding Emotion Dataset…")

    # Keep full multi-label list
    df["label"] = df["labels"].apply(lambda x: x)

    return df[["text", "label"]]


def split_and_save(df, prefix):
    train, temp = train_test_split(df, test_size=0.20, random_state=42)
    val, test = train_test_split(temp, test_size=0.50, random_state=42)

    train.to_csv(f"data/{prefix}_train.csv", index=False)
    val.to_csv(f"data/{prefix}_validation.csv", index=False)
    test.to_csv(f"data/{prefix}_test.csv", index=False)

    print(f"Saved {prefix} → Train:{len(train)} | Val:{len(val)} | Test:{len(test)}")


if __name__ == "__main__":

    print("\n=== DATA PREPARATION STARTED ===")

    go = load_goemotions()

    
    binary_df = build_binary_dataset(go.copy())
    split_and_save(binary_df, "binary")

    # 2. EMOTION MULTI-LABEL DATASET
    emotion_df = build_emotion_dataset(go.copy())
    split_and_save(emotion_df, "emotion")

    # Save emotion label list
    with open("data/emotion_labels.json", "w") as f:
        json.dump(GO_EMOTIONS, f, indent=2)

    print("\n=== COMPLETED SUCCESSFULLY ===")
