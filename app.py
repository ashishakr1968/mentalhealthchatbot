import torch
import json
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("🔧 Using device:", device)

EMO_MODEL_PATH = "models/emotion_model"

emotion_model = AutoModelForSequenceClassification.from_pretrained(
    EMO_MODEL_PATH
).to(device)

emotion_tok = AutoTokenizer.from_pretrained(EMO_MODEL_PATH)

with open("data/emotion_labels.json") as f:
    GO = json.load(f)

try:
    thresholds = json.load(open(f"{EMO_MODEL_PATH}/thresholds.json"))
    print("✨ Loaded thresholds:", thresholds)
except:
    print("⚠️ thresholds.json not found. Using default threshold=0.5")
    thresholds = [0.5] * len(GO)

def breathing(msg="", seconds=1.5):
    print(msg, end="", flush=True)
    for _ in range(3):
        print(".", end="", flush=True)
        time.sleep(seconds / 3)
    print()

def type_out(text, delay=0.02):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

def grounding_exercise():
    exercises = [
        "🌬️ *Breathe with me…*\nIn slowly through your nose…\nhold for a heartbeat…\nand let the air fall out gently.\n\nLet’s do this three times together.",
        "🌿 *Look around you…*\nName five gentle things you can see.\nColors, shadows, light… anything soft.\nYou're returning to the present moment.",
        "🕊️ *Place your hand on your heart…*\nFeel its quiet rhythm.\nLet each beat remind you:\n“You’re here. You’re safe. You’re real.”",
        "🌙 *A warm-light visualization…*\nImagine a soft glow on your chest.\nEach breath spreads it through your body,\nmelting the tension away.",
        "🍃 *5–4–3–2–1 grounding…*\n5 things you can see\n4 things you can touch\n3 things you hear\n2 things you smell\n1 thing inside your heart.\nYou are here.",
        "☁️ *Relax your face…*\nUnclench your jaw.\nSoften your shoulders.\nBreathe gently.\nYour body deserves this ease."
    ]
    return random.choice(exercises)

def select_primary_emotion(emotions):
    priority = [
    "sadness", "grief", "disappointment", "loneliness",
    "anger", "annoyance", "disapproval", "disgust",
    "fear", "anxiety", "nervousness",
    "joy", "contentment", "optimism", "gratitude", "excitement",
    "love", "affection", "caring", "admiration",
    "confusion", "curiosity", "surprise",
    "neutral", "trust"
    ]

    for p in priority:
        if p in emotions:
            return p
    return emotions[0]

def predict_emo(text):
    enc = emotion_tok(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        logits = emotion_model(**enc).logits

    probs = torch.sigmoid(logits).cpu().numpy()[0]

    preds = [GO[i] for i, p in enumerate(probs) if p >= thresholds[i]]

    if len(preds) < 2:
        sorted_idx = np.argsort(probs)[::-1]
        preds = [GO[sorted_idx[0]], GO[sorted_idx[1]]]

    return preds



def override_emotion(user_text, emotions):
    text = user_text.lower()

    # Normalize apostrophes: convert “don’t” → "dont"
    text = text.replace("’", "'").replace("don't", "dont")

    anger_words = [
        "angry", "annoyed", "irritated", "pissed", "furious",
        "mad", "frustrated", "rage", "hate", "irritating"
    ]

    sadness_words = [
        "sad", "down", "upset", "broken", "hurt", "depressed"
    ]

    fear_words = [
        "scared", "afraid", "terrified", "anxious", "panic", "panicking"
    ]

    joy_words = [
        "happy", "excited", "exciting", "proud", "joy", "delighted",
        "accomplished", "celebrating", "won", "achieved"
    ]

    love_words = [
        "love", "loving", "care", "caring", "affection",
        "crush", "falling for", "deep connection"
    ]

    confusion_words = [
        "dont know", "don't know", "confused", "confusing",
        "unclear", "lost", "mixed up", "overwhelmed",
        "foggy", "blank", "unsure"
    ]

    # 🔥 Rule-based Overrides
        # 💜 Special blended emotion case — sadness beneath anger
    if (("angry" in text or "anger" in text) and 
        ("sad" in text or "sadness" in text or "really sad" in text)):
        return ["sadness", "anger"]

    if any(w in text for w in anger_words):
        return ["anger", "annoyance"]

    if any(w in text for w in sadness_words):
        return ["sadness"]

    if any(w in text for w in fear_words):
        return ["fear", "nervousness"]

    if any(w in text for w in joy_words):
        return ["joy", "excitement"]

    if any(w in text for w in love_words):
        return ["love", "affection"]

    # ⭐ Strong confusion override
    if any(w in text for w in confusion_words):
        return ["confusion", "uncertainty"]

    return emotions



def generate_reply(user_input, emotions):
    emo = select_primary_emotion(emotions)

    if emo in ["sadness", "disappointment", "grief", "loneliness"]:
        return (
            "☁️ Your words feel like dusk settling on a quiet heart.\n"
            "I’m sitting beside this softness with you.\n\n"
            "❦ *“Some nights the moon cries in silver,\n"
            "    but even then she lights the sea.”*\n\n"
            "What does your heart whisper beneath the ache?"
        )

    elif emo in ["anger", "annoyance", "disapproval", "disgust"]:
        return (
            "🔥 There’s a storm swirling inside you — I can feel its heat.\n"
            "Your fire speaks of things that matter.\n\n"
            "⟡ *“Even the sun flares before it rests.\n"
            "    Even flames need someone to hear their crackle.”*\n\n"
            "Tell me what sparked this burning in your chest."
        )

    elif emo in ["fear", "anxiety", "nervousness", "apprehension", "overwhelm"]:
        return (
            "🌫️ I sense trembling in your thoughts… like your heart is holding too much.\n"
            "Let’s find your breath again.\n\n"
            f"{grounding_exercise()}\n\n"
            "Whenever you're ready, you can tell me what's weighing on you."
        )

    elif emo in ["joy", "contentment", "optimism", "gratitude", "excitement"]:
        return (
            "✨ Your words shimmer — I can almost feel the light they carry.\n"
            "Let’s hold this glow a little longer.\n\n"
            "✧ *“Joy is a lantern in the ribs,\n"
            "    glowing even when the world forgets to clap.”*\n\n"
            "What moment brought this spark alive in you?"
        )

    elif emo in ["love", "affection", "caring", "admiration"]:
        return (
            "♡ Your heart feels warm — like a candle in a quiet room.\n"
            "Love leaves a soft trail wherever it walks.\n\n"
            "❥ *“Some feelings don’t speak in words,\n"
            "    they press themselves gently into the soul.”*\n\n"
            "Tell me what tender thing is blooming inside you."
        )

    elif emo in ["confusion", "curiosity", "surprise"]:
        return (
            "⟢ Your thoughts feel like drifting stardust — searching for a shape.\n"
            "Wonder is its own quiet magic.\n\n"
            "⧖ *“Not all constellations are named;\n"
            "    some are simply felt in the dark.”*\n\n"
            "What question is tugging at your mind?"
        )

    else:
        return (
            "☁︎ I'm here in this moment with you.\n"
            "Your presence hums like a quiet melody.\n\n"
            "✦ *“Even ordinary breaths\n"
            "    carry galaxies if you listen closely.”*\n\n"
            "Tell me more — I’m listening."
        )

def chat():
    print("\n💬 Mental Health Chatbot\n")
    print("I'm here to listen. You can talk to me about anything.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            type_out("Take gentle care of yourself. I'm here whenever you return. 💛")
            break

        if user_input.lower() in ["ground me", "help me calm down", "i’m anxious", "i'm anxious", "i'm scared"]:
            type_out(grounding_exercise())
            continue

        breathing("Reading your emotions")
        raw_emotions = predict_emo(user_input)
        emotions = override_emotion(user_input, raw_emotions)

        print("Detected emotions:", emotions)

        reply = generate_reply(user_input, emotions)

        breathing("Thinking")
        type_out("Bot: " + reply + "\n")

if __name__ == "__main__":
    chat()




