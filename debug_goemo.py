from datasets import load_dataset

ds = load_dataset("go_emotions")
print("Columns in GoEmotions:", ds["train"].column_names)
print("Example row:", ds["train"][0])
