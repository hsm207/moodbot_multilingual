from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch


DATA = [
    {"text": "hey", "label": "greet"},
    {"text": "hello", "label": "greet"},
    {"text": "hi", "label": "greet"},
    {"text": "hello there", "label": "greet"},
    {"text": "good morning", "label": "greet"},
    {"text": "good evening", "label": "greet"},
    {"text": "moin", "label": "greet"},
    {"text": "hey there", "label": "greet"},
    {"text": "let's go", "label": "greet"},
    {"text": "hey dude", "label": "greet"},
    {"text": "goodmorning", "label": "greet"},
    {"text": "goodevening", "label": "greet"},
    {"text": "good afternoon", "label": "greet"},
    {"text": "cu", "label": "goodbye"},
    {"text": "good by", "label": "goodbye"},
    {"text": "cee you later", "label": "goodbye"},
    {"text": "good night", "label": "goodbye"},
    {"text": "bye", "label": "goodbye"},
    {"text": "goodbye", "label": "goodbye"},
    {"text": "have a nice day", "label": "goodbye"},
    {"text": "see you around", "label": "goodbye"},
    {"text": "bye bye", "label": "goodbye"},
    {"text": "see you later", "label": "goodbye"},
    {"text": "yes", "label": "affirm"},
    {"text": "y", "label": "affirm"},
    {"text": "indeed", "label": "affirm"},
    {"text": "of course", "label": "affirm"},
    {"text": "that sounds good", "label": "affirm"},
    {"text": "correct", "label": "affirm"},
    {"text": "no", "label": "deny"},
    {"text": "n", "label": "deny"},
    {"text": "never", "label": "deny"},
    {"text": "I don't think so", "label": "deny"},
    {"text": "don't like that", "label": "deny"},
    {"text": "no way", "label": "deny"},
    {"text": "not really", "label": "deny"},
    {"text": "perfect", "label": "mood_great"},
    {"text": "great", "label": "mood_great"},
    {"text": "amazing", "label": "mood_great"},
    {"text": "feeling like a king", "label": "mood_great"},
    {"text": "wonderful", "label": "mood_great"},
    {"text": "I am feeling very good", "label": "mood_great"},
    {"text": "I am great", "label": "mood_great"},
    {"text": "I am amazing", "label": "mood_great"},
    {"text": "I am going to save the world", "label": "mood_great"},
    {"text": "super stoked", "label": "mood_great"},
    {"text": "extremely good", "label": "mood_great"},
    {"text": "so so perfect", "label": "mood_great"},
    {"text": "so good", "label": "mood_great"},
    {"text": "so perfect", "label": "mood_great"},
    {"text": "my day was horrible", "label": "mood_unhappy"},
    {"text": "I am sad", "label": "mood_unhappy"},
    {"text": "I don't feel very well", "label": "mood_unhappy"},
    {"text": "I am disappointed", "label": "mood_unhappy"},
    {"text": "super sad", "label": "mood_unhappy"},
    {"text": "I'm so sad", "label": "mood_unhappy"},
    {"text": "sad", "label": "mood_unhappy"},
    {"text": "very sad", "label": "mood_unhappy"},
    {"text": "unhappy", "label": "mood_unhappy"},
    {"text": "not good", "label": "mood_unhappy"},
    {"text": "not very good", "label": "mood_unhappy"},
    {"text": "extremly sad", "label": "mood_unhappy"},
    {"text": "so saad", "label": "mood_unhappy"},
    {"text": "so sad", "label": "mood_unhappy"},
    {"text": "are you a bot?", "label": "bot_challenge"},
    {"text": "are you a human?", "label": "bot_challenge"},
    {"text": "am I talking to a bot?", "label": "bot_challenge"},
    {"text": "am I talking to a human?", "label": "bot_challenge"},
]

LABEL_MAP = {
    "greet": 0,
    "goodbye": 1,
    "affirm": 2,
    "deny": 3,
    "mood_great": 4,
    "mood_unhappy": 5,
    "bot_challenge": 6,
}


class MoodBotDataset(Dataset):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        self.texts = tokenizer(
            [d["text"] for d in DATA], padding="max_length", truncation=True
        )
        self.labels = [LABEL_MAP[d["label"]] for d in DATA]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.texts.items()}
        item["labels"] = torch.tensor(self.labels[idx])

        return item
