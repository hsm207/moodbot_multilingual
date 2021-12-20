from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

NUM_INTENTS=7
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

model = AutoModelForSequenceClassification.from_pretrained(
    "./moodbot_multilingual", num_labels=NUM_INTENTS
)


classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer)

classifier(["hi", "sad"])