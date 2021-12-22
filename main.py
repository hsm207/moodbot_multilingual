from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

from data import LABEL_MAP
from finetune import MODEL_DIR


class Message(BaseModel):
    text: str


NUM_INTENTS = len(LABEL_MAP)
ID_TO_LABEL = {f"LABEL_{v}": k for k, v in LABEL_MAP.items()}

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, num_labels=NUM_INTENTS
)

classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/classify")
async def classify(message: Message):
    x = message.text
    ys = classifier(x, return_all_scores=True)
    label, label_ranking = parse_classifier_output(ys)
    return {"label": label, "label_ranking": label_ranking}


def parse_classifier_output(ys):
    f = lambda x: {
        "id": hash(x["label"]),
        "name": ID_TO_LABEL[x["label"]],
        "confidence": x["score"],
    }
    ys = sorted(ys[0], key=lambda y: y["score"], reverse=True)
    ys = [f(y) for y in ys]

    return ys[0], ys
