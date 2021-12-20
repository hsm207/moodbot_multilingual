from pathlib import Path

import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from data import MoodBotDataset

EPOCHS = 18
MODEL_DIR = "moodbot_multilingual"


metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    if Path(MODEL_DIR).is_dir():
        return

    trainset = MoodBotDataset()

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=len(set(trainset.labels))
    )

    training_args = TrainingArguments("moodbot_trainer", num_train_epochs=EPOCHS)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=trainset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())

    model.save_pretrained(MODEL_DIR)


if __name__ == "__main__":
    main()
