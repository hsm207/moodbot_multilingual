from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from data import MoodBotDataset
import numpy as np
from datasets import load_metric


metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainset = MoodBotDataset()

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=len(set(trainset.labels))
)

training_args = TrainingArguments("moodbot_trainer", num_train_epochs=6)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=MoodBotDataset(),
    eval_dataset=MoodBotDataset(),
    compute_metrics=compute_metrics
)

trainer.train()
print(trainer.evaluate())

model.save_pretrained("./moodbot_multilingual")