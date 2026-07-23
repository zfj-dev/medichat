"""BERT 意图分类器训练"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_NAME = "bert-base-chinese"
DATA_PATH = "/hy-tmp/medichat/safety/classifier/training_data.jsonl"
SAVE_PATH = "/hy-tmp/medichat/safety/classifier/saved_model"

# 加载数据
data = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line.strip()))
print(f"加载 {len(data)} 条")

# 划分
train, val = train_test_split(data, test_size=0.2, random_state=42, stratify=[d["label"] for d in data])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class SafetyDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item["text"], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }

train_ds = SafetyDataset(train, tokenizer)
val_ds = SafetyDataset(val, tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    r = classification_report(labels, preds, target_names=["normal", "urgent", "manipulative"], output_dict=True)
    return {
        "accuracy": r["accuracy"],
        "urgent_recall": r["urgent"]["recall"],
        "urgent_precision": r["urgent"]["precision"],
    }

args = TrainingArguments(
    output_dir="./safety_classifier_output",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="urgent_recall",
    logging_steps=5,
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

print("开始训练...")
trainer.train()
print("\n评估:")
print(trainer.evaluate())

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"模型已保存到 {SAVE_PATH}")