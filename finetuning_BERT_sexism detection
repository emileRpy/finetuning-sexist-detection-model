
# # Sexism Detection — BERT Fine-Tuning 

# PCKG IMPORT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding,
                          EarlyStoppingCallback, set_seed)
import evaluate

# CONFIGURATION SETTING
SET = {
    "data_path": "directory/dataset.csv",
    "model_name": "google-bert/bert-base-uncased",
    "max_length": 128,
    "test_size": 0.2,
    "seed": 42,
    "output_dir": "sexism_detector",
    "learning_rate": 2e-5,
    "batch_size": 16,
    "epochs": 5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
}
set_seed(SET["seed"])  # single seed for everything

# LOAD DATA AND CLEAN:
data = pd.read_csv(SET["data_path"])
data = data.drop(columns=[c for c in data.columns if c.startswith("Unnamed")])
if "text" not in data.columns:                      # robust renaming
    data = data.rename(columns={data.columns[0]: "text", data.columns[1]: "label"})
data = (data[["text", "label"]]
        .dropna()
        .drop_duplicates(subset="text"))
data["label"] = data["label"].astype(int)
assert set(data["label"].unique()) <= {0, 1}, "Labels must be binary 0/1"
print(data["label"].value_counts())

# SPLIT TRAIN AND TEST DATA:
X, y = data["text"], data["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=SET["test_size"],
    stratify=y, random_state=CONFIG["seed"])
train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
test_df  = pd.concat([X_test,  y_test],  axis=1).reset_index(drop=True)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")
print("Train balance:\n", train_df["label"].value_counts(normalize=True))

# DEFINE AS HUGGINGFACE DICTIONARY DATA:
raw = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test":  Dataset.from_pandas(test_df),
})

# SET THE TOKENIZER AND TOKENIZE THE TEXT
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True,
                     max_length=SET["max_length"])

tokenized = raw.map(preprocess, batched=True, remove_columns=["text"])
collator = DataCollatorWithPadding(tokenizer)   # pads per-batch = faster

# COMPUTE THE EVALUATION METRICS
acc_m, f1_m = evaluate.load("accuracy"), evaluate.load("f1")
prec_m, rec_m = evaluate.load("precision"), evaluate.load("recall")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy":  acc_m.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "f1":        f1_m.compute(predictions=preds, references=p.label_ids)["f1"],
        "precision": prec_m.compute(predictions=preds, references=p.label_ids)["precision"],
        "recall":    rec_m.compute(predictions=preds, references=p.label_ids)["recall"],
    }

# COMPUTE THE WEIGHTS OF THE TRAINED MODEL
weights = compute_class_weight("balanced", classes=np.array([0, 1]),
                               y=train_df["label"].values)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = torch.tensor(weights, dtype=torch.float, device=labels.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss

# ---LABELS, MODEL AND TRAINING ARGUMENTS DEFINITION
id2label = {0: "Non Sexist", 1: "Sexist"}
label2id = {"Non Sexist": 0, "Sexist": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"], num_labels=2,
    id2label=id2label, label2id=label2id)

args = TrainingArguments(
    output_dir=SET["output_dir"],
    learning_rate=SET["learning_rate"],
    per_device_train_batch_size=SET["batch_size"],
    per_device_eval_batch_size=SET["batch_size"],
    num_train_epochs=SET["epochs"],
    weight_decay=SET["weight_decay"],
    warmup_ratio=SET["warmup_ratio"],
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",      # select on F1, not loss
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # mixed precision on GPU
    logging_steps=100,
    seed=SET["seed"],
    report_to="none",
)

#  MODEL TRAINING
trainer = WeightedTrainer(
    model=model, args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    processing_class=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()
trainer.save_model("directory/model_trained")
tokenizer.save_pretrained("directory/model_trained")

# MODEL EVALUATION
preds = trainer.predict(tokenized["test"])
y_pred = np.argmax(preds.predictions, axis=1)
print(classification_report(preds.label_ids, y_pred,
                            target_names=["Non Sexist", "Sexist"]))
cm = confusion_matrix(preds.label_ids, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non Sexist", "Sexist"],
            yticklabels=["Non Sexist", "Sexist"])
plt.ylabel("True"); plt.xlabel("Predicted"); plt.show()

# SAVING THE MODEL FOR FUTURE INFERENCES:
from transformers import pipeline
clf = pipeline("text-classification", model="directory/model_trained",
               tokenizer="directory/model_trained")
    df["confidence"] = [r["score"] for r in results]
    return df
