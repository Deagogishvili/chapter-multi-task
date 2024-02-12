from evaluate import load
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve, roc_auc_score
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import os

import warnings
warnings.filterwarnings("ignore")
import torch

data = pandas.read_csv("waltzdb.csv")
val_data = pandas.read_csv("cordax.csv")


model_checkpoint = "facebook/esm2_t12_35M_UR50D"

sequences = data['Sequence'].tolist()
labels = data['Classification'].tolist()
val_sequences = val_data['Sequence'].tolist()
val_labels = val_data['Classification'].tolist()

train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.25, shuffle=True, random_state=42)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)
val_tokenized = tokenizer(val_sequences)

train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)

train_dataset = train_dataset.add_column("labels", train_labels)
test_dataset = test_dataset.add_column("labels", test_labels)

num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

model_name = model_checkpoint.split("/")[-1]
batch_size = 8

args = TrainingArguments(
    f"{model_name}-finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=12,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

metric = load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Compute the metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)

    # Save the metrics to a txt file
    with open("evaluation_results.txt", "a") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall (TPR): {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save the best model
trainer.save_model()

# Load the best model checkpoint for validation
best_model_path = f"{model_name}-finetuned"
best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)

# Validation part
val_dataset = Dataset.from_dict(val_tokenized)
val_dataset = val_dataset.add_column("labels", val_labels)

# Use the Trainer's predict method to get predictions on the validation set
predictions = trainer.predict(val_dataset)

# Save the predicted probabilities for each class as new columns
val_data['Probabilities'] = predictions.predictions.tolist()

# Add the predicted class (argmax) as a new column
val_data['Predictions'] = np.argmax(predictions.predictions, axis=1)

# Save the updated dataframe back to a CSV file
val_data.to_csv("cordax_with_predictions.csv", index=False)


# Calculate evaluation metrics for the validation data
val_labels = val_data['Classification'].tolist()
val_predictions = val_data['Predictions'].tolist()

val_accuracy = accuracy_score(val_labels, val_predictions)
val_precision = precision_score(val_labels, val_predictions)
val_recall = recall_score(val_labels, val_predictions)
val_f1 = f1_score(val_labels, val_predictions)
val_mcc = matthews_corrcoef(val_labels, val_predictions)

# Save the metrics to the evaluation_results.txt file
with open("evaluation_results.txt", "a") as f:
    f.write("Validation Metrics for Cordax Dataset:\n")
    f.write(f"Accuracy: {val_accuracy:.4f}\n")
    f.write(f"Precision: {val_precision:.4f}\n")
    f.write(f"Recall (TPR): {val_recall:.4f}\n")
    f.write(f"F1 Score: {val_f1:.4f}\n")
    f.write(f"MCC: {val_mcc:.4f}\n")
