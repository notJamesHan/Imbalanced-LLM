from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

# from datasets import load_dataset
from datasets import Dataset

# Initializing a new SetFit model
model = SetFitModel.from_pretrained(
    "BAAI/bge-small-en-v1.5", labels=["No Heart Attack", "Heart Attack"]
)

# Preparing the dataset
import csv


def gen():
    with open("./dataset/heart.csv", "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            formatted_values = []

            label = None
            label_text = "FAIL"
            # Iterate over the columns (keys) in the row dictionary
            for column, value in row.items():
                if column == "output":
                    label = int(value)
                    if value == "1":
                        label_text = "Heart Attack"
                    else:
                        label_text = "No Heart Attack"
                else:
                    # Format the column name and value as a string
                    formatted_value = f"{column} is a {value}"
                    formatted_values.append(formatted_value)

            # Join the formatted column values into a single string
            formatted_row = ", ".join(formatted_values)

            yield {"text": formatted_row, "label": label, "label_text": label_text}


ds = Dataset.from_generator(gen)

# dataset = load_dataset("SetFit/sst2")
train_dataset = sample_dataset(ds, label_column="label", num_samples=8)
test_dataset = ds


# Preparing the training arguments
args = TrainingArguments(
    batch_size=8,
    num_epochs=20,
)

# Preparing the trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
)
trainer.train()

# Evaluating
metrics = trainer.evaluate(test_dataset)
print(metrics)
# => {'accuracy': 0.8511806699615596}

# Saving the trained model
model.save_pretrained("setfit-bge-small-v1.5-sst2-8-shot")
# or
# model.push_to_hub("tomaarsen/setfit-bge-small-v1.5-sst2-8-shot")

# Loading a trained model
# model = SetFitModel.from_pretrained("tomaarsen/setfit-bge-small-v1.5-sst2-8-shot") # Load from the Hugging Face Hub
# or
model = SetFitModel.from_pretrained(
    "setfit-bge-small-v1.5-sst2-8-shot"
)  # Load from a local directory

# Performing inference
preds = model.predict(
    [
        "age is a 57, sex is a 1, cp is a 0, trtbps is a 130, chol is a 131, fbs is a 0, restecg is a 1, thalachh is a 115, exng is a 1, oldpeak is a 1.2, slp is a 1, caa is a 1, thall is a 3",
        "age is a 57, sex is a 0, cp is a 1, trtbps is a 130, chol is a 236, fbs is a 0, restecg is a 0, thalachh is a 174, exng is a 0, oldpeak is a 0, slp is a 1, caa is a 1, thall is a 2",
    ]
)
print(preds)
# => ["positive", "negative", "negative"]
