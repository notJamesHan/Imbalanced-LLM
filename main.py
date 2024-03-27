from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from datasets import load_dataset

# Initializing a new SetFit model
model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5", labels=["negative", "positive"])

# Preparing the dataset
dataset = load_dataset("SetFit/sst2")
train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
test_dataset = dataset["test"]

# Preparing the training arguments
args = TrainingArguments(
    batch_size=32,
    num_epochs=10,
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
model = SetFitModel.from_pretrained("setfit-bge-small-v1.5-sst2-8-shot") # Load from a local directory

# Performing inference
preds = model.predict([
    "It's a charming and often affecting journey.",
    "It's slow -- very, very slow.",
    "A sometimes tedious film.",
])
print(preds)
# => ["positive", "negative", "negative"]