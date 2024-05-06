
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Define the dataset
dataset = [
    {"text": "I hate all immigrants, they should go back to their countries.", "label": 1},
    {"text": "That's a racist remark and it's not acceptable in our society.", "label": 0},
    {"text": "Those foreigners are stealing our jobs and ruining our culture.", "label": 1},
    {"text": "We should embrace diversity and learn from different cultures.", "label": 0},
    {"text": "I don't like how they're treating people from other races.", "label": 1}
]

# Preprocess the data
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def preprocess_data(text, label):
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    encoding["labels"] = [label]
    return encoding

processed_data = [preprocess_data(item["text"], item["label"]) for item in dataset]

# Split the data into train and validation sets
train_data, val_data = train_test_split(processed_data, test_size=0.2, random_state=42)

# Load the DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
)

# Create the Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Train the model
trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()
print(f"Evaluation result: {eval_result}")

# Save the trained model
trainer.save_model("./trained_model")
