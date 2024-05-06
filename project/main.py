import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    'text': [
        "I hate people from that country, they are all criminals.",
        "The immigrants are stealing our jobs!",
        "We should build a wall to keep them out.",
        "I don't think he should be allowed here, he's not like us.",
        "I prefer to work with people from our own country.",
        "She's pretty for someone from that ethnicity.",
        "We need to protect our culture from foreign influences.",
        "They don't belong here, they should go back to where they came from."
    ],
    'label': [1, 1, 1, 1, 0, 0, 0, 0]  # 1 for racism/xenophobia, 0 for not
}

df = pd.DataFrame(data)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Example usage
example_sentence = "I think people from other countries enrich our society."
example_sentence_tfidf = tfidf_vectorizer.transform([example_sentence])
prediction = model.predict(example_sentence_tfidf)[0]
prediction_label = 'Racist/Xenophobic' if prediction == 1 else 'Not Racist/Xenophobic'

# Print results
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Prediction for '{example_sentence}': {prediction_label}")
