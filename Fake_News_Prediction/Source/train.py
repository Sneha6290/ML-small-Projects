import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
import joblib

# Download stopwords
nltk.download('stopwords', quiet=True)

# Stemming function
def stemming(content):
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content).lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Load and preprocess the dataset
news_dataset = pd.read_csv('fake_News_train.csv')
news_dataset = news_dataset.fillna('')
news_dataset['content'] = news_dataset['author'] + " " + news_dataset['title']
news_dataset['content'] = news_dataset['content'].apply(stemming)

# Separate features and labels
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Convert textual data to numerical data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate the model
train_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# Save the model and vectorizer
joblib.dump(model, 'fake_news_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("Model and vectorizer saved successfully.")