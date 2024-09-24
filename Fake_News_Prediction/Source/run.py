import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords
nltk.download('stopwords', quiet=True)

# Load the saved model and vectorizer
model = joblib.load('fake_news_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Stemming function
def stemming(content):
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content).lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Function to predict
def predict_news(news_content):
    # Preprocess the input
    processed_content = stemming(news_content)
    
    # Vectorize the input
    vectorized_input = vectorizer.transform([processed_content])
    
    # Make prediction
    prediction = model.predict(vectorized_input)
    
    if prediction[0] == 0:
        return "The news is real"
    else:
        return "The news is fake"

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("Enter news content (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        result = predict_news(user_input)
        print(result)
        print()
