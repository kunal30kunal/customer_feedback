import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load your cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")

# Use the correct text and label columns
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "sentiment_small_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Small sentiment model saved successfully!")

