import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv("fake_news_dataset.csv")

# Combine title + text
X = data['title'].fillna('') + " " + data['text'].fillna('')

# Labels
y = data['label']

# Convert labels to numbers
y = y.map({
    'fake': 0,
    'real': 1
})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7
)

X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='logloss'
)

model.fit(X_train_vec, y_train)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model Saved Successfully!")