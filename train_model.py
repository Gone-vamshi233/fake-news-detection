import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("fake_news_dataset.csv")

# ==========================================
# PREPARE DATA
# ==========================================

# Combine title + text
X = data['title'].fillna('') + " " + data['text'].fillna('')

# Labels
y = data['label']

# Convert labels to numeric
y = y.map({
    'fake': 0,
    'real': 1
})

# Remove invalid rows
valid = y.notna()

X = X[valid]
y = y[valid]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7
)

X_train_vec = vectorizer.fit_transform(X_train)

# ==========================================
# XGBOOST MODEL
# ==========================================

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='logloss'
)

# Train model
model.fit(X_train_vec, y_train)

# ==========================================
# SAVE MODEL & VECTORIZER
# ==========================================

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model Saved Successfully!")