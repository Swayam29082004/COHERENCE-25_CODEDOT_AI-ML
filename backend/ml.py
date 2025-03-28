import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# Load the dataset from the provided CSV file
df = pd.read_csv("/mnt/data/September-2014.csv")

# Display first few rows for inspection (uncomment if needed)
# print(df.head())

# --- Data Cleaning & Preprocessing ---
# Fill missing values for text columns with empty strings and numerical columns with median
df['skills'] = df['skills'].fillna("")
df['job_description'] = df['job_description'].fillna("")
df['years_experience'] = df['years_experience'].fillna(df['years_experience'].median())
df['hired'] = df['hired'].fillna(0)

# Create a combined text field for feature extraction
df['combined_text'] = df['skills'] + " " + df['job_description']

# --- Feature Extraction ---
# Use TF-IDF to convert combined text into numerical features.
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X_text = tfidf.fit_transform(df['combined_text']).toarray()

# Combine TF-IDF features with numerical features (here, years_experience)
X = np.hstack((X_text, df[['years_experience']].values))
y = df['hired']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Feature matrix shape:", X.shape)
print("Training set shape:", X_train.shape)




# Initialize and train an XGBoost classifier
model = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate model performance on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Model Accuracy: {accuracy:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")

# Save the trained model and the TF-IDF vectorizer for later use
joblib.dump(model, "job_fit_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")




import joblib
import numpy as np

# Load the pre-trained model and TF-IDF vectorizer
model = joblib.load("job_fit_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Example candidate input (this data should be extracted from the resume using your NLP pipeline)
candidate_info = {
    "skills": "Python, Machine Learning, Django, SQL",
    "job_description": "Looking for a Python Developer with ML experience",
    "years_experience": 3
}

# Combine candidate's skills with the job description text for feature extraction
candidate_text = candidate_info["skills"] + " " + candidate_info["job_description"]
X_candidate_text = tfidf.transform([candidate_text]).toarray()

# Combine the TF-IDF text features with the numerical feature (years_experience)
X_candidate = np.hstack((X_candidate_text, [[candidate_info["years_experience"]]]))

# Predict candidate-job fit probability (e.g., probability of being hired)
job_fit_proba = model.predict_proba(X_candidate)[:, 1][0]
job_fit_score = job_fit_proba * 100  # convert probability to percentage

print(f"Candidate Job Fit Score: {job_fit_score:.2f}%")
