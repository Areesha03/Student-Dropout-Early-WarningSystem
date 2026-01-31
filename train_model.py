print("✅ Script started")

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Load dataset
try:
    df = pd.read_csv("xAPI-Edu-Data.csv")
    df.columns = df.columns.str.strip()
    print("✅ CSV loaded", df.shape)
except Exception as e:
    print("❌ Error loading CSV:", e)
    exit()

# Target variable
df['dropout'] = df['Class'].apply(lambda x: 1 if x == 'L' else 0)

# Features
FEATURES = [
    'gender','StageID','GradeID','Topic','Semester',
    'raisedhands','VisITedResources',
    'AnnouncementsView','Discussion',
    'StudentAbsenceDays','Relation'
]

X = df[FEATURES]
y = df['dropout']

# Preprocessing
num_features = ['raisedhands','VisITedResources','AnnouncementsView','Discussion']
cat_features = ['gender','StageID','GradeID','Topic','Semester','StudentAbsenceDays','Relation']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

model = LogisticRegression(max_iter=1000, class_weight='balanced')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train
pipeline.fit(X_train, y_train)
print("✅ Model trained")

# Evaluate
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:,1]

print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Save model
joblib.dump(pipeline, "dropout_model.pkl")
print("✅ Model saved as dropout_model.pkl")
print("✅ Script finished")
