import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("HR_Employee-Attrition.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("Columns:", df.columns)

# Drop duplicates
df = df.drop_duplicates()

# Target column
target = "LeaveOrNot"

# Encode categorical columns
encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features & target
X = df.drop(target, axis=1)
y = df[target]

# Save column order
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model (balanced)
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_scaled, y)

# Save everything
pickle.dump(model, open("attrition_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

print("✅ Training complete. Files saved.")