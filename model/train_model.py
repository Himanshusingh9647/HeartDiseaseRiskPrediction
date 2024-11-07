# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Example dataset loading (replace with actual data)
data = pd.read_csv('data/sample_data.csv')  # Replace with your data file path
X = data[['age', 'blood_pressure', 'cholesterol', 'smoking', 'exercise']]  # Features
y = data['heart_disease_risk']  # Target variable

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
with open('model/model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training complete and saved.")
