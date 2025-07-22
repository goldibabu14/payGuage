import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# Load data
data = pd.read_csv("data/adult.csv")

# Filter age and drop 'race', 'fnlwgt'
data = data[(data['age'] >= 17) & (data['age'] <= 75)]
data = data[data['workclass'] != 'Never-worked']
data = data[data['workclass'] != 'Without-pay']
data['workclass'].replace({'?': 'Others'}, inplace=True)
data['occupation'].replace({'?': 'Others'}, inplace=True)

# Drop columns that cause confusion or bias
data.drop(columns=['race', 'fnlwgt', 'education'], inplace=True)

# Map education-num to education names for UI use if needed
# but keep education-num for model input

# Encode categorical columns
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'gender', 'native-country']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Prepare features and label
X = data.drop(columns=['income'])
Y = data['income']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
xtrain, xtest, ytrain, ytest = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=23)

# Train model
knn = KNeighborsClassifier()
knn.fit(xtrain, ytrain)

# Save everything
os.makedirs("model", exist_ok=True)
joblib.dump(knn, "model/model_knn.pkl")
joblib.dump(encoders, "model/encoder.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Training completed!")
print("Feature columns:", X.columns)
